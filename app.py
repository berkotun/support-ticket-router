"""NovaAssist Support Ticket Router - Gradio app.

For every free-text message the app shows:
  1. preprocessing: normalized text, spelling corrections, extracted codes
  2. the routing decision of the quantized distilled model (DistilBERT INT8):
     intent, confidence and team; low-confidence / out-of-scope -> "Human agent";
     greetings and small talk -> an ELIZA-style rule-based reply
  3. the predictions of all four models side by side, with response times
Everything runs on CPU in this single Python process.

Run:  python app.py   ->  http://127.0.0.1:7860
"""
import html
import time

import gradio as gr
import pandas as pd
import torch

from src import config

torch.set_grad_enabled(False)


def ensure_models():
    """Download the trained models from the Hugging Face Hub on the first run."""
    if (config.MODELS / "student_int8").exists():
        return
    if config.MODELS_REPO.startswith("YOUR_HF_USERNAME"):
        raise SystemExit("Models not found. Set MODELS_REPO in src/config.py "
                         "(or the MODELS_REPO environment variable) to your Hugging Face repo.")
    from huggingface_hub import snapshot_download
    print(f"Downloading models: {config.MODELS_REPO} -> {config.MODELS}")
    snapshot_download(config.MODELS_REPO, local_dir=config.MODELS)


ensure_models()
from src.models import load_all  # noqa: E402  (after the models are available)
from src.preprocess import analyze  # noqa: E402
from src.routing import decide, load_thresholds  # noqa: E402

LABELS, MODELS = load_all(for_app=True)
TAUS = load_thresholds()
ROUTER = "student_int8"          # the quantized Transformer makes the decision (R5)
HISTORY_COLS = ["#", "Message", "Intent", "Confidence", "Routed to", "Latency (ms)"]


def run_model(key, text):
    t0 = time.perf_counter()
    p = MODELS[key].probs([text])[0]
    ms = (time.perf_counter() - t0) * 1000
    return decide(text, p, LABELS, TAUS.get(key, 0.5)), ms


def decision_card(d, ms):
    human = d["team"] == "Human agent"
    color = "#fff4e5" if human else "#e8f5e9"
    reply = (f"<div style='margin-top:8px'>💬 <b>Bot reply:</b> {html.escape(d['reply'])}</div>"
             if d["reply"] else "")
    reason = ""
    if human:
        reason = ("predicted out-of-scope" if d["intent"] == config.OOS else
                  f"confidence {d['conf']:.0%} is below the threshold "
                  f"{TAUS.get(ROUTER, 0.5):.0%}")
        reason = f"<div style='font-size:13px'>Reason: {reason}</div>"
    return (f"<div style='background:{color};color:#111;padding:14px;border-radius:10px'>"
            f"<div style='font-size:13px;opacity:.7'>Decision by DistilBERT student · INT8 · "
            f"{ms:.1f} ms on CPU</div>"
            f"<div style='font-size:24px;font-weight:700'>➡ {html.escape(d['team'])}</div>"
            f"<div>Intent: <code>{html.escape(d['intent'])}</code> · confidence "
            f"<b>{d['conf']:.1%}</b></div>{reason}{reply}</div>")


def preprocessing_md(a):
    fixes = ", ".join(f"`{w}` → `{c}`" for w, c in a["fixes"]) or "none"
    codes = ", ".join(f"**{k}**: {', '.join(v)}" for k, v in a["codes"].items()) or "none"
    return (f"**Normalized:** `{a['normalized']}`  \n**Spelling corrections (edit distance):** "
            f"{fixes}  \n**Extracted codes (regex):** {codes}")


def route(text, history):
    text = (text or "").strip()
    if not text:
        return gr.update(), gr.update(), gr.update(), history
    a = analyze(text)
    rows, router = [], None
    for key, m in MODELS.items():
        d, ms = run_model(key, text)
        if key == ROUTER:
            router = (d, ms)
        rows.append([m.name, m.precision, d["intent"], round(d["conf"], 3), d["team"], round(ms, 2)])
    side = pd.DataFrame(rows, columns=["Model", "Precision", "Predicted intent", "Confidence",
                                       "Would route to", "Latency (ms)"])
    d, ms = router
    history = pd.concat([history, pd.DataFrame(
        [[len(history) + 1, text, d["intent"], round(d["conf"], 3), d["team"], round(ms, 2)]],
        columns=HISTORY_COLS)], ignore_index=True)
    return decision_card(d, ms), preprocessing_md(a), side, history


EXAMPLES = [
    "my mastercrad was declined at the store, card ending in 4321",   # typo + code
    "how much do i have in my savings account?",
    "is flight UA 1234 delayed?",
    "my dishwasher DW-6534 won't drain",                                # out-of-scope for CLINC150
    "hello there!",                                                      # ELIZA reply
    "can you tell me how to fix a flat tire",
    "what is the best pizza topping in your opinion",                   # hard case
]

with gr.Blocks(title="NovaAssist Support Ticket Router") as demo:
    gr.Markdown("## 📨 NovaAssist Support Ticket Router\n"
                "Routes English support messages to the right team (CLINC150: 150 intents, "
                "10 product areas). Runs on CPU. Routing decision: **distilled + INT8 quantized "
                "DistilBERT**.")
    history_state = gr.State(pd.DataFrame(columns=HISTORY_COLS))
    with gr.Row():
        with gr.Column(scale=3):
            box = gr.Textbox(label="Customer message", placeholder="Type a support message…",
                             lines=2)
            btn = gr.Button("Route message", variant="primary")
            gr.Examples(EXAMPLES, inputs=box, label="Example messages")
        with gr.Column(scale=3):
            card = gr.HTML("<div style='opacity:.6'>The routing decision appears here.</div>")
            prep = gr.Markdown()
    side = gr.Dataframe(label="All four models side by side", interactive=False)
    hist = gr.Dataframe(pd.DataFrame(columns=HISTORY_COLS), label="Message history",
                        interactive=False)

    ins, outs = [box, history_state], [card, prep, side, history_state]
    box.submit(route, ins, outs).then(lambda h: h, history_state, hist)
    btn.click(route, ins, outs).then(lambda h: h, history_state, hist)

if __name__ == "__main__":
    demo.launch()
