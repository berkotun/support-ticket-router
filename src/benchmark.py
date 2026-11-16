"""Evaluate every model on the SAME held-out test set and build the results table (R3, R4).

Per model:
  - in-scope accuracy        : correct intent on the 4,500 in-scope test messages
  - out-of-scope recall      : share of the 1,000 out-of-scope test messages sent to a human
  - routing accuracy         : correct team (or human agent) on all 5,500 test messages
  - the same two task metrics on the typo test set (test_typos.csv)
  - size (MB)                : model files on disk
  - latency (ms / message)   : CPU, batch size 1, mean and p95 over 200 test
                               messages after 10 warm-up runs, end to end
                               (normalization + spelling correction + model)

The out-of-scope threshold tau of each model is chosen on the VALIDATION set;
the test set is used only for the final numbers.

Usage:   python -m src.benchmark
Output:  results/results_table.md, results_table.csv, failures.csv, hardware.txt
         models/thresholds.json
"""
import json
import platform
import time

import numpy as np
import pandas as pd
import torch

from . import config
from .domains import HUMAN, team_of
from .models import load_all
from .prepare_data import load_split
from .routing import THRESHOLDS_FILE, final_intents

N_LATENCY = 200   # requirement: >= 100 runs
N_WARMUP = 10
TAUS = np.round(np.arange(0.0, 0.96, 0.05), 2)


def hardware_info():
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return (f"CPU: {cpu} | torch threads: {torch.get_num_threads()} | "
            f"Python {platform.python_version()} | torch {torch.__version__} | "
            f"{platform.system()} {platform.release()}")


def task_metrics(gold, pred):
    gold, pred = np.array(gold), np.array(pred)
    ins, oos = gold != config.OOS, gold == config.OOS
    team = lambda xs: np.array([HUMAN if x == config.OOS else team_of(x) for x in xs])
    return {"in_scope_acc": float(np.mean(pred[ins] == gold[ins])),
            "oos_recall": float(np.mean(pred[oos] == config.OOS)) if oos.any() else float("nan"),
            "routing_acc": float(np.mean(team(pred) == team(gold)))}


def tune_tau(probs, gold, labels):
    best = None
    for tau in TAUS:
        m = task_metrics(gold, final_intents(probs, labels, tau))
        score = (m["in_scope_acc"] + m["oos_recall"]) / 2
        if best is None or score > best[0]:
            best = (score, float(tau))
    return best[1]


def latency_ms(model, texts):
    for t in texts[:N_WARMUP]:
        model.probs([t])
    times = []
    for t in texts[:N_LATENCY]:
        t0 = time.perf_counter()
        model.probs([t])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def main():
    torch.manual_seed(config.SEED)
    val, test, typos = load_split("val"), load_split("test"), load_split("test_typos")
    labels, models = load_all()
    hw = hardware_info()
    print(hw)
    rng = np.random.RandomState(config.SEED)
    lat_texts = list(test.text.iloc[rng.permutation(len(test))])

    rows, taus, preds = [], {}, {"text": test.text, "gold": test.intent}
    for key, m in models.items():
        print(f"evaluating: {m.name}")
        tau = tune_tau(m.probs(val.text), val.intent, labels)
        taus[key] = tau
        p_test = final_intents(m.probs(test.text), labels, tau)
        p_typo = final_intents(m.probs(typos.text), labels, tau)
        mt, my = task_metrics(test.intent, p_test), task_metrics(typos.intent, p_typo)
        mean_ms, p95_ms = latency_ms(m, lat_texts)
        preds[key], preds[key + "_typos"] = p_test, p_typo
        rows.append({
            "Model": m.name,
            "Precision": m.precision,
            "In-scope accuracy": round(mt["in_scope_acc"], 4),
            "Out-of-scope recall": round(mt["oos_recall"], 4),
            "Routing accuracy": round(mt["routing_acc"], 4),
            "In-scope acc. (typo test)": round(my["in_scope_acc"], 4),
            "OOS recall (typo test)": round(my["oos_recall"], 4),
            "Threshold tau (from val)": tau,
            "Size (MB)": round(m.size_mb(), 1),
            "Latency mean (ms)": round(mean_ms, 2),
            "Latency p95 (ms)": round(p95_ms, 2),
        })

    config.RESULTS.mkdir(exist_ok=True)
    (config.MODELS / THRESHOLDS_FILE).write_text(json.dumps(taus, indent=1))
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS / "results_table.csv", index=False)
    md = table.to_markdown(index=False)
    n_oos = int((test.intent == config.OOS).sum())
    (config.RESULTS / "results_table.md").write_text(
        f"# Results\n\nTest set: official CLINC150 (plus) test set, {len(test)} messages "
        f"({len(test) - n_oos} in-scope, {n_oos} out-of-scope). Typo test: the same messages "
        f"with keyboard typos (src/typos.py, seed {config.SEED}).\n\n"
        f"Hardware: {hw}\n\nLatency: batch size 1, mean over {N_LATENCY} messages after "
        f"{N_WARMUP} warm-up runs, end to end (normalization + spelling correction + model).\n\n{md}\n")
    (config.RESULTS / "hardware.txt").write_text(hw + "\n")
    pd.DataFrame(preds).to_csv(config.RESULTS / "predictions.csv", index=False)

    # For the "failure example" slide: errors of the model used in the app
    err = pd.DataFrame({"text": test.text, "typo_text": typos.text, "gold": test.intent,
                        "pred_student_int8": preds["student_int8"],
                        "pred_on_typo_text": preds["student_int8_typos"]})
    err[err.gold != err.pred_student_int8].to_csv(config.RESULTS / "failures.csv", index=False)

    print("\n" + md)


if __name__ == "__main__":
    main()
