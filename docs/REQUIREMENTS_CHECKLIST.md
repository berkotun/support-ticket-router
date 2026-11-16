# Requirements checklist (SWE015 homework → this repository)

## Use Case 2 – required components

| Requirement (homework text) | Implementation |
|---|---|
| take a free-text message → normalize it | `src/normalize.py` → `normalize()`, used by `src/preprocess.py` |
| predict the intent with a confidence score → show the routed team | `app.py` decision card (intent, confidence, team), `src/routing.py`, `src/domains.py` |
| low-confidence or out-of-scope messages go to "human agent" | `src/routing.py` → threshold τ per model (tuned on validation), `oos` label |
| greetings and small talk get a rule-based reply | `src/eliza.py` (ELIZA regex rules + reflection), used in `src/routing.py` |
| predictions of all four models shown side by side | `app.py` → "All four models side by side" table |
| **B:** regex normalization and model-code extraction | `src/normalize.py` → `normalize()`, `extract_codes()` |
| **B:** edit-distance correction of misspelled product names | `src/spell.py` → `edit_distance()`, `SpellCorrector` (+ `PRODUCTS` catalog) |
| **B:** ELIZA-style replies for small talk | `src/eliza.py` |
| **B:** tf-idf + logistic regression baseline | `src/train_tfidf.py` |
| **A:** feedforward network on averaged word2vec / fastText embeddings, SGD + backpropagation | `src/train_fasttext_ffn.py` (own fastText, `torch.optim.SGD`, `loss.backward()`) |
| **C:** BERT fine-tuned for intent classification | `src/train_teacher.py`, `src/transformer_utils.py` |
| **D:** knowledge distillation into a smaller student, then quantization | `src/distill.py` (BERT-base → DistilBERT), `src/quantize.py` (dynamic INT8) |
| Metrics: accuracy, out-of-scope recall | `src/benchmark.py` → in-scope accuracy, OOS recall (+ routing accuracy) |

## R1–R8

| | Requirement | Evidence |
|---|---|---|
| R1 | interactive app accepting new typed input | `app.py` (Gradio), `python app.py` |
| R2 | ≥ 1 technique from each area A–D + technique map slide | README "Technique map" (copy to slide) |
| R3 | four models on the same held-out test set | `src/benchmark.py` evaluates every model on `test.csv` (official CLINC150 test) |
| R4 | results table: metric, size (MB), latency (ms, ≥ 100 runs, hardware) | `results/results_table.md` (200 runs, hardware line) |
| R5 | app runs the quantized Transformer | `app.py` → `ROUTER = "student_int8"` |
| R6 | architecture diagram matching the code | README "Architecture" (Mermaid) |
| R7 | repo with README, single run command, requirements.txt, no large weights | README, `python app.py`, `requirements.txt`, `.gitignore` excludes `models/`; weights on the Hugging Face Hub |
| R8 | AI usage slide | presentation |

## Gate checks before submission

- [ ] Presentation PDF (8–12 slides): first slide has name, student ID, use case, repo link and video link
- [ ] Slides follow Introduction / Development / Conclusion
- [ ] Technique map, architecture diagram, results table, ≥ 1 failure example, AI usage slide
- [ ] Every number in the slides comes from `results/` (own runs)
- [ ] Video ≤ 2:00, narrated, shows the live app with ≥ 2 new inputs (one difficult), plus the architecture and the results table
- [ ] Video link (YouTube unlisted / Drive "anyone with the link") and repo link open in a private browser window
- [ ] Hugging Face model repo is public; `python app.py` works in a fresh folder following the README
