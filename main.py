"""
main.py
─────────────────────────────────────────────────────────────────────────────
Clinical NER pipeline:  train → infer → benchmark → HTML + CSV report.

To swap the backbone model, change BASE_MODEL in config.py. That's it.
"""

import re
import io
import csv
import base64
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from spacy import displacy
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
)
from tqdm import tqdm
import evaluate

from config import (
    BASE_MODEL, DATASET_NAME, YOUR_MODEL_PATH, PRETRAINED_BASELINE,
    DATASET_PERCENTAGE, RANDOM_DATASET_VIZ_COUNT,
    LABEL_LIST, ID2LABEL, LABEL2ID,
    CUSTOM_SENTENCES, COLORS,USE_CRF
)
from ner_model import TransformerNERWithCRF


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def align_labels_with_tokens(labels, word_ids):
    new_labels, current_word = [], None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            new_labels.append(-100 if word_id is None else labels[word_id])
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            new_labels.append(label + 1 if label % 2 == 1 else label)  # B → I
    return new_labels


def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    tag_key    = "ner_tags" if "ner_tags" in examples else "tags"
    tokenized["labels"] = [
        align_labels_with_tokens(labels, tokenized.word_ids(batch_index=i))
        for i, labels in enumerate(examples[tag_key])
    ]
    return tokenized


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_new_model():
    print(f"\n[TRAIN] Starting training pipeline using '{BASE_MODEL}' …")

    raw_datasets = load_dataset(DATASET_NAME)
    tokenizer    = AutoTokenizer.from_pretrained(BASE_MODEL, add_prefix_space=True)

    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    print("[TRAIN] Initialising TransformerNERWithCRF …")
    model = TransformerNERWithCRF.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        use_crf=USE_CRF,
    )

    args = TrainingArguments(
        output_dir=YOUR_MODEL_PATH,
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="no",
        remove_unused_columns=False,
    )

    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        emissions   = torch.tensor(p.predictions, device=model.device)
        labels_arr  = p.label_ids

        # Handle decoding based on the model mode
        if getattr(model, "use_crf", True):
            model._apply_transition_constraints()
            with torch.no_grad():
                best_tags_list = model.crf.decode(emissions)
        else:
            best_tags_list = emissions.argmax(dim=-1).tolist()

        true_preds, true_labels = [], []
        for pred_seq, label_seq in zip(best_tags_list, labels_arr):
            preds_row, labels_row = [], []
            for p_tag, l_tag in zip(pred_seq, label_seq):
                if l_tag != -100:
                    preds_row.append(LABEL_LIST[p_tag])
                    labels_row.append(LABEL_LIST[l_tag])
            true_preds.append(preds_row)
            true_labels.append(labels_row)

        results = metric.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall":    results["overall_recall"],
            "f1":        results["overall_f1"],
            "accuracy":  results["overall_accuracy"],
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(YOUR_MODEL_PATH)
    tokenizer.save_pretrained(YOUR_MODEL_PATH)
    print(f"[TRAIN] Model saved to {YOUR_MODEL_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE UTILS
# ══════════════════════════════════════════════════════════════════════════════

def extract_dosages(text):
    pattern = r"\b\d+\.?\d*\s?(mg|g|ml|mcg|units?)\b"
    return [
        {"start": m.start(), "end": m.end(), "label": "DOSAGE", "text": m.group()}
        for m in re.finditer(pattern, text, re.IGNORECASE)
    ]


def tokens_to_text_and_entities(tokens, tags):
    text, entities, current_ent = " ".join(tokens), [], None
    char_pos = 0
    for token, tag_id in zip(tokens, tags):
        start, end   = char_pos, char_pos + len(token)
        label_full   = ID2LABEL.get(tag_id, "O")
        label_type   = label_full[2:].upper() if "-" in label_full else None

        if label_full.startswith("B-"):
            if current_ent: entities.append(current_ent)
            current_ent = {"start": start, "end": end, "label": label_type, "text": token}

        elif label_full.startswith("I-") and current_ent:
            if label_type == current_ent["label"]:
                current_ent["end"]   = end
                current_ent["text"] += " " + token
            else:
                entities.append(current_ent)
                current_ent = {"start": start, "end": end, "label": label_type, "text": token}
        else:
            if current_ent: entities.append(current_ent)
            current_ent = None

        char_pos = end + 1

    if current_ent:
        entities.append(current_ent)
    return text, entities


def predict_with_custom_model(text, model, tokenizer):
    """Viterbi decode with our CRF model, return entity spans."""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt",
                       return_offsets_mapping=True, truncation=True)
    offsets = inputs.pop("offset_mapping")[0].tolist()
    inputs  = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        predicted_ids = model.decode(inputs["input_ids"], inputs["attention_mask"])[0]

    entities, current_ent = [], None
    for tag_id, (start, end) in zip(predicted_ids, offsets):
        if start == end:       # special token
            continue
        label_full = model.config.id2label[tag_id]
        label_type = label_full[2:] if "-" in label_full else None

        if label_full == "O":
            if current_ent: entities.append(current_ent); current_ent = None

        elif label_full.startswith("B-"):
            if current_ent: entities.append(current_ent)
            current_ent = {"start": start, "end": end,
                           "label": label_type, "text": text[start:end]}

        elif label_full.startswith("I-"):
            if current_ent and current_ent["label"] == label_type:
                current_ent["end"]  = end
                current_ent["text"] = text[current_ent["start"]:end]
            else:
                if current_ent: entities.append(current_ent)
                current_ent = {"start": start, "end": end,
                               "label": label_type, "text": text[start:end]}

    if current_ent:
        entities.append(current_ent)
    return entities


def predict_with_pipeline(text, pipe):
    """Inference via a standard HuggingFace pipeline (baseline model)."""
    return [
        {"start": p["start"], "end": p["end"],
         "label": p["entity_group"].upper(),
         "text":  text[p["start"]:p["end"]]}
        for p in pipe(text)
        if p["entity_group"].upper() in {"CHEMICAL", "DISEASE"}
    ]


def merge_dosages(entities, text):
    """Append regex-detected dosage spans, avoiding overlaps."""
    existing = [(e["start"], e["end"]) for e in entities]
    extras   = [
        d for d in extract_dosages(text)
        if not any(not (d["end"] <= s or d["start"] >= e) for s, e in existing)
    ]
    return sorted(entities + extras, key=lambda x: x["start"])


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _overlaps(e1, e2):
    return (e1["label"] == e2["label"] and
            max(e1["start"], e2["start"]) < min(e1["end"], e2["end"]))


def accumulate_stats(gt_ents, pred_ents, stats):
    matched_gt, matched_pred = set(), set()
    for gi, gt in enumerate(gt_ents):
        for pi, pred in enumerate(pred_ents):
            if pi in matched_pred: continue
            if _overlaps(gt, pred):
                stats[gt["label"]]["tp"] += 1
                matched_gt.add(gi); matched_pred.add(pi); break
    for gi, gt  in enumerate(gt_ents):
        if gi not in matched_gt:   stats[gt["label"]]["fn"]   += 1
    for pi, pred in enumerate(pred_ents):
        if pi not in matched_pred: stats[pred["label"]]["fp"] += 1
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def _p_r_f1(s):
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def save_csv_report(folder, stats_ours, stats_pre):
    path = folder / "statistics.csv"
    with open(path, mode="w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Label", "TP", "FP", "FN", "Precision", "Recall", "F1"])
        for label in ["CHEMICAL", "DISEASE"]:
            for name, stats in [("Custom CRF", stats_ours), ("Pretrained", stats_pre)]:
                s        = stats[label]
                p, r, f1 = _p_r_f1(s)
                w.writerow([name, label, s["tp"], s["fp"], s["fn"],
                             f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])
    print(f"CSV saved → {path}")


def create_plots(stats_ours, stats_pre):
    label_keys = ["CHEMICAL", "DISEASE"]

    # F1 comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x, w = np.arange(len(label_keys)), 0.35
    ax.bar(x - w/2, [_p_r_f1(stats_ours[l])[2] for l in label_keys], w,
           label="Your CRF Model", color="#6c5ce7")
    ax.bar(x + w/2, [_p_r_f1(stats_pre[l])[2]  for l in label_keys], w,
           label="Pretrained RoBERTa", color="#0984e3")
    ax.set(ylabel="F1 Score", title="Model F1 Comparison",
           xticks=x, xticklabels=label_keys)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0); plot1 = base64.b64encode(buf.read()).decode(); plt.close()

    # Error distribution stacked bar
    fig, ax = plt.subplots(figsize=(8, 5))
    models  = ["Your CRF Model", "Pretrained RoBERTa"]
    fps = [sum(stats_ours[l]["fp"] for l in label_keys),
           sum(stats_pre[l]["fp"]  for l in label_keys)]
    fns = [sum(stats_ours[l]["fn"] for l in label_keys),
           sum(stats_pre[l]["fn"]  for l in label_keys)]
    ax.bar(models, fns, label="Missed Entity (FN)", color="#ff7675")
    ax.bar(models, fps, bottom=fns, label="Hallucination (FP)", color="#ffeaa7")
    ax.set(ylabel="Total Errors", title="Error Distribution"); ax.legend()
    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0); plot2 = base64.b64encode(buf.read()).decode(); plt.close()

    return plot1, plot2


def generate_html_report(custom_rows, dataset_rows, stats_ours, stats_pre,
                         sample_size, plot1, plot2):
    def macro(stats):
        tp = sum(stats[l]["tp"] for l in ["CHEMICAL", "DISEASE"])
        fp = sum(stats[l]["fp"] for l in ["CHEMICAL", "DISEASE"])
        fn = sum(stats[l]["fn"] for l in ["CHEMICAL", "DISEASE"])
        s  = {"tp": tp, "fp": fp, "fn": fn}
        return _p_r_f1(s)

    op, or_, of1 = macro(stats_ours)
    pp, pr,  pf1 = macro(stats_pre)

    def viz_cards(rows):
        html = ""
        for row in rows:
            html += (
                f'<div class="viz-card">'
                f'<div class="viz-header">'
                f'<div class="viz-col-header" style="color:#6c5ce7">Your Model ({BASE_MODEL})</div>'
                f'<div class="viz-col-header" style="color:#0984e3">Pretrained Baseline</div>'
                f'</div>'
                f'<div class="viz-content">'
                f'<div class="viz-col">{row["html_ours"]}</div>'
                f'<div class="viz-col">{row["html_pre"]}</div>'
                f'</div></div>'
            )
        return html

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Clinical NER Report</title>
<style>
  body{{font-family:'Segoe UI',Roboto,sans-serif;background:#f4f6f9;margin:0;padding:0;color:#333}}
  .header{{background:linear-gradient(135deg,#6c5ce7,#a29bfe);color:#fff;padding:40px 20px;
           text-align:center;margin-bottom:40px;box-shadow:0 4px 6px rgba(0,0,0,.1)}}
  .header h1{{margin:0;font-size:2.5em}} .header p{{margin-top:10px;font-size:1.2em;opacity:.9}}
  .container{{max-width:1100px;margin:0 auto;padding:0 20px}}
  .card{{background:#fff;padding:25px;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.05);margin-bottom:40px}}
  table{{width:100%;border-collapse:collapse;text-align:center}}
  th{{background:#f8f9fa;color:#555;font-weight:600;padding:12px;border-bottom:2px solid #eee}}
  td{{padding:12px;border-bottom:1px solid #eee;font-size:1.05em}}
  .highlight{{color:#2d3436;font-weight:bold}}
  .charts-row{{display:flex;gap:20px;margin-bottom:40px;flex-wrap:wrap}}
  .chart-box{{flex:1;background:#fff;padding:20px;border-radius:12px;
              box-shadow:0 2px 10px rgba(0,0,0,.05);text-align:center;min-width:300px}}
  .chart-box img{{max-width:100%;height:auto}}
  .section-title{{font-size:1.5em;margin-bottom:20px;color:#2d3436;
                  border-left:5px solid #6c5ce7;padding-left:15px}}
  .viz-card{{background:#fff;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.05);
             margin-bottom:25px;overflow:hidden}}
  .viz-header{{display:flex;background:#f8f9fa;border-bottom:1px solid #eee}}
  .viz-col-header{{flex:1;text-align:center;padding:10px;font-weight:600;
                   color:#636e72;font-size:.9em;text-transform:uppercase}}
  .viz-content{{display:flex}}
  .viz-col{{flex:1;padding:20px;border-right:1px solid #f1f1f1;font-size:15px;line-height:1.6}}
  .viz-col:last-child{{border-right:none}}
  .legend{{text-align:center;margin-bottom:30px;font-size:.9em}}
  .dot{{height:10px;width:10px;border-radius:50%;display:inline-block;margin-right:5px}}
  footer{{text-align:center;padding:20px;color:#aaa;font-size:.8em}}
</style></head><body>
<div class="header">
  <h1>Clinical NER Benchmark Report</h1>
  <p>Custom {BASE_MODEL} + CRF  vs  Pretrained Baseline  |  {sample_size} BC5CDR samples</p>
</div>
<div class="container">
  <div class="card">
    <table>
      <tr><th>Metric (Macro Avg)</th><th>Your Custom CRF Model</th><th>Pretrained Baseline</th></tr>
      <tr><td>Precision</td><td>{op:.2%}</td><td>{pp:.2%}</td></tr>
      <tr><td>Recall</td>   <td>{or_:.2%}</td><td>{pr:.2%}</td></tr>
      <tr class="highlight"><td>F1 Score</td><td>{of1:.2%}</td><td>{pf1:.2%}</td></tr>
    </table>
  </div>
  <div class="charts-row">
    <div class="chart-box"><h3>F1 Performance</h3><img src="data:image/png;base64,{plot1}"/></div>
    <div class="chart-box"><h3>Error Analysis</h3><img src="data:image/png;base64,{plot2}"/></div>
  </div>
  <div class="legend">
    <span style="margin-right:15px"><span class="dot" style="background:#aa9cfc"></span>Chemical</span>
    <span style="margin-right:15px"><span class="dot" style="background:#ff9a8d"></span>Disease</span>
    <span><span class="dot" style="background:#feca57"></span>Dosage (regex)</span>
  </div>
  <div class="section-title">Custom Test Sentences</div>
  {viz_cards(custom_rows)}
  <div class="section-title">Random Dataset Samples</div>
  {viz_cards(dataset_rows)}
  <footer>Generated {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</footer>
</div></body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(model_path, baseline_path, train_first=False):
    """
    Runs the evaluation pipeline.
    Args:
        model_path (str): Path to your custom trained model.
        baseline_path (str): HuggingFace hub path for the baseline pipeline.
        train_first (bool): If True, triggers the training pipeline before evaluating.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_dir = Path(f"./reports/{timestamp}")
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"Report directory: {report_dir}")

    # ── 1. Train if requested ─────────────────────────────────────────────
    if train_first:
        print("\n[INFO] Training requested before evaluation...")
        train_new_model()

    # ── 2. Load models (Using the function arguments!) ────────────────────
    print(f"\nLoading custom model from: {model_path} …")
    print(f"Loading baseline pipeline from: {baseline_path} …")
    try:
        tokenizer_ours = AutoTokenizer.from_pretrained(model_path)

        # ADD ID2LABEL AND LABEL2ID HERE
        model_ours = TransformerNERWithCRF.from_pretrained(
            model_path,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            use_crf=USE_CRF
        )
        model_ours.eval()

        pipe_pre = pipeline(
            "token-classification",
            model=baseline_path,
            aggregation_strategy="simple",
        )
    except Exception as e:
        print(f"Error loading models: {e}")
        return  # Use return instead of exit(1) so it doesn't kill the Jupyter kernel

    # ── 3. Custom sentences ───────────────────────────────────────────────
    print("\nProcessing custom sentences …")
    opts = {"colors": COLORS}
    custom_rows = []

    for text in CUSTOM_SENTENCES:
        our_preds = predict_with_custom_model(text, model_ours, tokenizer_ours)
        pre_preds = predict_with_pipeline(text, pipe_pre)
        custom_rows.append({
            "html_ours": displacy.render(
                {"text": text, "ents": merge_dosages(our_preds, text)},
                style="ent", manual=True, options=opts, page=False),
            "html_pre": displacy.render(
                {"text": text, "ents": merge_dosages(pre_preds, text)},
                style="ent", manual=True, options=opts, page=False),
        })

    # ── 4. Benchmark on dataset ───────────────────────────────────────────
    print("\nBenchmarking on dataset …")
    dataset = load_dataset(DATASET_NAME)
    tag_key = "ner_tags" if "ner_tags" in dataset["test"].features else "tags"
    subset = (dataset["test"].shuffle(seed=42)  # Added a seed for reproducible notebook runs
              .select(range(int(len(dataset["test"]) * DATASET_PERCENTAGE))))

    entity_labels = ["CHEMICAL", "DISEASE"]
    empty_counts = lambda: {l: {"tp": 0, "fp": 0, "fn": 0} for l in entity_labels}
    stats_ours, stats_pre = empty_counts(), empty_counts()
    dataset_rows = []

    for idx, row in tqdm(enumerate(subset), total=len(subset)):
        text, gt_ents = tokens_to_text_and_entities(row["tokens"], row[tag_key])
        our_preds = predict_with_custom_model(text, model_ours, tokenizer_ours)
        pre_preds = predict_with_pipeline(text, pipe_pre)

        stats_ours = accumulate_stats(gt_ents, our_preds, stats_ours)
        stats_pre = accumulate_stats(gt_ents, pre_preds, stats_pre)

        if idx < RANDOM_DATASET_VIZ_COUNT:
            dataset_rows.append({
                "html_ours": displacy.render(
                    {"text": text, "ents": merge_dosages(our_preds, text)},
                    style="ent", manual=True, options=opts, page=False),
                "html_pre": displacy.render(
                    {"text": text, "ents": merge_dosages(pre_preds, text)},
                    style="ent", manual=True, options=opts, page=False),
            })

    # ── 5. Generate reports ───────────────────────────────────────────────
    print("\nGenerating reports …")
    plot1, plot2 = create_plots(stats_ours, stats_pre)
    html = generate_html_report(
        custom_rows, dataset_rows, stats_ours, stats_pre,
        len(subset), plot1, plot2,
    )

    (report_dir / "report.html").write_text(html, encoding="utf-8")
    save_csv_report(report_dir, stats_ours, stats_pre)
    print(f"\n[DONE] Reports saved to: {report_dir.absolute()}")

    # Returning the HTML string so the Jupyter Notebook can display it directly!
    return html
