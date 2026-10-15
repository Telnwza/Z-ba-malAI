"""Sweep THRESHOLD over tests/threshold_eval.csv and report precision / recall / F1 for to_genz.

Each row: text, convert, keep. convert = words that should be converted, keep = words that must
not be converted ("*" = the whole sentence must stay unchanged); both "|"-separated in order of
appearance. Conversions of words in neither column are ignored. Which Gen Z word is chosen is
not checked, only whether each word gets converted. Needs the real similarity model.

Run: python tests/tune_threshold.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # ให้ import src ได้เมื่อรันเป็นสคริปต์
from src.backend.translator import THRESHOLD, Translator  # noqa: E402

EVAL = Path(__file__).with_name("threshold_eval.csv")


def load():
    """[(text, convert spans, keep spans)] from the CSV."""
    with open(EVAL, encoding="utf-8-sig", newline="") as f:
        return [(row["text"], _spans(row["text"], row["convert"]), _spans(row["text"], row["keep"]))
                for row in csv.DictReader(f)]


def _spans(text, cell):
    """Spans of the "|"-separated words of cell in text ("*" = the whole text)."""
    if cell.strip() == "*":
        return [(0, len(text))]
    spans, pos = [], 0
    for w in filter(None, cell.split("|")):
        i = text.find(w, pos)
        if i < 0:
            raise ValueError(f"ไม่เจอ '{w}' ใน '{text}' (ต้องเรียงตามลำดับในประโยค)")
        spans.append((i, i + len(w)))
        pos = i + len(w)
    return spans


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def evaluate(t, rows):
    """(tp, fp, fn, errors): a word counts as converted when an applied span overlaps it."""
    tp = fp = fn = 0
    errors = []
    for text, exp, keep in rows:
        res = t.translate(text, "to_genz")
        got = [tuple(a["span"]) for a in res["applied_rules"]]
        hit = sum(any(_overlaps(e, g) for g in got) for e in exp)
        extra = sum(any(_overlaps(g, k) for k in keep) for g in got)
        tp, fn, fp = tp + hit, fn + len(exp) - hit, fp + extra
        if hit < len(exp) or extra:
            errors.append(f"{text} -> {res['output']}  (ขาด {len(exp) - hit}, เกิน {extra})")
    return tp, fp, fn, errors


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    rows = load()
    t = Translator()

    # threshold ต่ำสุด = จับคำอย่างเดียว คำที่ยังขาดตรงนี้คือหา keyword ไม่เจอ ไม่เกี่ยวกับ threshold
    t.threshold = -np.inf
    tp, fp, fn, errors = evaluate(t, rows)
    print(f"ไม่มี threshold: P={f1(tp, fp, fn)[0]:.2f} R={f1(tp, fp, fn)[1]:.2f} F1={f1(tp, fp, fn)[2]:.2f}")

    results = []
    print(f"\n{'threshold':>9} {'P':>5} {'R':>5} {'F1':>5}  tp  fp  fn")
    for th in np.round(np.arange(-0.10, 0.31, 0.02), 2):
        t.threshold = th
        tp, fp, fn, errors = evaluate(t, rows)
        p, r, f = f1(tp, fp, fn)
        results.append((f, th, errors))
        mark = "  <- ค่าปัจจุบัน" if th == THRESHOLD else ""
        print(f"{th:>9.2f} {p:>5.2f} {r:>5.2f} {f:>5.2f} {tp:>3} {fp:>3} {fn:>3}{mark}")

    best_f, best_th, errors = max(results, key=lambda x: x[0])
    print(f"\nดีที่สุด: threshold={best_th:.2f} F1={best_f:.2f}  ประโยคที่ผิด {len(errors)} ประโยค:")
    print("\n".join("  " + e for e in errors))


if __name__ == "__main__":
    main()
