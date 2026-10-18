"""Sweep THRESHOLD and the scoring options over tests/threshold_eval.csv; report precision / recall / F1 for to_genz.

Each row: text, convert, keep. convert = words that should be converted, keep = words that must
not be converted ("*" = the whole sentence must stay unchanged); both "|"-separated in order of
appearance. Conversions of words in neither column are ignored. Which Gen Z word is chosen is
not checked, only whether each word gets converted. Needs the real similarity model.

Settings compared in one run (see CONFIGS):
  window          score the whole sentence (None) or only the text around the matched word
  contrast weight final = train - weight * contrast
  register        add the formality score (-1 formal .. +1 chat, voted by the nearest reference
                  sentences of all rules) times a weight, so formal sentences lose score
  margin          drop a match when train - contrast is below this
Scores change scale with these settings, so each setting gets its own threshold grid.

Rows whose text is identical to a Reference (Train/Contrast) sentence are dropped by default: the
model has seen that exact sentence, so the result says nothing. Use --keep-leaked to keep them.

Run: python tests/tune_threshold.py          (add --sweep to print the full threshold curves)
     python tests/tune_threshold.py --check new.csv
         no sweep: score a new CSV (same columns) with the values set in src/backend/translator.py.
         Use it on sentences that were never used to write the dataset or to pick the settings.
"""
import argparse
import csv
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # ให้ import src ได้เมื่อรันเป็นสคริปต์
from src.backend import translator as tr  # noqa: E402
from src.backend.translator import Translator  # noqa: E402

EVAL = Path(__file__).with_name("threshold_eval.csv")
STEP = 0.02
OLD_THRESHOLD = 0.0       # ค่าเริ่มต้นเดิมของ translator.py ก่อนปรับ (ใช้เป็นจุดเทียบ)

# (window, contrast weight, margin, register weight, register k)
CONFIGS = [
    (None, 1.0, None, 0.0, 10),     # ค่าเริ่มต้นเดิม (ใช้เทียบ)
    (None, 0.0, None, 0.5, 10),     # ผลดีที่สุดของรอบก่อน
]
CONFIGS += [(None, w, None, reg, 10) for w in (0.0, 0.5) for reg in (0.7, 1.0, 1.5)]
CONFIGS += [(None, 0.0, None, reg, k) for reg in (0.5, 1.0) for k in (5, 20)]


def load(path=EVAL):
    """[(text, convert spans, keep spans)] from the CSV."""
    with open(path, encoding="utf-8-sig", newline="") as f:
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


def per_row(t, rows):
    """[(hit, miss, extra, error text or None)] for each row: a word counts as converted when an applied span overlaps it."""
    out = []
    for text, exp, keep in rows:
        res = t.translate(text, "to_genz")
        got = [tuple(a["span"]) for a in res["applied_rules"]]
        hit = sum(any(_overlaps(e, g) for g in got) for e in exp)
        extra = sum(any(_overlaps(g, k) for k in keep) for g in got)
        miss = len(exp) - hit
        err = f"{text} -> {res['output']}  (ขาด {miss}, เกิน {extra})" if miss or extra else None
        out.append((hit, miss, extra, err))
    return out


def evaluate(t, rows):
    """(tp, fp, fn, errors) summed over rows."""
    r = per_row(t, rows)
    return sum(x[0] for x in r), sum(x[2] for x in r), sum(x[1] for x in r), [x[3] for x in r if x[3]]


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _sum(block):
    """block (rows, 3) of (hit, miss, extra) -> (tp, fp, fn)."""
    s = block.sum(axis=0)
    return s[0], s[2], s[1]


def pick(counts, idx):
    """Best threshold index on the rows idx (median of the ties, so we land in the middle of a plateau)."""
    f = np.array([f1(*_sum(counts[i, idx]))[2] for i in range(len(counts))])
    ties = np.flatnonzero(f >= f.max() - 1e-9)
    return int(ties[len(ties) // 2]), float(f.max())


def make_grid(t, rows):
    """Threshold grid covering the scores this setting actually produces for the words found in the test set."""
    t.threshold = -np.inf
    sc = [p.score for text, _, _ in rows for p in t._propose(text, "to_genz", t.rules) if p.score is not None]
    if not sc:
        return np.round(np.arange(-0.2, 1.0, STEP), 2) + 0.0
    lo = np.floor(min(sc) / STEP) * STEP - STEP
    return np.round(np.arange(lo, max(sc) + 2 * STEP, STEP), 2) + 0.0     # + 0.0 กัน -0.00


def label(cfg):
    rad, w, m, reg, k = cfg
    return (f"window={rad} w_contrast={w} margin={m} register={reg}" + (f"(k={k})" if reg else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-leaked", action="store_true", help="keep rows identical to a Reference sentence")
    ap.add_argument("--check", metavar="CSV", help="score this CSV with the values in translator.py and exit (no sweep)")
    ap.add_argument("--sweep", action="store_true", help="print P/R/F1 at every threshold for the best and the current setting")
    args = ap.parse_args()

    rows = load(Path(args.check) if args.check else EVAL)
    tr.load_engine()
    if tr.engine is not None:
        ref = {x["sentence"] for x in tr.engine.train_meta} | {x["sentence"] for x in tr.engine.contrast_meta}
        leaked = [r for r in rows if r[0].strip() in ref]
        if leaked:
            print(f"คำเตือน: {len(leaked)} ประโยคในชุดทดสอบซ้ำกับประโยคอ้างอิง (Train/Contrast) เป๊ะ ๆ:")
            print("\n".join("  " + r[0] for r in leaked))
            if not args.keep_leaked:
                rows = [r for r in rows if r[0].strip() not in ref]
                print(f"-> ตัดออกจากการคำนวณ เหลือ {len(rows)} ประโยค (ใช้ --keep-leaked ถ้าต้องการเก็บไว้)\n")

    if args.check:
        t = Translator()                              # ใช้ค่าคงที่ใน translator.py ตามที่ตั้งไว้จริง
        tp, fp, fn, errors = evaluate(t, rows)
        p, r, f = f1(tp, fp, fn)
        print(f"ใช้ค่าใน translator.py: THRESHOLD={t.threshold} CONTRAST_WEIGHT={t.contrast_weight} "
              f"WINDOW_RADIUS={t.window_radius} REGISTER_WEIGHT={t.register_weight} REGISTER_K={t.register_k} MARGIN_MIN={t.margin_min}")
        print(f"{len(rows)} ประโยค: P={p:.2f} R={r:.2f} F1={f:.2f}  (tp={tp} fp={fp} fn={fn})")
        print("ประโยคที่ผิด:" if errors else "ไม่มีประโยคที่ผิด")
        print("\n".join("  " + e for e in errors))
        return

    @lru_cache(maxsize=None)
    def cached(text, span=None, radius=None, contrast_weight=1.0, register_weight=0.0, register_k=10):
        return tuple(tr.score_rules(text, span=span, radius=radius, contrast_weight=contrast_weight,
                                    register_weight=register_weight, register_k=register_k))

    n_pos = sum(len(r[1]) for r in rows)
    n_neg = sum(1 for r in rows if r[2])
    print(f"ชุดทดสอบ: {len(rows)} ประโยค | คำที่ควรแปลง {n_pos} | ประโยคที่ต้องคงเดิม {n_neg} | ทดลอง {len(CONFIGS)} การตั้งค่า")

    even, odd = np.arange(len(rows)) % 2 == 0, np.arange(len(rows)) % 2 == 1
    everyone = np.ones(len(rows), bool)
    results = []
    for cfg in CONFIGS:
        rad, w, m, reg, k = cfg
        t = Translator(scorer=cached, window_radius=rad, contrast_weight=w, margin_min=m,
                       register_weight=reg, register_k=k)
        grid = make_grid(t, rows)
        t.threshold = -np.inf                         # จับคำอย่างเดียว คำที่ขาดตรงนี้ไม่เกี่ยวกับ threshold
        base = f1(*evaluate(t, rows)[:3])
        counts = np.zeros((len(grid), len(rows), 3))
        errors_at = []
        for i, th in enumerate(grid):
            t.threshold = th
            res = per_row(t, rows)
            counts[i] = [(x[0], x[1], x[2]) for x in res]
            errors_at.append([x[3] for x in res if x[3]])
        bi, _ = pick(counts, everyone)
        p, r, f = f1(*_sum(counts[bi]))
        # ประมาณผลกับข้อมูลที่ไม่เคยเห็น: เลือก threshold จากครึ่งหนึ่ง ทดสอบอีกครึ่ง สลับกัน
        cv = np.zeros(3)
        for a, b in ((even, odd), (odd, even)):
            ti, _ = pick(counts, a)
            cv += np.array(_sum(counts[ti][b]))
        plateau = int(sum(f1(*_sum(counts[i]))[2] >= f - 0.01 for i in range(len(grid))))
        results.append(dict(cfg=cfg, grid=grid, counts=counts, bi=bi, th=grid[bi], p=p, r=r, f=f,
                            cv=f1(*cv)[2], plateau=plateau, base=base, errors=errors_at[bi]))
        print(f"  ... {label(cfg)} F1={f:.2f}", flush=True)

    cur = results[0]
    ranked = sorted(results, key=lambda x: -x["cv"])
    print(f"\n{'window':>6} {'w_contr':>7} {'register':>10} {'margin':>7} {'thr':>6} {'P':>5} {'R':>5} {'F1':>5} {'F1(2-fold)':>10} {'plateau':>8}")
    for x in ranked:
        rad, w, m, reg, k = x["cfg"]
        regtxt = f"{reg}(k={k})" if reg else "-"
        mark = "  <- ค่าเริ่มต้นเดิม" if x is cur else ""
        print(f"{str(rad):>6} {w:>7.2f} {regtxt:>10} {str(m):>7} {x['th']:>6.2f} {x['p']:>5.2f} {x['r']:>5.2f} {x['f']:>5.2f} {x['cv']:>10.2f} {x['plateau']:>8}{mark}")
    print("F1(2-fold) = เลือก threshold จากครึ่งหนึ่งแล้ววัดอีกครึ่ง (ใกล้ความจริงกว่า F1 ที่เลือกจากชุดเดียวกัน)")
    print(f"plateau   = จำนวน threshold (ห่างกัน {STEP}) ที่ F1 ต่ำกว่าค่าสูงสุดไม่เกิน 0.01 (ยิ่งมาก = ยิ่งไม่ไวต่อค่า)")

    ci = int(np.argmin(np.abs(cur["grid"] - OLD_THRESHOLD)))
    p, r, f = f1(*_sum(cur["counts"][ci]))
    print(f"\nค่าเริ่มต้นเดิม (threshold={cur['grid'][ci]:.2f}, w_contrast=1.0, ไม่มีตัวตรวจระดับภาษา): P={p:.2f} R={r:.2f} F1={f:.2f}")
    print(f"จับคำอย่างเดียว ไม่มี threshold: P={cur['base'][0]:.2f} R={cur['base'][1]:.2f} F1={cur['base'][2]:.2f}")

    def curve(title, x):
        print(f"\n{title}\n{'threshold':>9} {'P':>5} {'R':>5} {'F1':>5}  tp  fp  fn")
        for i, th in enumerate(x["grid"]):
            tp, fp, fn = _sum(x["counts"][i])
            pp, rr, ff = f1(tp, fp, fn)
            print(f"{th:>9.2f} {pp:>5.2f} {rr:>5.2f} {ff:>5.2f} {int(tp):>3} {int(fp):>3} {int(fn):>3}")

    best = ranked[0]
    if args.sweep:
        curve("เส้นโค้งของค่าเริ่มต้นเดิม", cur)
        curve(f"เส้นโค้งของตัวที่ดีที่สุด ({label(best['cfg'])})", best)

    rad, w, m, reg, k = best["cfg"]
    gain = best["cv"] - cur["cv"]
    print(f"\nดีที่สุด (ตาม F1 2-fold): {label(best['cfg'])} threshold={best['th']:.2f}")
    print(f"เทียบค่าเริ่มต้นเดิม F1(2-fold) {cur['cv']:.2f} -> {best['cv']:.2f} ({gain:+.2f})"
          + ("  *ต่างกันไม่เกิน 0.02 ถือว่าอยู่ในระดับสุ่มของชุดทดสอบขนาดนี้*" if abs(gain) <= 0.02 else ""))
    print("หมายเหตุ: เลือกจาก %d การตั้งค่าบนชุดเดียวกัน ตัวเลขจึงดีกว่าความจริงเล็กน้อย" % len(CONFIGS))
    print("\nค่าที่จะใส่ใน src/backend/translator.py:")
    print(f"  THRESHOLD = {best['th']:.2f}\n  CONTRAST_WEIGHT = {w}\n  WINDOW_RADIUS = {rad}\n"
          f"  REGISTER_WEIGHT = {reg}\n  REGISTER_K = {k}\n  MARGIN_MIN = {m}")
    print(f"\nประโยคที่ยังผิด {len(best['errors'])} ประโยค:")
    print("\n".join("  " + e for e in best["errors"]))


if __name__ == "__main__":
    main()
