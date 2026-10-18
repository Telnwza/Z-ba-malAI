import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from similarity_engine import DEFAULT_MODEL, clean_text, encode_texts
from pathlib import Path
HERE = Path(__file__).resolve().parent
DEFAULT_EXCEL = str(HERE.parents[2] / "data" / "dataset.xlsx")
SHEET = "02_PROGRAM"


def _is_true(v):
    """Enabled อาจเป็น bool / 1 / 'TRUE' / 'yes' ขึ้นกับว่า Excel บันทึกมายังไง"""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return not pd.isna(v) and v != 0
    return str(v).strip().upper() in {"TRUE", "1", "YES", "Y", "T"}


def _none_if_nan(v):
    if pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


def split_sentences(text):
    """แยกด้วย || , clean (NFC/ช่องว่าง), ตัดซ้ำโดยคงลำดับ"""
    if pd.isna(text):
        return []
    parts = (clean_text(s) for s in str(text).split("||"))
    return list(dict.fromkeys(s for s in parts if s))


def load_examples(excel_file=DEFAULT_EXCEL, sheet=SHEET):
    """คืน (rules, train_data, contrast_data)"""
    df = pd.read_excel(excel_file, sheet_name=sheet, header=1)
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(subset=["Rule ID"])
    df = df[df["Rule ID"].astype(str).str.strip() != ""]
    if df.empty:
        raise ValueError("Rule ID Not Found")
    df = df[df["Enabled"].map(_is_true)]

    rules, train_data, contrast_data = [], [], []
    seen_rule_ids = set()
    for _, row in df.iterrows():
        rid = str(row["Rule ID"]).strip()
        if rid in seen_rule_ids:
            raise ValueError(f"Rule ID ซ้ำ: {rid}")
        seen_rule_ids.add(rid)
        rules.append({
            "rule_id": rid,
            "data_id": _none_if_nan(row.get("Data ID")),
            "operation": _none_if_nan(row.get("Operation")),
            "style_level": _none_if_nan(row.get("Style Level")),
            "content_label": _none_if_nan(row.get("Content Label")),
            "source": _none_if_nan(row.get("Source Pattern")),
            "target": _none_if_nan(row.get("Target Pattern")),
            "variants": _none_if_nan(row.get("Accepted Variants")),
            "max_apply": _none_if_nan(row.get("Max Apply")),
        })
        tr = split_sentences(row["Train Examples"])
        co = split_sentences(row["Contrast Examples"])
        for s in tr:
            train_data.append({"sentence": s, "rule_id": rid, "kind": "train"})
        for s in co:
            contrast_data.append({"sentence": s, "rule_id": rid, "kind": "contrast"})
        both = set(tr) & set(co)
        if both:
            print(f"Warning {rid}: ประโยคเดียวกันอยู่ทั้ง Train และ Contrast -> {sorted(both)[:2]}")
        if not tr:
            print(f"Warning {rid}: ไม่มี Train Examples (rule นี้จะไม่ถูกให้คะแนน)")
    return rules, train_data, contrast_data


def load_cache(out_dir, model_name):
    """ใช้เวกเตอร์เดิมซ้ำถ้าโมเดลเดียวกันและประโยคเดียวกัน (incremental encode)"""
    cache = {}
    try:
        with open(os.path.join(out_dir, "metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("config", {}).get("model_name") != model_name:
            return cache
        for kind, fn in (("train", "E_train.npy"), ("contrast", "E_contrast.npy")):
            E = np.load(os.path.join(out_dir, fn))
            if E.ndim == 2 and E.shape[0] == len(meta[kind]):
                for item, v in zip(meta[kind], E):
                    cache[item["sentence"]] = v
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default=DEFAULT_EXCEL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--force", action="store_true", help="ไม่ใช้ cache")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"1. โหลดข้อมูลจาก {args.excel} (ชีท {SHEET})...")
    rules, train_data, contrast_data = load_examples(args.excel)
    print(f"   Rule ที่เปิดใช้งาน: {len(rules)} | Train: {len(train_data)} | Contrast: {len(contrast_data)}")
    if not train_data:
        raise SystemExit("ไม่มีประโยค Train เลย")

    print("2. เตรียม encoder...")
    cache = {} if args.force else load_cache(args.out_dir, args.model)
    all_sentences = list(dict.fromkeys(
        [x["sentence"] for x in train_data] + [x["sentence"] for x in contrast_data]))
    missing = [s for s in all_sentences if s not in cache]
    print(f"   ประโยคไม่ซ้ำ {len(all_sentences)} | ใช้ cache {len(all_sentences) - len(missing)} | encode ใหม่ {len(missing)}")

    if missing:
        from sentence_transformers import SentenceTransformer
        print(f"   กำลังโหลดโมเดล '{args.model}' ...")
        encoder = SentenceTransformer(args.model)
        print("3. Encode + L2 normalize ...")
        V = encode_texts(encoder, missing, args.model, role="passage")
        for s, v in zip(missing, V):
            cache[s] = v

    E_train = np.stack([cache[x["sentence"]] for x in train_data]).astype(np.float32)
    dim = E_train.shape[1]
    if contrast_data:
        E_contrast = np.stack([cache[x["sentence"]] for x in contrast_data]).astype(np.float32)
    else:
        E_contrast = np.empty((0, dim), dtype=np.float32)
    print(f"   E_train {E_train.shape} | E_contrast {E_contrast.shape}")

    print("4. บันทึกไฟล์...")
    np.save(os.path.join(args.out_dir, "E_train.npy"), E_train)
    np.save(os.path.join(args.out_dir, "E_contrast.npy"), E_contrast)
    metadata = {
        "config": {
            "model_name": args.model,
            "dim": int(dim),
            "n_rules": len(rules),
            "n_train": len(train_data),
            "n_contrast": len(contrast_data),
            "excel": os.path.basename(args.excel),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "rules": rules,
        "train": train_data,
        "contrast": contrast_data,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Success -> {args.out_dir}/E_train.npy, E_contrast.npy, metadata.json")


if __name__ == "__main__":
    main()