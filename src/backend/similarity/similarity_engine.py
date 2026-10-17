import json
import os
import re
import unicodedata
import zlib
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "BAAI/bge-m3"


class EngineNotReady(Exception):
    """โมเดลหรือไฟล์เวกเตอร์ยังโหลดไม่เสร็จ -> backend ควรตอบ 503"""


# ==========================================
# พีชคณิตเชิงเส้นพื้นฐาน (numpy ล้วน)
# ==========================================
def normalize(v, axis=-1, eps=1e-12):
    """L2 normalize: v / ||v||  (แถวศูนย์ยังคงเป็นศูนย์)"""
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + eps)


def cosine(a, b):
    """cos(a, b) = a·b / (||a|| ||b||)"""
    a = normalize(a)
    b = normalize(b)
    return float(np.dot(a, b))


def search(q, E, k=5):
    """nearest neighbors: scores = E @ q (E, q normalize แล้ว) คืน (index, score) top-k"""
    scores = E @ q
    k = min(k, len(scores))
    top = np.argsort(-scores)[:k]
    return top, scores[top]


# ==========================================
# Helper ร่วมกับ build_vector_db.py / evaluate.py
# ==========================================
def clean_text(s):
    """NFC + ยุบช่องว่าง (สระ/วรรณยุกต์ภาษาไทยที่พิมพ์ต่างลำดับจะได้เป็นแบบเดียวกัน)"""
    s = unicodedata.normalize("NFC", str(s))
    return re.sub(r"\s+", " ", s).strip()


def prefixes_for(model_name):
    """(prefix ฝั่ง query, prefix ฝั่งประโยคอ้างอิง)
    ตระกูล E5 ต้องมี prefix; งาน semantic similarity แบบสมมาตรให้ใช้ 'query: ' ทั้งสองฝั่ง"""
    if "e5" in model_name.lower():
        return "query: ", "query: "
    return "", ""


def encode_texts(encoder, texts, model_name, role="passage"):
    """encode -> L2 normalize ด้วย normalize() ของเราเอง -> float32 (n, d)"""
    q_prefix, p_prefix = prefixes_for(model_name)
    prefix = q_prefix if role == "query" else p_prefix
    V = encoder.encode([prefix + t for t in texts], batch_size=32)
    return normalize(np.asarray(V, dtype=np.float32))


class HashingEncoder:
    """encoder ปลอม (char-trigram hashing) ไว้ทดสอบโค้ดโดยไม่ต้องโหลดโมเดลจริง
    *** ห้ามใช้วัดผลจริง — ผลที่ได้ไม่ใช่ semantic ***"""

    def __init__(self, dim=512):
        self.dim = dim

    def encode(self, texts, **kwargs):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = f"  {t}  "
            for j in range(len(t) - 2):
                h = zlib.crc32(t[j:j + 3].encode("utf-8"))
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return out


# ==========================================
# State ระดับ Global (โหลดครั้งเดียวตอนเปิดเซิร์ฟเวอร์)
# ==========================================
model = None
model_name = None
config = {}
E_train = None
E_contrast = None
train_meta = []
contrast_meta = []
rules_info = {}
all_rule_ids = []
train_idx = {}       # rule_id -> np.array ของแถวใน E_train
contrast_idx = {}    # rule_id -> np.array ของแถวใน E_contrast
E_all = None         # train + contrast ต่อกัน ใช้โหวตเพื่อนบ้านใน register_score
reg_labels = None    # +1 = แถวของ train, -1 = แถวของ contrast


def is_ready():
    return model is not None and E_train is not None


def init_engine(data_dir=None, model_name_override=None, encoder=None):
    """โหลดเวกเตอร์ + metadata + โมเดล  (เรียกซ้ำได้: state จะถูกสร้างใหม่ทั้งหมด)
    encoder: ใส่ object ที่มี .encode(list[str]) เพื่อข้ามการโหลดโมเดลจริง (ใช้ทดสอบ)"""
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    global model, model_name, config, E_train, E_contrast, train_meta, contrast_meta
    global rules_info, all_rule_ids, train_idx, contrast_idx, E_all, reg_labels

    with open(os.path.join(data_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    cfg = metadata.get("config", {})
    built_with = cfg.get("model_name")
    name = model_name_override or built_with or DEFAULT_MODEL
    if built_with and name != built_with:
        raise ValueError(
            f"เวกเตอร์ถูกสร้างด้วย '{built_with}' แต่กำลังจะโหลด '{name}' — "
            f"ต้องรัน build_vector_db.py ใหม่ด้วยโมเดลเดียวกัน")

    Et = normalize(np.load(os.path.join(data_dir, "E_train.npy")))
    try:
        Ec = normalize(np.load(os.path.join(data_dir, "E_contrast.npy")))
    except FileNotFoundError:
        Ec = np.empty((0, Et.shape[1]), dtype=np.float32)
    if Ec.ndim != 2 or Ec.shape[0] == 0:
        Ec = np.empty((0, Et.shape[1]), dtype=np.float32)

    tm, cm = metadata["train"], metadata["contrast"]
    if len(tm) != Et.shape[0] or len(cm) != Ec.shape[0]:
        raise ValueError(
            f"metadata ไม่ตรงกับ .npy (train {len(tm)} vs {Et.shape[0]}, "
            f"contrast {len(cm)} vs {Ec.shape[0]}) — run build_vector_db.py ใหม่")

    if encoder is None:
        print("ระบบกำลังโหลด Sentence Model (ใช้เวลาสักครู่)...")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(name)

    # --- สร้าง index (ล้างของเก่าทุกครั้ง ไม่ append ซ้ำ) ---
    t_map, c_map = {}, {}
    for i, item in enumerate(tm):
        t_map.setdefault(item["rule_id"], []).append(i)
    for i, item in enumerate(cm):
        c_map.setdefault(item["rule_id"], []).append(i)

    model, model_name, config = encoder, name, cfg
    E_train, E_contrast, train_meta, contrast_meta = Et, Ec, tm, cm
    rules_info = {r["rule_id"]: r for r in metadata.get("rules", [])}
    all_rule_ids = sorted(t_map)
    train_idx = {r: np.array(v) for r, v in t_map.items()}
    contrast_idx = {r: np.array(v) for r, v in c_map.items()}
    E_all = np.vstack([Et, Ec]) if len(Ec) else Et
    reg_labels = np.r_[np.ones(len(Et)), -np.ones(len(Ec))].astype(np.float32)
    _encode_cached.cache_clear()

    print(f"โหลดระบบ Similarity เรียบร้อย (model={name}, Train={Et.shape[0]}, "
          f"Contrast={Ec.shape[0]}, Rule={len(all_rule_ids)}, dim={Et.shape[1]})")


def engine_info():
    return {
        "ready": is_ready(),
        "model": model_name,
        "n_rules": len(all_rule_ids),
        "n_train": 0 if E_train is None else int(E_train.shape[0]),
        "n_contrast": 0 if E_contrast is None else int(E_contrast.shape[0]),
        "dim": None if E_train is None else int(E_train.shape[1]),
        "scoring": "max_contrast",
    }


@lru_cache(maxsize=2048)
def _encode_cached(text):
    q = encode_texts(model, [text], model_name, role="query")[0]
    q.setflags(write=False)
    return q


def encode_query(text):
    if not is_ready():
        raise EngineNotReady("model or vectors not loaded")
    q = _encode_cached(clean_text(text))
    if q.shape[0] != E_train.shape[1]:
        raise ValueError("มิติของ query ไม่ตรงกับเวกเตอร์ที่เก็บไว้ (โมเดลคนละตัว?)")
    return q


# ==========================================
# ฟังก์ชันหลักสำหรับโฟล์ม
# ==========================================
def register_score(q, k=10):
    """ตัวตรวจ "ระดับภาษา": โหวตจากประโยคอ้างอิงที่ใกล้ q ที่สุด k ตัว ของ **ทุก rule** รวมกัน
    Train = +1 (ประโยคที่ rule ควรโดน เป็นภาษาแชท), Contrast = -1 (ไม่ควรโดน ส่วนใหญ่เป็นภาษาทางการ/ความหมายอื่น)
    ถ่วงน้ำหนักด้วย cosine คืนค่า -1..+1 (ใกล้ -1 = ใกล้ประโยค Contrast เช่น ประโยคทางการ)
    ค่านี้ไม่ขึ้นกับ rule จึงช่วยได้แม้ rule นั้นมี Contrast ไม่ครอบคลุม"""
    sims = E_all @ q
    k = max(1, min(int(k), len(sims)))
    top = np.argpartition(-sims, k - 1)[:k]
    w = np.clip(sims[top], 1e-6, None)
    return float((reg_labels[top] * w).sum() / w.sum())


_LEAD_VOWELS = "เแโใไ"   # สระหน้า: ต้องมีพยัญชนะตามหลังเสมอ


def context_window(text, span, radius=15):
    """ตัดเฉพาะข้อความรอบคำที่เจอ: span=(start, end) นับตัวอักษรในข้อความต้นฉบับแบบ slice
    ขยายซ้าย/ขวาข้างละ radius ตัวอักษร (ประโยคอ้างอิงยาวเฉลี่ย ~31 ตัวอักษร
    radius=15 จึงให้หน้าต่างยาวใกล้เคียงกัน) ไม่ตัดกลางพยางค์ไทย
    span=None หรือ radius=None หรือ span ผิดช่วง -> คืนข้อความเต็ม"""
    if span is None or radius is None:
        return text
    n = len(text)
    start, end = int(span[0]), int(span[1])
    if not (0 <= start <= end <= n):
        return text
    left = max(0, start - radius)
    right = min(n, end + radius)
    # สระบน/ล่าง/วรรณยุกต์ (Mn) ต้องมีพยัญชนะฐานติดมา และสระหน้าต้องมีพยัญชนะตามหลัง
    while left > 0 and (unicodedata.category(text[left]) == "Mn" or text[left - 1] in _LEAD_VOWELS):
        left -= 1
    while right < n and (unicodedata.category(text[right]) == "Mn" or text[right - 1] in _LEAD_VOWELS):
        right += 1
    return text[left:right]


def score_rules(text, top_k=None, rule_ids=None, contrast_weight=1.0, span=None, radius=15,
                register_weight=0.0, register_k=10):
    """
    คืน list ของ dict เรียง final_score มาก -> น้อย:
      rule_id, final_score, train_score, contrast_score,
      best_train_sentence, best_contrast_sentence (None ถ้า rule ไม่มี contrast)

    สูตรเดียว (ตามสเปก):
      train_score    = max cos(q, ประโยคอ้างอิง train ของ rule)
      contrast_score = max cos(q, contrast ของ rule)  หรือ 0 ถ้าไม่มี contrast
      final_score    = train_score - contrast_weight * contrast_score + register_weight * register_score
      margin         = train_score - contrast_score   (ไว้ใช้เป็นตัวกรอง: margin < 0 = ใกล้ประโยค contrast กว่า train)

    rule_ids: ให้คะแนนเฉพาะ rule ที่ระบุ (None = ทุก rule)
    contrast_weight: น้ำหนักของ contrast ในสูตร (1.0 = สูตรเดิมตามสเปก, 0.0 = จัดอันดับด้วย train อย่างเดียว)
    register_weight, register_k: บวกคะแนน "ระดับภาษา" (register_score, -1..+1) เข้า final ทุก rule เท่ากัน
                  ประโยคทางการได้ค่าติดลบ -> คะแนนลด 0.0 = ไม่ใช้ (เหมือนเดิม)
    span, radius: translator ส่งตำแหน่งคำ (start, end) มา -> ให้คะแนนเฉพาะช่วงรอบคำ (ดู context_window)
                  ไม่ส่ง span = ให้คะแนนทั้งข้อความเหมือนเดิม
    top_k: ตัดเหลือ k อันดับแรก (None = คืนทุก rule ให้ translator กรอง threshold เอง)
    ข้อความว่าง -> []   ยังไม่พร้อม -> raise EngineNotReady
    """
    if not is_ready():
        raise EngineNotReady("model or vectors not loaded")
    text = clean_text(context_window(text, span, radius))   # ตัดหน้าต่างก่อน แล้วค่อย clean
    if not text:
        return []

    q = encode_query(text)
    t_all = E_train @ q
    c_all = E_contrast @ q if E_contrast.shape[0] else np.empty(0, dtype=np.float32)
    register = register_score(q, register_k) if register_weight else 0.0

    wanted = all_rule_ids if rule_ids is None else [r for r in rule_ids if r in train_idx]
    results = []
    for rid in wanted:
        ti = train_idx[rid]
        bt = int(ti[np.argmax(t_all[ti])])
        ci = contrast_idx.get(rid)
        bc = int(ci[np.argmax(c_all[ci])]) if ci is not None and len(ci) else None

        train_score = float(t_all[bt])
        contrast_score = float(c_all[bc]) if bc is not None else 0.0
        results.append({
            "rule_id": rid,
            "final_score": train_score - contrast_weight * contrast_score + register_weight * register,
            "margin": train_score - contrast_score,
            "register": register,
            "train_score": train_score,
            "contrast_score": contrast_score,
            "best_train_sentence": train_meta[bt]["sentence"],
            "best_contrast_sentence": contrast_meta[bc]["sentence"] if bc is not None else None,
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_k] if top_k is not None else results


def search_sentences(text, k=5, kind="train"):
    """ประโยคอ้างอิงที่ใกล้ข้อความที่สุด k อันดับ (สำหรับ endpoint /search)"""
    q = encode_query(text)
    E, meta = (E_contrast, contrast_meta) if kind == "contrast" else (E_train, train_meta)
    if E.shape[0] == 0:
        return []
    top, sc = search(q, E, k)
    return [{"sentence": meta[i]["sentence"], "rule_id": meta[i]["rule_id"], "score": float(s)}
            for i, s in zip(top, sc)]


# ==========================================
# รันทดสอบ
# ==========================================
if __name__ == "__main__":
    init_engine()
    print("\n--- พิมพ์ 'exit' เพื่อออก ---")
    while True:
        test_text = input("\nใส่ข้อความที่ต้องการทดสอบ: ")
        if test_text.lower() == "exit":
            break
        if not test_text.strip():
            continue
        print("\nTop 5 (max train − max contrast):")
        for idx, r in enumerate(score_rules(test_text, top_k=5), 1):
            print(f"{idx}. {r['rule_id']:<5} final={r['final_score']:.4f} "
                  f"(train={r['train_score']:.4f}, contrast={r['contrast_score']:.4f})"
                  f"  ← {r['best_train_sentence']}")
