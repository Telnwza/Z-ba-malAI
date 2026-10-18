# Z-ba-malAI
📄 [เอกสารโปรเจกต์ (Google Docs)](https://docs.google.com/document/d/1Opkayj902YEVEXJktI4yvAUjIlf8E0QFhtKoZ7hqR8M/edit?usp=drivesdk)

API แปลงภาษาไทยปกติ ↔ ภาษา Gen Z ตาม spec `POST /translate` ในเอกสารแผนงาน

## โครงสร้าง

```
Z-ba-malAI/
├── src/
│   ├── backend/
│   │   ├── app.py              # FastAPI POST /translate
│   │   ├── rules.py            # โหลด rule จากชีท 02_PROGRAM ใน data/dataset.xlsx
│   │   ├── translator.py       # หาคำในประโยค, ตัดด้วยคะแนน, เลือก rule เมื่อชนกัน, แทนที่
│   │   └── similarity/         # ให้คะแนน rule ด้วย cosine similarity: engine, สคริปต์สร้างเวกเตอร์, .npy + metadata.json
│   └── frontend/               # หน้าเว็บ (ยังว่าง)
├── data/
│   └── dataset.xlsx            # dataset ของ rule (ใช้ชีท 02_PROGRAM)
├── docs/                       # เอกสาร (ยังว่าง)
├── tests/
│   ├── test_translator.py      # เทสต์ translator และ API
│   ├── threshold_eval.csv      # ประโยคพร้อมเฉลยว่าคำไหนควร/ไม่ควรแปลง (ร่าง)
│   └── tune_threshold.py       # ไล่หาค่า THRESHOLD จาก threshold_eval.csv
└── requirements.txt
```

## รัน (Windows, Python 3.10+)

รันจากโฟลเดอร์ `Z-ba-malAI` ใน Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
uvicorn src.backend.app:app --reload
```

เปิด http://127.0.0.1:8000/docs เพื่อลองเรียก API · รันเทสต์: `python -m pytest -q`

### โมเดล `BAAI/bge-m3`

`to_genz` ต้องใช้โมเดลแปลงประโยคที่ผู้ใช้พิมพ์เป็นเวกเตอร์ (เวกเตอร์ของประโยคใน dataset คำนวณไว้แล้วใน `src/backend/similarity/*.npy`)
`to_normal` และเทสต์ไม่ต้องใช้โมเดล

- เซิร์ฟเวอร์โหลดโมเดลตอนเปิด (`uvicorn` จะใช้เวลาสักพักก่อนพร้อม) ครั้งแรกจะดาวน์โหลดเอง (~2 GB)
  เก็บไว้ที่ `~/.cache/huggingface` ครั้งต่อไปไม่ต้องโหลดอีก
- ถ้าโหลดไม่สำเร็จ เซิร์ฟเวอร์ยังเปิดได้และพิมพ์สาเหตุไว้ใน log; `to_normal` ใช้ได้ ส่วน `to_genz` จะลองโหลดใหม่ตอนถูกเรียก
- ดาวน์โหลดโมเดลไว้ก่อนได้ด้วย `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"`

ถ้าเจอ `SSLError ... CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate` ตอนโหลด
(มักเกิดจากแอนตี้ไวรัสหรือเน็ตองค์กร/มหาลัยที่ดักตรวจ https) ให้ Python ใช้ใบรับรองของ Windows แล้วรันใหม่

```bash
python -m pip install pip-system-certs
```

หรือลองต่อเน็ตอื่น (เช่นแชร์จากมือถือ) แค่ตอนโหลดครั้งแรก ห้ามแก้ด้วยการปิดการตรวจ SSL (`verify=False`)

## API: `POST /translate`

แปลงข้อความหนึ่งข้อความตาม `direction` แล้วคืนผลพร้อมคำที่ถูกแปลงและตัวเลือกของแต่ละคำ รับและคืนเป็น JSON (UTF-8)

### Request body

| Field | Type | จำเป็น | ค่าเริ่มต้น | คำอธิบาย |
| --- | --- | --- | --- | --- |
| `text` | string | ใช่ | – | ข้อความที่จะแปลง ยาว 1–500 ตัวอักษร |
| `direction` | `"to_genz"` \| `"to_normal"` | ใช่ | – | ทิศทางการแปลง |
| `max_style_level` | int 1–4 | ไม่ | 3 | ใช้เฉพาะ rule ที่ Style Level ≤ ค่านี้ |
| `allow_profanity` | bool | ไม่ | false | true = ใช้ rule ที่ Content Label เป็น Profanity/Insult ได้ และเปิดระดับ 4 ให้เอง |
| `top_k` | int 1–10 | ไม่ | 5 | จำนวน `candidates` สูงสุดต่อคำที่ถูกแปลง |

### Response 200

| Field | Type | คำอธิบาย |
| --- | --- | --- |
| `output` | string | ข้อความหลังแปลง (เท่ากับข้อความเดิมถ้าไม่มี rule ไหนผ่าน) |
| `applied_rules` | array | คำที่ถูกแปลง 1 ตัวต่อ 1 คำ เรียงตามตำแหน่งในประโยค ถ้าว่าง = ไม่มีการแปลง |

แต่ละตัวใน `applied_rules`

| Field | Type | คำอธิบาย |
| --- | --- | --- |
| `from` | string | คำหรือวลีในข้อความเดิมที่ถูกแทน |
| `span` | [int, int] | ตำแหน่งตัวอักษร [เริ่ม, จบ) ของ `from` ในข้อความเดิม แบบ slice ของ Python |
| `output_span` | [int, int] | ตำแหน่งของคำที่ใส่แทนใน `output` ใช้ไฮไลต์ในผลลัพธ์ |
| `candidates` | array | ตัวเลือกคำแปลของคำนี้ ไม่เกิน `top_k` ตัว ตัวแรกคือตัวที่ใช้จริงเสมอ ที่เหลือเรียงตามคะแนน |

แต่ละตัวใน `candidates`

| Field | Type | คำอธิบาย |
| --- | --- | --- |
| `rule_id` | string | เช่น `"P018"` จากชีท `02_PROGRAM` (เลขเลื่อนได้เมื่อ dataset เปลี่ยน) |
| `to` | string | คำที่ใส่แทน (ถ้า rule มี `{X}` จะเติมคำแล้ว เช่น `"เหนื่อยจัด"`) |
| `score` | number \| null | คะแนน `final_score` จาก similarity (ดูหัวข้อ "การให้คะแนน") ทิศทาง `to_normal` เป็น null (เจอแบบตรงตัว) |
| `applied` | bool | true = ตัวที่ใช้จริงใน `output` |

หมายเหตุ

- `allow_profanity: true` จะเปิดให้ใช้ rule ระดับ 4 ด้วย เพราะคำหยาบส่วนใหญ่อยู่ระดับ 4
- `to_genz`: rule หนึ่งใช้ได้ไม่เกินค่า Max Apply ใน dataset ต่อประโยค · `to_normal`: แปลงทุกที่ที่เจอ

### Errors

| Status | เมื่อไหร่ |
| --- | --- |
| 422 | `text` ว่างหรือยาวเกิน 500, `direction` ไม่ถูก, ค่าอยู่นอกช่วง (รูปแบบมาตรฐานของ FastAPI: `{"detail": [...]}`) |

### ตัวอย่าง

ผลรันจริงกับ `data/dataset.xlsx` และโมเดล `BAAI/bge-m3` (ปัดคะแนนเหลือ 2 ตำแหน่ง)

```json
// request
{ "text": "ไม่เข้าใจเลยครับ เหนื่อยมาก", "direction": "to_genz", "top_k": 3 }

// response
{
  "output": "งงเลยคับ เหนื่อยม้าก",
  "applied_rules": [
    { "from": "ไม่เข้าใจ", "span": [0, 9], "output_span": [0, 2],
      "candidates": [{ "rule_id": "P018", "to": "งง", "score": 0.19, "applied": true }] },
    { "from": "ครับ", "span": [12, 16], "output_span": [5, 8],
      "candidates": [{ "rule_id": "P003", "to": "คับ", "score": 0.06, "applied": true }] },
    { "from": "มาก", "span": [24, 27], "output_span": [16, 20],
      "candidates": [
        { "rule_id": "P042", "to": "ม้าก", "score": 0.21, "applied": true },
        { "rule_id": "P013", "to": "โคตรเหนื่อย", "score": 0.2, "applied": false },
        { "rule_id": "P015", "to": "เหนื่อยเกิน", "score": 0.14, "applied": false }
      ] }
  ]
}
```

```json
// request
{ "text": "เรื่องนี้ sus มาก", "direction": "to_normal" }

// response
{
  "output": "เรื่องนี้ น่าสงสัย มาก",
  "applied_rules": [
    { "from": "sus", "span": [10, 13], "output_span": [10, 18],
      "candidates": [{ "rule_id": "P100", "to": "น่าสงสัย", "score": null, "applied": true }] }
  ]
}
```

ลองยิงจาก command line:

```bash
curl -X POST http://127.0.0.1:8000/translate -H "Content-Type: application/json" \
  -d '{"text": "ไม่เข้าใจเลยครับ", "direction": "to_genz"}'
```

## การให้คะแนน (Similarity)

`src/backend/translator.py` เรียก `score_rules(text)` ใน `src/backend/similarity/similarity_engine.py` ซึ่งคืนคะแนนของทุก rule โดย `final_score = max cos(train) − CONTRAST_WEIGHT × max cos(contrast) + REGISTER_WEIGHT × register_score` `register_score` (−1 ถึง +1) คือผลโหวตของประโยคอ้างอิงที่ใกล้ข้อความที่สุด 10 ประโยคจากทุก rule ถ้าใกล้ประโยค Contrast (ภาษาทางการ) ค่าจะติดลบและคะแนนลด ใช้ rule ใน `to_genz` เฉพาะเมื่อ `final_score ≥ THRESHOLD` ค่าทั้งหมดตั้งใน `src/backend/translator.py`

- เวกเตอร์ใน `similarity/*.npy` + `metadata.json` ต้องสร้างจาก dataset ชุดเดียวกับ `data/dataset.xlsx` เพราะเลข Rule ID เลื่อนได้
  เมื่อ dataset เปลี่ยนให้รัน `python build_vector_db.py` ในโฟลเดอร์ `src/backend/similarity` (อ่าน `data/dataset.xlsx` เป็นค่าเริ่มต้น)
- ถ้าใช้ engine ไม่ได้ จะไม่กรองด้วยคะแนน ระบบทำงานแบบ keyword match ล้วน (`score` เป็น `null`)
- หาค่า `THRESHOLD`: `python tests/tune_threshold.py` (ใช้โมเดลจริง) แสดง precision / recall / F1 ของแต่ละค่าบน
  `tests/threshold_eval.csv` คอลัมน์ `convert` = คำที่ควรแปลง, `keep` = คำที่ห้ามแปลง (`*` = ห้ามแปลงทั้งประโยค) คั่นด้วย `|` 
  `python tests/tune_threshold.py` ไล่หลายชุดค่าในรอบเดียว (เพิ่ม `--sweep` เพื่อดูเส้นโค้ง) และ `python tests/tune_threshold.py --check ไฟล์.csv` ตรวจชุดทดสอบใหม่ด้วยค่าที่ตั้งไว้ ต้องรันใหม่ทุกครั้งที่ dataset หรือเวกเตอร์เปลี่ยน
