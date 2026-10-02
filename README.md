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
│   │   └── translator.py       # หาคำในประโยค, ตัดด้วยคะแนน, เลือก rule เมื่อชนกัน, แทนที่
│   └── frontend/               # หน้าเว็บ (ยังว่าง)
├── data/
│   └── dataset.xlsx            # dataset ของคนที่ 1
├── docs/                       # เอกสาร (ยังว่าง)
├── tests/
│   └── test_translator.py      # เทสต์ translator และ API
└── requirements.txt
```

`src/backend/similarity_engine.py` ของคนที่ 2 ยังไม่มี (ดูหัวข้อ "เชื่อมกับงานคนที่ 2")

## รัน (Windows, Python 3.10+)

รันจากโฟลเดอร์ `Z-ba-malAI` ใน Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
uvicorn src.backend.app:app --reload
```

เปิด http://127.0.0.1:8000/docs เพื่อลองเรียก API · รันเทสต์: `python -m pytest -q`

## API: `POST /translate`

แปลงข้อความหนึ่งข้อความตาม `direction` แล้วคืนผลพร้อม rule ที่ใช้และคะแนน รับและคืนเป็น JSON (UTF-8)

### Request body

| Field | Type | จำเป็น | ค่าเริ่มต้น | คำอธิบาย |
| --- | --- | --- | --- | --- |
| `text` | string | ใช่ | – | ข้อความที่จะแปลง ยาว 1–500 ตัวอักษร |
| `direction` | `"to_genz"` \| `"to_normal"` | ใช่ | – | ทิศทางการแปลง |
| `max_style_level` | int 1–4 | ไม่ | 3 | ใช้เฉพาะ rule ที่ Style Level ≤ ค่านี้ |
| `allow_profanity` | bool | ไม่ | false | true = ใช้ rule ที่ Content Label เป็น Profanity/Insult ได้ |
| `top_k` | int 1–10 | ไม่ | 5 | จำนวน `candidates` สูงสุดที่คืนใน response |

### Response 200

| Field | Type | คำอธิบาย |
| --- | --- | --- |
| `output` | string | ข้อความหลังแปลง (เท่ากับข้อความเดิมถ้าไม่มี rule ไหนผ่าน) |
| `applied_rules` | array | rule ที่ใช้จริง เรียงตามตำแหน่งในประโยค ถ้าว่าง = ไม่มีการแปลง |
| `candidates` | array | ส่งทุกครั้ง ไม่เกิน `top_k` ตัว รวมทั้งตัวที่ใช้และไม่ได้ใช้ |

แต่ละตัวใน `applied_rules`

| Field | Type | คำอธิบาย |
| --- | --- | --- |
| `rule_id` | string | เช่น `"P018"` จากชีท `02_PROGRAM` (เลขเลื่อนได้เมื่อ dataset เปลี่ยน) |
| `from` | string | คำหรือวลีที่ถูกแทน |
| `to` | string | คำหรือวลีที่ใส่แทน |
| `span` | [int, int] | ตำแหน่งตัวอักษร [เริ่ม, จบ) ของ `from` ในข้อความเดิม แบบ slice ของ Python |
| `score` | number \| null | คะแนน final จากคนที่ 2 ทิศทาง `to_normal` เป็น null (เจอแบบตรงตัว) |

แต่ละตัวใน `candidates` (`to_genz` เรียงตามคะแนนมากไปน้อย)

| Field | Type | คำอธิบาย |
| --- | --- | --- |
| `rule_id` | string | รหัส rule |
| `to` | string | คำ Gen Z ของ rule (Target Pattern) |
| `score` | number \| null | คะแนน final จากคนที่ 2 ทิศทาง `to_normal` เป็น null |
| `applied` | bool | true = rule นี้อยู่ใน `applied_rules` ด้วย |

### Errors

| Status | เมื่อไหร่ |
| --- | --- |
| 422 | `text` ว่างหรือยาวเกิน 500, `direction` ไม่ถูก, ค่าอยู่นอกช่วง (รูปแบบมาตรฐานของ FastAPI: `{"detail": [...]}`) |

### ตัวอย่าง

ผลรันจริงกับ `data/dataset.xlsx` ตอนที่ยังไม่มี `similarity_engine.py` คะแนนจึงเป็น 0 ทุกตัว

```json
// request
{ "text": "ไม่เข้าใจเลยครับ", "direction": "to_genz", "top_k": 3 }

// response
{
  "output": "งงเลยคับ",
  "applied_rules": [
    { "rule_id": "P018", "from": "ไม่เข้าใจ", "to": "งง", "span": [0, 9], "score": 0.0 },
    { "rule_id": "P003", "from": "ครับ", "to": "คับ", "span": [12, 16], "score": 0.0 }
  ],
  "candidates": [
    { "rule_id": "P018", "to": "งง", "score": 0.0, "applied": true },
    { "rule_id": "P003", "to": "คับ", "score": 0.0, "applied": true }
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
    { "rule_id": "P100", "from": "sus", "to": "น่าสงสัย", "span": [10, 13], "score": null }
  ],
  "candidates": [
    { "rule_id": "P100", "to": "sus", "score": null, "applied": true }
  ]
}
```

ลองยิงจาก command line:

```bash
curl -X POST http://127.0.0.1:8000/translate -H "Content-Type: application/json" \
  -d '{"text": "ไม่เข้าใจเลยครับ", "direction": "to_genz"}'
```

## เชื่อมกับงานคนที่ 2

วางไฟล์ `src/backend/similarity_engine.py` โดยมีฟังก์ชัน

```python
def score_rules(text: str) -> list[tuple[str, float, float, float]]:
    """[(rule_id, final, train_score, contrast_score), ...]  rule_id = P001… จากชีท 02_PROGRAM"""
```

ถ้ายังไม่มีไฟล์นี้ ทุก rule ได้คะแนน 0 ระบบจะทำงานแบบ keyword match ล้วน
เมื่อมีแล้วให้ปรับ `THRESHOLD` ใน `src/backend/translator.py`
