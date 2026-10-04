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
| `score` | number \| null | คะแนน final จากคนที่ 2 ทิศทาง `to_normal` เป็น null (เจอแบบตรงตัว) |
| `applied` | bool | true = ตัวที่ใช้จริงใน `output` |

หมายเหตุ

- `allow_profanity: true` จะเปิดให้ใช้ rule ระดับ 4 ด้วย เพราะคำหยาบส่วนใหญ่อยู่ระดับ 4
- `to_genz`: rule หนึ่งใช้ได้ไม่เกินค่า Max Apply ใน dataset ต่อประโยค · `to_normal`: แปลงทุกที่ที่เจอ

### Errors

| Status | เมื่อไหร่ |
| --- | --- |
| 422 | `text` ว่างหรือยาวเกิน 500, `direction` ไม่ถูก, ค่าอยู่นอกช่วง (รูปแบบมาตรฐานของ FastAPI: `{"detail": [...]}`) |

### ตัวอย่าง

ผลรันจริงกับ `data/dataset.xlsx` ตอนที่ยังไม่มี `similarity_engine.py` คะแนนจึงเป็น 0 ทุกตัว

```json
// request
{ "text": "ไม่เข้าใจเลยครับ เหนื่อยมาก", "direction": "to_genz", "top_k": 3 }

// response
{
  "output": "งงเลยคับ เหนื่อยจัด",
  "applied_rules": [
    { "from": "ไม่เข้าใจ", "span": [0, 9], "output_span": [0, 2],
      "candidates": [{ "rule_id": "P018", "to": "งง", "score": 0.0, "applied": true }] },
    { "from": "ครับ", "span": [12, 16], "output_span": [5, 8],
      "candidates": [{ "rule_id": "P003", "to": "คับ", "score": 0.0, "applied": true }] },
    { "from": "เหนื่อยมาก", "span": [17, 27], "output_span": [9, 19],
      "candidates": [
        { "rule_id": "P014", "to": "เหนื่อยจัด", "score": 0.0, "applied": true },
        { "rule_id": "P015", "to": "เหนื่อยเกิน", "score": 0.0, "applied": false },
        { "rule_id": "P016", "to": "เหนื่อยสุด", "score": 0.0, "applied": false }
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

## เชื่อมกับงานคนที่ 2

วางไฟล์ `src/backend/similarity_engine.py` โดยมีฟังก์ชัน

```python
def score_rules(text: str) -> list[tuple[str, float, float, float]]:
    """[(rule_id, final, train_score, contrast_score), ...]  rule_id = P001… จากชีท 02_PROGRAM"""
```

ถ้ายังไม่มีไฟล์นี้ ทุก rule ได้คะแนน 0 ระบบจะทำงานแบบ keyword match ล้วน
เมื่อมีแล้วให้ปรับ `THRESHOLD` ใน `src/backend/translator.py`
