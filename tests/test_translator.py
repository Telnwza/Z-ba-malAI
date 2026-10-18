"""Run: python -m pytest -q"""
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from src.backend.rules import load_rules
from src.backend.translator import Translator


@pytest.fixture(scope="module")
def rules():
    return load_rules()


def make(rules, scores_by_target):
    """Translator with fake scores (rules referenced by Target Pattern, since P... ids can shift)."""
    ids = {r.target: r.rule_id for r in rules}
    rows = [(ids[t], s, 0, 0) for t, s in scores_by_target.items()]
    rows += [(r.rule_id, -1.0, 0, 0) for r in rules if r.target not in scores_by_target]
    return Translator(rules, scorer=lambda text: rows, threshold=0.0)   # เทสต์ใช้คะแนนปลอม ไม่ผูกกับ THRESHOLD จริง


def test_to_genz(rules):
    res = make(rules, {"งง": 0.41, "คับ": 0.22}).translate("ไม่เข้าใจครับ", "to_genz")
    assert res["output"] == "งงคับ"
    assert [(c["from"], c["span"], c["output_span"]) for c in res["applied_rules"]] == [
        ("ไม่เข้าใจ", [0, 9], [0, 2]), ("ครับ", [9, 13], [2, 5])]
    first = res["applied_rules"][0]["candidates"][0]
    assert (first["to"], first["applied"]) == ("งง", True)


def test_alternatives_grouped_under_word(rules):
    res = make(rules, {"โคตร {X}": 0.3, "{X} จัด": 0.6}).translate("เหนื่อยมาก", "to_genz")
    assert res["output"] == "เหนื่อยจัด"
    cands = res["applied_rules"][0]["candidates"]
    assert [(c["to"], c["applied"]) for c in cands] == [("เหนื่อยจัด", True), ("โคตรเหนื่อย", False)]
    one = make(rules, {"โคตร {X}": 0.3, "{X} จัด": 0.6}).translate("เหนื่อยมาก", "to_genz", top_k=1)
    assert len(one["applied_rules"][0]["candidates"]) == 1


def test_low_score_blocks_rule(rules):
    res = make(rules, {"คับ": -0.2}).translate("เรียนอาจารย์ครับ", "to_genz")
    assert res == {"output": "เรียนอาจารย์ครับ", "applied_rules": []}


def test_x_patterns(rules):
    t = make(rules, {"lowkey {X}": 0.5, "in my {X} era": 0.5})
    assert t.translate("แอบอยากไป", "to_genz")["output"] == "lowkey อยากไป"
    assert t.translate("ช่วงนี้กำลังอินกับการอ่าน", "to_genz")["output"] == "in my การอ่าน era"


def test_pattern_with_space_inside_token(rules):
    # "เงียบ ๆ" ถูกตัดเป็นคำเดียวที่มีช่องว่างข้างใน
    res = make(rules, {"ghost": 0.5}).translate("เขาหายไปเงียบ ๆ ไม่ตอบแชทเลย", "to_genz")
    assert res["output"] == "เขา ghost ไม่ตอบแชทเลย"


def test_space_between_thai_and_latin(rules):
    res = make(rules, {"sus": 0.5}).translate("เขาดูน่าสงสัยนะ", "to_genz")
    assert res["output"] == "เขาดู sus นะ"
    assert res["applied_rules"][0]["output_span"] == [6, 9]


def test_allow_profanity_unlocks_level_4(rules):
    t = make(rules, {"กู": 0.9})
    assert t.translate("ฉันเหนื่อย", "to_genz")["output"] == "ฉันเหนื่อย"
    assert t.translate("ฉันเหนื่อย", "to_genz", allow_profanity=True)["output"] == "กูเหนื่อย"


def test_max_apply(rules):
    text = "ขอบคุณครับ เดี๋ยวส่งให้ครับ"
    assert make(rules, {"คับ": 0.5}).translate(text, "to_genz")["output"] == "ขอบคุณคับ เดี๋ยวส่งให้ครับ"
    more = [replace(r, max_apply=2) if r.target == "คับ" else r for r in rules]
    assert make(more, {"คับ": 0.5}).translate(text, "to_genz")["output"] == "ขอบคุณคับ เดี๋ยวส่งให้คับ"


def test_to_normal(rules):
    t = make(rules, {})
    res = t.translate("เรื่องนี้ sus มาก", "to_normal")
    assert res["output"] == "เรื่องนี้ น่าสงสัย มาก"
    assert res["applied_rules"][0]["span"] == [10, 13]
    assert res["applied_rules"][0]["candidates"][0]["score"] is None
    assert t.translate("lowkey อยากไป", "to_normal")["output"] == "แอบอยากไป"
    # to_normal แปลงทุกที่ ไม่สน Max Apply
    assert t.translate("ขอบคุณคับ ส่งให้คับ", "to_normal")["output"] == "ขอบคุณครับ ส่งให้ครับ"


def test_api():
    from src.backend.app import app
    c = TestClient(app)
    r = c.post("/translate", json={"text": "เรื่องนี้ sus มาก", "direction": "to_normal"})
    assert r.status_code == 200
    assert set(r.json()) == {"output", "applied_rules"}
    assert c.post("/translate", json={"text": "", "direction": "to_genz"}).status_code == 422
