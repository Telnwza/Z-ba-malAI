"""Run: python -m pytest -q"""
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
    return Translator(rules, scorer=lambda text: rows)


def test_to_genz(rules):
    res = make(rules, {"งง": 0.41, "คับ": 0.22}).translate("ไม่เข้าใจครับ", "to_genz")
    assert res["output"] == "งงคับ"
    assert [(a["from"], a["to"], a["span"]) for a in res["applied_rules"]] == [
        ("ไม่เข้าใจ", "งง", [0, 9]), ("ครับ", "คับ", [9, 13])]


def test_low_score_blocks_rule(rules):
    res = make(rules, {"คับ": -0.2}).translate("เรียนอาจารย์ครับ", "to_genz")
    assert res["output"] == "เรียนอาจารย์ครับ"


def test_conflict_higher_score_wins(rules):
    res = make(rules, {"โคตร {X}": 0.3, "{X} จัด": 0.6}).translate("เหนื่อยมาก", "to_genz")
    assert res["output"] == "เหนื่อยจัด"


def test_x_patterns(rules):
    t = make(rules, {"lowkey {X}": 0.5, "in my {X} era": 0.5})
    assert t.translate("แอบอยากไป", "to_genz")["output"] == "lowkey อยากไป"
    assert t.translate("ช่วงนี้กำลังอินกับการอ่าน", "to_genz")["output"] == "in my การอ่าน era"


def test_to_normal(rules):
    t = make(rules, {})
    res = t.translate("เรื่องนี้ sus มาก", "to_normal")
    assert res["output"] == "เรื่องนี้ น่าสงสัย มาก"
    assert res["applied_rules"][0]["span"] == [10, 13]
    assert t.translate("lowkey อยากไป", "to_normal")["output"] == "แอบอยากไป"


def test_api():
    from src.backend.app import app
    c = TestClient(app)
    r = c.post("/translate", json={"text": "เรื่องนี้ sus มาก", "direction": "to_normal"})
    assert r.status_code == 200
    assert set(r.json()) == {"output", "applied_rules", "candidates"}
    assert c.post("/translate", json={"text": "", "direction": "to_genz"}).status_code == 422
