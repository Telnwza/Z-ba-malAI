"""Load rules from the 02_PROGRAM sheet."""
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATASET = Path(__file__).resolve().parents[2] / "data" / "dataset.xlsx"


@dataclass
class Rule:
    """One row of the 02_PROGRAM sheet, with search forms prepared for both directions."""
    rule_id: str
    style_level: int
    content_label: str
    target: str
    triggers: list[str]        # รูปภาษาปกติที่ใช้ค้น (Normal → Gen Z)
    informal_forms: list[str]  # รูป Gen Z ที่ใช้ค้น (Gen Z → Normal)


def _split(text, sep):
    """Split a cell by sep, strip whitespace and drop empty or "-" parts (empty cell -> [])."""
    if pd.isna(text):
        return []
    return [p.strip() for p in str(text).split(sep) if p.strip() not in ("", "-")]


def load_rules(path=DATASET) -> list[Rule]:
    """Read enabled rules from the 02_PROGRAM sheet, skipping the blank rows at the end.

    target = first Target Pattern, triggers come from Source Pattern,
    informal_forms = all Target Patterns + Accepted Variants.
    """
    df = pd.read_excel(path, sheet_name="02_PROGRAM", header=1)
    rules = []
    for _, r in df.iterrows():
        if not str(r["Rule ID"]).startswith("P") or str(r["Operation"]) in ("0", "nan"):
            continue
        if str(r["Enabled"]).lower() not in ("true", "1"):
            continue
        targets = [t.rstrip(".…") for t in _split(r["Target Pattern"], "/")]
        # Source Pattern: ตัดวงเล็บอธิบายออก และข้ามส่วนที่เป็นคำอธิบาย เช่น "particle กันเอง"
        triggers = [re.sub(r"\(.*?\)", "", s).strip() for s in _split(r["Source Pattern"], "/")]
        triggers = [s for s in triggers if s and "particle" not in s and s not in targets]
        # Variants: ข้ามแบบ "คำหลัก + คำอื่น" (เช่น "slay มาก") เพราะคำหลัก match อยู่แล้ว
        variants = [v for v in _split(r["Accepted Variants"], "|")
                    if not any(t != v and t in v for t in targets)]
        rules.append(Rule(
            rule_id=str(r["Rule ID"]),
            style_level=int(r["Style Level"]),
            content_label=str(r["Content Label"]),
            target=targets[0],
            triggers=triggers,
            informal_forms=list(dict.fromkeys(targets + variants)),
        ))
    return rules
