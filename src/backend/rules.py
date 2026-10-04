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
    max_apply: int = 1         # ใช้ได้กี่ครั้งต่อประโยค (เฉพาะ Normal → Gen Z)


def load_rules(path=DATASET) -> list[Rule]:
    """Read enabled rules from the 02_PROGRAM sheet, skipping the blank rows at the end.

    target = first Target Pattern, triggers come from Source Pattern,
    informal_forms = all Target Patterns + Accepted Variants.
    max_apply = Max Apply column (defaults to 1 when empty).
    """
    df = pd.read_excel(path, sheet_name="02_PROGRAM", header=1)
    return [_to_rule(row) for _, row in df.iterrows() if _is_active(row)]


def _is_active(row):
    """True for a real rule row (not a blank row at the end) that is Enabled."""
    is_rule = str(row["Rule ID"]).startswith("P") and str(row["Operation"]) not in ("0", "nan")
    return is_rule and str(row["Enabled"]).lower() in ("true", "1")


def _to_rule(row):
    """Build a Rule from one active row."""
    targets = [t.rstrip(".…") for t in _split(row["Target Pattern"], "/")]
    return Rule(
        rule_id=str(row["Rule ID"]),
        style_level=int(row["Style Level"]),
        content_label=str(row["Content Label"]),
        target=targets[0],
        triggers=_triggers(row["Source Pattern"], targets),
        informal_forms=list(dict.fromkeys(targets + _variants(row["Accepted Variants"], targets))),
        max_apply=int(row["Max Apply"]) if pd.notna(row["Max Apply"]) else 1,
    )


def _triggers(source_pattern, targets):
    """Normal-language search forms from Source Pattern."""
    # ตัดวงเล็บอธิบายออก และข้ามส่วนที่เป็นคำอธิบาย เช่น "particle กันเอง"
    forms = [re.sub(r"\(.*?\)", "", s).strip() for s in _split(source_pattern, "/")]
    return [s for s in forms if s and "particle" not in s and s not in targets]


def _variants(accepted_variants, targets):
    """Extra Gen Z search forms from Accepted Variants."""
    # ข้ามแบบ "คำหลัก + คำอื่น" (เช่น "slay มาก") เพราะคำหลัก match อยู่แล้ว
    return [v for v in _split(accepted_variants, "|")
            if not any(t != v and t in v for t in targets)]


def _split(text, sep):
    """Split a cell by sep, strip whitespace and drop empty or "-" parts (empty cell -> [])."""
    if pd.isna(text):
        return []
    return [p.strip() for p in str(text).split(sep) if p.strip() not in ("", "-")]
