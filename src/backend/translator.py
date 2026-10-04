"""Translation logic for both directions.

Normal -> Gen Z: source word found in the sentence + score from person 2 >= THRESHOLD -> replace.
Gen Z -> Normal: Gen Z word (Target / Variants) found in the sentence -> replace with the normal word.
Overlapping rules: highest score wins, then the longer span, then the lower Style Level.
"""
import re
from typing import NamedTuple

from pythainlp.corpus.common import thai_words
from pythainlp.tokenize import Trie, word_tokenize

from .rules import Rule, load_rules

THRESHOLD = 0.0                 # คนที่ 2 หาค่าจริงจาก test set แล้วมาแก้ตรงนี้
PROFANE = {"Profanity", "Insult"}

try:
    # งานคนที่ 2: score_rules(text) -> [(rule_id, final, train, contrast), ...]
    from .similarity_engine import score_rules
except ImportError:
    def score_rules(text):      # ยังไม่มี → ทุก rule ได้ 0 ระบบจะทำงานแบบ keyword match
        return []


class Word(NamedTuple):
    """A token of the sentence and its start offset in the text."""
    text: str
    start: int

    @property
    def end(self):
        return self.start + len(self.text)


class Proposal(NamedTuple):
    """One possible replacement: text[start:end] -> replacement, suggested by rule."""
    start: int
    end: int
    replacement: str
    rule: Rule
    score: float | None

    def overlaps(self, other):
        return self.start < other.end and other.start < self.end


class Translator:
    """Sentence translator. Build once, then call translate() repeatedly.

    rules: optional; loaded from the dataset when omitted.
    scorer: a function shaped like person 2's score_rules; tests pass fake scores here.
    """

    def __init__(self, rules: list[Rule] | None = None, scorer=score_rules):
        """Prepare the rules and the tokenizer dictionary (common Thai words + rule words, so Gen Z words tokenize correctly)."""
        self.rules = rules or load_rules()
        self.scorer = scorer
        words = set(thai_words())
        for r in self.rules:
            for form in r.triggers + r.informal_forms:
                words.update(w for w in re.split(r"\{X\}|\s+", form) if w)
        self.trie = Trie(words)

    def translate(self, text, direction, max_style_level=3, allow_profanity=False, top_k=5):
        """Translate text in the given direction ("to_genz" | "to_normal").

        Returns a dict {output, applied_rules} per the POST /translate spec: one entry per replaced
        word, each with its spans in text and output and up to top_k candidate translations.
        Rules are filtered by Style Level and Content Label first; overlaps are resolved by
        score > longer span > lower Style Level. A rule applies at most Max Apply times per
        sentence in to_genz; to_normal replaces every occurrence.
        """
        if allow_profanity:
            max_style_level = 4     # คำหยาบส่วนใหญ่อยู่ระดับ 4 ติ๊กแล้วต้องเห็นผล
        rules = [r for r in self.rules if r.style_level <= max_style_level
                 and (allow_profanity or r.content_label not in PROFANE)]

        proposals = self._propose(text, direction, rules)
        proposals.sort(key=lambda p: (-(p.score or 0), -(p.end - p.start), p.rule.style_level))
        chosen = _resolve(proposals, limit_per_rule=(direction == "to_genz"))
        return _build_response(text, chosen, proposals, top_k)

    def _propose(self, text, direction, rules):
        """Every place in text where a rule's pattern matches, as Proposals (unsorted)."""
        words = self._tokenize(text)
        scores = {row[0]: float(row[1]) for row in self.scorer(text)}
        proposals = []
        for r in rules:
            if direction == "to_genz":
                score = scores.get(r.rule_id, 0.0)
                if score < THRESHOLD:
                    continue
                pairs = [(t, r.target) for t in r.triggers]
            else:
                score = None
                pairs = [(f, self._normal_for(r, f)) for f in r.informal_forms]
            for pattern, template in pairs:
                if template is None:
                    continue
                for start, end, x in _find(words, pattern):
                    proposals.append(Proposal(start, end, _fill(template, x), r, score))
        return proposals

    def _tokenize(self, text):
        """Split text into Words, keeping whitespace tokens so offsets stay exact."""
        out, pos = [], 0
        for w in word_tokenize(text, custom_dict=self.trie, keep_whitespace=True):
            out.append(Word(w, pos))
            pos += len(w)
        return out

    @staticmethod
    def _normal_for(rule, form):
        """Normal-language replacement for this Gen Z form (a {X} pattern must pair with a {X} trigger)."""
        same = [t for t in rule.triggers if ("{X}" in t) == ("{X}" in form)]
        return same[0] if same else None


def _resolve(proposals, limit_per_rule):
    """Pick non-overlapping proposals, going through them in order (best first).

    limit_per_rule: True = a rule is used at most rule.max_apply times (to_genz only).
    """
    chosen, used = [], {}
    for p in proposals:
        rid = p.rule.rule_id
        if limit_per_rule and used.get(rid, 0) >= p.rule.max_apply:
            continue
        if any(p.overlaps(q) for q in chosen):
            continue
        chosen.append(p)
        used[rid] = used.get(rid, 0) + 1
    return chosen


def _build_response(text, chosen, proposals, top_k):
    """Apply the chosen replacements left to right and build {output, applied_rules}."""
    output, shift, applied = text, 0, []    # shift = ความยาวที่เปลี่ยนไปสะสม ใช้หาตำแหน่งใน output
    for p in sorted(chosen, key=lambda p: p.start):
        out_start = p.start + shift
        output = output[:out_start] + p.replacement + output[p.end + shift:]
        shift += len(p.replacement) - (p.end - p.start)
        applied.append({
            "from": text[p.start:p.end],
            "span": [p.start, p.end],
            "output_span": [out_start, out_start + len(p.replacement)],
            "candidates": _candidates(p, proposals, top_k),
        })
    return {"output": output, "applied_rules": applied}


def _candidates(chosen, proposals, top_k):
    """The applied proposal first, then other rules matching an overlapping span (best first)."""
    result = [_as_candidate(chosen, applied=True)]
    seen = {chosen.rule.rule_id}
    for q in proposals:
        if len(result) >= top_k:
            break
        if q.overlaps(chosen) and q.rule.rule_id not in seen:
            seen.add(q.rule.rule_id)
            result.append(_as_candidate(q, applied=False))
    return result


def _as_candidate(p, applied):
    """A Proposal as one entry of the candidates list in the response."""
    return {"rule_id": p.rule.rule_id, "to": p.replacement, "score": p.score, "applied": applied}


def _find(words, pattern):
    """Find pattern in the sentence on word boundaries; return [(start, end, X)].

    Supports one {X} slot: '{X} มาก' (X = previous word), 'แอบ {X}' (X = following words
    up to a space), 'ช่วงนี้อินกับ {X} อีกแล้ว' (X = the words in between).
    """
    words = [w for w in words if not w.text.isspace()]
    pre, _, post = (p.strip() for p in pattern.partition("{X}"))
    results = []
    for a, b in _literal(words, pre or post):
        if "{X}" not in pattern:
            results.append((words[a].start, words[b].end, None))
        elif not pre:                                   # {X} มาก
            if a > 0:
                results.append((words[a - 1].start, words[b].end, words[a - 1].text))
        elif not post:                                  # แอบ {X}
            j = b + 1
            while j + 1 < len(words) and words[j].end == words[j + 1].start:
                j += 1                                  # เก็บคำที่ติดกันจนเจอช่องว่าง
            if j < len(words):
                results.append((words[a].start, words[j].end, _join(words[b + 1:j + 1])))
        else:                                           # pre {X} post
            for c, d in _literal(words, post):
                if c > b + 1:
                    results.append((words[a].start, words[d].end, _join(words[b + 1:c])))
                    break
    return results


def _literal(words, lit):
    """Find word/phrase lit starting and ending on word boundaries; return [(first word index, last word index)]."""
    key = re.sub(r"\s+", "", lit).casefold()
    hits = []
    for i in range(len(words)):
        acc = ""
        for j in range(i, len(words)):
            acc += words[j].text.casefold()
            if acc == key:
                hits.append((i, j))
            if not key.startswith(acc):
                break
    return hits


def _join(words):
    """Concatenate the text of consecutive Words (used to fill {X})."""
    return "".join(w.text for w in words)


def _fill(template, x):
    """Replace {X} in template with x, then remove spaces between adjacent Thai characters."""
    out = template.replace("{X}", x or "").strip()
    return re.sub(r"(?<=[฀-๿]) +(?=[฀-๿])", "", out)   # ไทยติดไทยไม่เว้นวรรค
