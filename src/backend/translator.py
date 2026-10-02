"""Translation logic for both directions.

Normal -> Gen Z: source word found in the sentence + score from person 2 >= THRESHOLD -> replace.
Gen Z -> Normal: Gen Z word (Target / Variants) found in the sentence -> replace with the normal word.
Overlapping rules: highest score wins, then the longer span, then the lower Style Level.
"""
import re

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

        Returns a dict {output, applied_rules, candidates} per the POST /translate spec.
        Rules are filtered by Style Level and Content Label first; overlaps are resolved by
        score > longer span > lower Style Level (each rule applies at most once per sentence).
        """
        allowed = [r for r in self.rules if r.style_level <= max_style_level
                   and (allow_profanity or r.content_label not in PROFANE)]
        tokens = self._tokenize(text)
        scores = {row[0]: float(row[1]) for row in self.scorer(text)}

        proposals = []  # (start, end, replacement, rule, score)
        for r in allowed:
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
                for start, end, x in _find(tokens, pattern):
                    proposals.append((start, end, _fill(template, x), r, score))

        # แก้ rule ชนกัน
        proposals.sort(key=lambda p: (-(p[4] or 0), -(p[1] - p[0]), p[3].style_level))
        chosen, used = [], set()
        for p in proposals:
            if p[3].rule_id in used or any(p[0] < q[1] and q[0] < p[1] for q in chosen):
                continue
            chosen.append(p)
            used.add(p[3].rule_id)

        output = text
        for start, end, rep, _, _ in sorted(chosen, key=lambda p: -p[0]):
            output = output[:start] + rep + output[end:]

        applied = [{"rule_id": r.rule_id, "from": text[s:e], "to": rep, "span": [s, e], "score": sc}
                   for s, e, rep, r, sc in sorted(chosen, key=lambda p: p[0])]

        # candidates: rule ที่คะแนนสูงสุด top_k ตัว (รวมตัวที่เจอคำในประโยค)
        by_id = {r.rule_id: r for r in allowed}
        ids = list(dict.fromkeys([p[3].rule_id for p in proposals] + list(scores)))
        ids = [i for i in ids if i in by_id]
        if direction == "to_genz":
            ids.sort(key=lambda i: -scores.get(i, 0.0))
        candidates = [{"rule_id": i, "to": by_id[i].target,
                       "score": scores.get(i, 0.0) if direction == "to_genz" else None,
                       "applied": i in used} for i in ids[:top_k]]
        return {"output": output, "applied_rules": applied, "candidates": candidates}

    def _tokenize(self, text):
        """Return [(word, start offset)]."""
        out, pos = [], 0
        for w in word_tokenize(text, custom_dict=self.trie, keep_whitespace=True):
            out.append((w, pos))
            pos += len(w)
        return out

    @staticmethod
    def _normal_for(rule, form):
        """Normal-language replacement for this Gen Z form (a {X} pattern must pair with a {X} trigger)."""
        same = [t for t in rule.triggers if ("{X}" in t) == ("{X}" in form)]
        return same[0] if same else None


def _find(tokens, pattern):
    """Find pattern in the sentence on word boundaries; return [(start, end, X)].

    Supports one {X} slot: '{X} มาก' (X = previous word), 'แอบ {X}' (X = following words
    up to a space), 'ช่วงนี้อินกับ {X} อีกแล้ว' (X = the words in between).
    """
    words = [(w, s) for w, s in tokens if not w.isspace()]
    pre, _, post = (p.strip() for p in pattern.partition("{X}"))
    has_x = "{X}" in pattern
    results = []
    for a, b in _literal(words, pre or post):
        if not has_x:
            results.append((words[a][1], _end(words[b]), None))
        elif not pre:                                   # {X} มาก
            if a > 0:
                results.append((words[a - 1][1], _end(words[b]), words[a - 1][0]))
        elif not post:                                  # แอบ {X}
            j = b + 1
            while j + 1 < len(words) and _end(words[j]) == words[j + 1][1]:
                j += 1                                  # เก็บคำที่ติดกันจนเจอช่องว่าง
            if j < len(words):
                results.append((words[a][1], _end(words[j]), "".join(w for w, _ in words[b + 1:j + 1])))
        else:                                           # pre {X} post
            for c, d in _literal(words, post):
                if c > b + 1:
                    x = "".join(w for w, _ in words[b + 1:c])
                    results.append((words[a][1], _end(words[d]), x))
                    break
    return results


def _literal(words, lit):
    """Find word/phrase lit starting and ending on word boundaries; return [(first word index, last word index)]."""
    key = re.sub(r"\s+", "", lit).casefold()
    hits = []
    for i in range(len(words)):
        acc = ""
        for j in range(i, len(words)):
            acc += words[j][0].casefold()
            if acc == key:
                hits.append((i, j))
            if not key.startswith(acc):
                break
    return hits


def _end(word):
    """Exclusive end offset of a (word, start) pair."""
    return word[1] + len(word[0])


def _fill(template, x):
    """Replace {X} in template with x, then remove spaces between adjacent Thai characters."""
    out = template.replace("{X}", x or "").strip()
    return re.sub(r"(?<=[฀-๿]) +(?=[฀-๿])", "", out)   # ไทยติดไทยไม่เว้นวรรค
