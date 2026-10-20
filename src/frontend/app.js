/* Z BA MALAI — frontend logic (vanilla JS, no build step) */
(function () {
  'use strict';

  /* ----------------------------------------------------------- config --- */

  // ?api=http://127.0.0.1:8000 overrides the API host.
  // Served by FastAPI itself (port 8000) -> same origin, no CORS needed.
  var params = new URLSearchParams(location.search);
  var API_BASE =
    params.get('api') ||
    ((location.protocol === 'file:' || ['localhost', '127.0.0.1'].includes(location.hostname)) && location.port !== '8000'
      ? 'http://localhost:8000' : location.origin);
  var ENDPOINT = API_BASE.replace(/\/+$/, '') + '/translate';

  var MAX_STYLE_LEVEL  = 3;    // 1-4
  var TOP_K            = 5;    // 1-10
  var MAX_CHARS        = 500;

  var LABELS = {
    to_genz:   { left: 'NORMAL', right: 'GEN Z'  },
    to_normal: { left: 'GEN Z',  right: 'NORMAL' }
  };

  /* -------------------------------------------------------------- dom --- */

  function $(id) { return document.getElementById(id); }

  var els = {
    input:      $('input'),
    output:     $('output'),
    status:     $('status'),
    swap:       $('swap'),
    profanity:  $('profanity'),
    labelInput: $('labelInput'),
    labelOutput:$('labelOutput'),
    more:       $('more'),
    rules:      $('rules'),
    spinner:    $('spinner'),
    counter:    $('counter'),
    count:      $('count'),
    apiLabel:   $('apiLabel')
  };

  /* ------------------------------------------------------------ state --- */

  var state = {
    direction: 'to_genz',   // 'to_genz' | 'to_normal'
    output: '',             // string from the API
    rules: [],              // normalized applied_rules
    choices: [],            // per-rule user override, null = API default
    seq: 0                  // guards against out-of-order responses
  };

  var controller = null;    // in-flight request
  var ruleEls = [];         // <details> per rule, for highlight -> rule jumps

  /* --------------------------------------------------------- helpers --- */

  function cp(str) { return Array.from(str); }          // code points
  function cpLen(str) { return cp(str).length; }

  // API indices are Python string offsets; slice on code points to stay correct.
  function sliceByCp(str, s, e) { return cp(str).slice(s, e).join(''); }

  function normalizeSpan(s) {
    if (Array.isArray(s)) return [Number(s[0]) || 0, Number(s[1]) || 0];
    if (s && typeof s === 'object')
      return [Number(s.start) || 0, Number(s.end) || 0];
    return [0, 0];
  }

  function normalizeRule(r) {
    var cands = Array.isArray(r && r.candidates) ? r.candidates : [];
    return {
      from:        typeof r.from === 'string' ? r.from : '',
      span:        normalizeSpan(r.span),
      output_span: normalizeSpan(r.output_span),
      candidates:  cands.map(function (c) {
        return {
          rule_id: c && c.rule_id != null ? String(c.rule_id) : '',
          to:      c && typeof c.to === 'string' ? c.to : '',
          score:   c && typeof c.score === 'number' ? c.score : null,
          applied: !!(c && c.applied)
        };
      })
    };
  }

  function appliedOf(rule) {
    for (var i = 0; i < rule.candidates.length; i++) {
      if (rule.candidates[i].applied) return rule.candidates[i];
    }
    return rule.candidates[0] || null;
  }

  // Text of this rule as the API produced it.
  function defaultText(rule) {
    var a = appliedOf(rule);
    return a ? a.to : sliceByCp(state.output, rule.output_span[0], rule.output_span[1]);
  }

  // Text this rule contributes right now (user override wins).
  function currentText(rule, i) {
    return state.choices[i] != null ? state.choices[i] : defaultText(rule);
  }

  /* ------------------------------------------------------------ render --- */

  function syncLabels() {
    var l = LABELS[state.direction];
    els.labelInput.textContent  = l.left;
    els.labelOutput.textContent = l.right;
  }

  function setStatus(msg, kind) {
    els.status.className = 'status' + (kind ? ' status--' + kind : '');
    if (!msg) { els.status.hidden = true; els.status.textContent = ''; return; }
    els.status.hidden = false;
    els.status.textContent = msg;
  }

  function setLoading(on) {
    els.spinner.hidden = !on;
  }

  function updateCount() {
    var n = cpLen(els.input.value);
    if (els.count) els.count.textContent = String(n);
    if (els.counter) els.counter.classList.toggle('counter--max', n >= MAX_CHARS);
  }

  /**
   * Build the text to display plus the highlight spans, applying any candidate
   * the user picked. Replacements are spliced from the end backwards so the
   * earlier offsets stay valid.
   */
  function compose() {
    var cps = cp(state.output);
    var items = [];

    for (var i = 0; i < state.rules.length; i++) {
      var rule = state.rules[i];
      var s = rule.output_span[0];
      var e = Math.min(rule.output_span[1], cps.length);
      if (s < 0) s = 0;
      if (e < s) e = s;
      if (s >= cps.length) continue;
      items.push({ i: i, s: s, e: e });
    }
    items.sort(function (a, b) { return a.s - b.s; });

    // Replacements requested by the user, skipping any overlap.
    var reps = [];
    var lastEnd = -1;
    for (var j = 0; j < items.length; j++) {
      var it = items[j];
      if (it.s < lastEnd) continue;
      lastEnd = it.e;
      var chosen = state.choices[it.i];
      if (chosen != null && chosen !== defaultText(state.rules[it.i])) {
        reps.push({ i: it.i, s: it.s, e: it.e, text: chosen });
      }
    }

    if (!reps.length) {
      return {
        text: state.output,
        spans: items.map(function (it) {
          return { i: it.i, s: it.s, e: it.e, changed: false };
        })
      };
    }

    // splice from the end backwards so earlier offsets stay valid
    var out = cps.slice();
    reps.slice().sort(function (a, b) { return b.s - a.s; }).forEach(function (r) {
      var replacement = cp(r.text);
      Array.prototype.splice.apply(out, [r.s, r.e - r.s].concat(replacement));
    });

    var changed = {};
    reps.forEach(function (r) { changed[r.i] = r; });

    var shift = 0;
    var spans = items.map(function (it) {
      var rep = changed[it.i];
      var text = rep ? rep.text : cps.slice(it.s, it.e).join('');
      var start = it.s + shift;
      var len = cpLen(text);
      shift += len - (it.e - it.s);
      return { i: it.i, s: start, e: start + len, changed: !!rep };
    });

    return { text: out.join(''), spans: spans };
  }

  function renderOutput() {
    var composed = compose();
    var text = composed.text;
    els.output.textContent = '';

    if (!text) {
      var ph = document.createElement('span');
      ph.className = 'field__ph';
      ph.textContent = 'translation';
      els.output.appendChild(ph);
      return;
    }

    var cps = cp(text);
    var spans = composed.spans.slice().sort(function (a, b) { return a.s - b.s; });
    var cursor = 0;

    spans.forEach(function (sp) {
      if (sp.s < cursor || sp.e > cps.length || sp.s >= sp.e) return;
      if (sp.s > cursor) {
        els.output.appendChild(document.createTextNode(cps.slice(cursor, sp.s).join('')));
      }
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'hl' + (sp.changed ? ' hl--changed' : '');
      btn.dataset.rule = String(sp.i);
      btn.title = 'ดูตัวเลือกของคำนี้';
      btn.textContent = cps.slice(sp.s, sp.e).join('');
      els.output.appendChild(btn);
      cursor = sp.e;
    });

    if (cursor < cps.length) {
      els.output.appendChild(document.createTextNode(cps.slice(cursor).join('')));
    }
  }

  function renderRules() {
    els.rules.textContent = '';
    ruleEls = [];

    if (!state.output) { els.more.hidden = true; return; }
    if (state.rules.length){els.more.hidden = false;}

    state.rules.forEach(function (rule, i) {
      els.rules.appendChild(buildRule(rule, i));
    });
  }

  function buildRule(rule, i) {
    var li = document.createElement('li');
    li.className = 'rule';

    var details = document.createElement('details');

    var summary = document.createElement('summary');
    var from = span('rule__from', rule.from);
    var arrow = span('rule__arrow', '→');
    var to = span('rule__to', currentText(rule, i));
    summary.appendChild(from);
    summary.appendChild(arrow);
    summary.appendChild(to);
    if (state.choices[i] != null) {
      summary.appendChild(span('rule__flag', 'แก้ไขแล้ว'));
    }

    var cands = document.createElement('div');
    cands.className = 'cands';

    rule.candidates.forEach(function (c) {
      var active = c.to === currentText(rule, i);
      var chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip' + (active ? ' chip--active' : '');
      chip.setAttribute('aria-pressed', active ? 'true' : 'false');
      chip.appendChild(span('chip__to', c.to));
      if (c.score != null) chip.appendChild(span('chip__score', c.score.toFixed(2)));
      if (c.rule_id) chip.appendChild(span('chip__id', c.rule_id));
      chip.addEventListener('click', function () { choose(i, c.to); });
      cands.appendChild(chip);
    });

    if (state.choices[i] != null) {
      var reset = document.createElement('button');
      reset.type = 'button';
      reset.className = 'rule__reset';
      reset.textContent = '↺ ใช้ค่าที่ระบบเลือก';
      reset.addEventListener('click', function () { choose(i, null); });
      cands.appendChild(reset);
    }

    details.appendChild(summary);
    details.appendChild(cands);
    li.appendChild(details);
    ruleEls[i] = details;
    return li;
  }

  function span(cls, text) {
    var el = document.createElement('span');
    el.className = cls;
    el.textContent = text;
    return el;
  }

  function choose(ruleIndex, text) {
    state.choices[ruleIndex] = text;
    renderOutput();
    renderRules();
    focusRule(ruleIndex);
  }

  function focusRule(ruleIndex) {
    var details = ruleEls[ruleIndex];
    if (!details) return;
    ruleEls.forEach(function (el) { if (el) el.open = false; });
    details.open = true;
    if (details.scrollIntoView) {
      var reduce = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      details.scrollIntoView({ block: 'nearest', behavior: reduce ? 'auto' : 'smooth' });
    }
  }

  /* ------------------------------------------------------------ request --- */

  function formatValidation(body) {
    var d = body && body.detail;
    if (Array.isArray(d) && d.length) {
      var loc = Array.isArray(d[0].loc) ? d[0].loc[d[0].loc.length - 1] : '';
      return 'ข้อมูลไม่ถูกต้อง' + (loc ? ' (' + loc + ')' : '') + ': ' + (d[0].msg || '');
    }
    if (typeof d === 'string') return d;
    return 'ข้อมูลไม่ถูกต้อง (422)';
  }

  function run() {
    var text = els.input.value;

    if (!text.trim()) {
      if (controller) controller.abort();
      controller = null;
      state.seq++;                       // invalidate anything in flight
      state.output = '';
      state.rules = [];
      state.choices = [];
      setLoading(false);
      setStatus('');
      renderOutput();
      renderRules();
      return;
    }

    if (controller) controller.abort();
    controller = new AbortController();

    var seq = ++state.seq;
    var payload = {
      text: text,
      direction: state.direction,
      max_style_level: MAX_STYLE_LEVEL,
      allow_profanity: els.profanity.checked,
      top_k: TOP_K
    };

    setLoading(true);
    setStatus('');

    fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal
    })
      .then(function (res) {
        if (seq !== state.seq) return null;
        if (res.status === 422) {
          return res.json().catch(function () { return null; }).then(function (body) {
            if (seq !== state.seq) return null;
            setLoading(false);
            setStatus(formatValidation(body), 'error');
            return null;
          });
        }
        if (!res.ok) {
          setLoading(false);
          setStatus('เซิร์ฟเวอร์ตอบกลับ ' + res.status, 'error');
          return null;
        }
        return res.json().then(function (data) {
          if (seq !== state.seq) return null;
          state.output = typeof data.output === 'string' ? data.output : '';
          state.rules = Array.isArray(data.applied_rules)
            ? data.applied_rules.map(normalizeRule)
            : [];
          state.choices = new Array(state.rules.length).fill(null);
          setLoading(false);
          setStatus('');
          renderOutput();
          renderRules();
          return null;
        });
      })
      .catch(function (err) {
        if (err && err.name === 'AbortError') return;
        if (seq !== state.seq) return;
        setLoading(false);
        setStatus('เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ — ตรวจสอบว่า backend รันอยู่ที่ ' + API_BASE, 'error');
      });
  }

  /* ------------------------------------------------------------- events --- */

  // Send immediately on every input change; no debounce timer.
  els.input.addEventListener('input', function () {
    updateCount();
    run();
  });

  els.swap.addEventListener('click', function () {
    state.direction = state.direction === 'to_genz' ? 'to_normal' : 'to_genz';
    syncLabels();
    run();
  });

  els.profanity.addEventListener('change', run);

  els.output.addEventListener('click', function (ev) {
    var target = ev.target;
    while (target && target !== els.output && !target.dataset.rule) {
      target = target.parentNode;
    }
    if (target && target !== els.output && target.dataset.rule != null) {
      focusRule(Number(target.dataset.rule));
    }
  });

  /* --------------------------------------------------------------- init --- */

  syncLabels();
  updateCount();
  if (els.apiLabel) els.apiLabel.textContent = API_BASE;
  renderOutput();
  renderRules();
})();
