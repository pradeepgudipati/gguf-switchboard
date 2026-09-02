(function () {
  'use strict';

  var MODEL_STORAGE_KEY = 'gguf-switchboard-swagger-model';
  var allModels = [];

  // ---------- utilities ----------

  function $(id) {
    return document.getElementById(id);
  }

  function formatModelLabel(m) {
    if (!m || !m.id) return '';
    var parts = [m.id];
    var ctx = m.context_size || m.max_context_length;
    if (ctx) parts.push('ctx ' + ctx);
    if (m.min_vram_gb) parts.push('~' + m.min_vram_gb + 'GB');
    return parts.join(' · ');
  }

  var MODEL_GROUPS = [
    { label: 'Chat', kinds: ['chat', 'coder'] },
    { label: 'Vision', kinds: ['vision'] },
    { label: 'Embedding', kinds: ['embedding'] },
    { label: 'Reranker', kinds: ['reranker'] },
    { label: 'Audio', kinds: ['audio'] }
  ];

  function groupModels(models) {
    var groups = MODEL_GROUPS.map(function (g) { return { label: g.label, items: [] }; });
    var other = { label: 'Other', items: [] };
    (models || []).forEach(function (m) {
      var kind = (m && m.kind ? String(m.kind) : '').toLowerCase();
      var target = other;
      for (var i = 0; i < MODEL_GROUPS.length; i++) {
        if (MODEL_GROUPS[i].kinds.indexOf(kind) !== -1) { target = groups[i]; break; }
      }
      target.items.push(m);
    });
    return groups.concat([other])
      .filter(function (g) { return g.items.length > 0; })
      .map(function (g) {
        g.items.sort(function (a, b) { return String(a.id).localeCompare(String(b.id)); });
        return g;
      });
  }

  function badgeClassForLocation(location) {
    return 'badge badge-' + location;
  }

  function humanizeLocation(location) {
    var map = {
      structured_tool_calls: 'Structured tool call',
      plain_text_json_dump: 'Dumped as plain text',
      leaked_into_reasoning: 'Leaked into reasoning',
      no_tool_call_detected: 'No tool call detected'
    };
    return map[location] || location;
  }

  function jsonPre(value) {
    var pre = document.createElement('pre');
    pre.className = 'json-view';
    pre.textContent = JSON.stringify(value, null, 2);
    return pre;
  }

  function rawToggle(label, value) {
    var details = document.createElement('details');
    details.className = 'raw-toggle';
    var summary = document.createElement('summary');
    summary.textContent = label;
    details.appendChild(summary);
    details.appendChild(jsonPre(value));
    return details;
  }

  function classificationBlock(classification) {
    var wrap = document.createElement('div');

    var badge = document.createElement('span');
    badge.className = badgeClassForLocation(classification.location);
    badge.textContent = humanizeLocation(classification.location);
    wrap.appendChild(badge);

    if (classification.detected_json_snippet) {
      var snippetPre = document.createElement('pre');
      snippetPre.className = 'json-view';
      snippetPre.style.marginTop = '10px';
      snippetPre.textContent = classification.detected_json_snippet;
      wrap.appendChild(snippetPre);
    }

    if (classification.notes && classification.notes.length) {
      var ul = document.createElement('ul');
      ul.className = 'notes-list';
      classification.notes.forEach(function (note) {
        var li = document.createElement('li');
        li.textContent = note;
        ul.appendChild(li);
      });
      wrap.appendChild(ul);
    }

    return wrap;
  }

  function setStatus(el, text, isError) {
    el.textContent = text || '';
    el.classList.toggle('error', !!isError);
  }

  async function postJson(path, body, extraHeaders) {
    var headers = { 'Content-Type': 'application/json' };
    if (extraHeaders) {
      Object.keys(extraHeaders).forEach(function (k) { headers[k] = extraHeaders[k]; });
    }
    var res = await fetch(path, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(body)
    });
    var text = await res.text();
    var data;
    try {
      data = text ? JSON.parse(text) : {};
    } catch (e) {
      data = { error: { message: text } };
    }
    if (!res.ok) {
      var message =
        (data && data.error && data.error.message) || 'HTTP ' + res.status;
      var err = new Error(message);
      err.body = data;
      throw err;
    }
    return data;
  }

  // ---------- model list ----------

  function populateSelect(select, models, selected) {
    select.innerHTML = '';
    groupModels(models).forEach(function (group) {
      var og = document.createElement('optgroup');
      og.label = group.label;
      group.items.forEach(function (m) {
        var opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = formatModelLabel(m);
        og.appendChild(opt);
      });
      select.appendChild(og);
    });
    if (selected && models.some(function (m) { return m.id === selected; })) {
      select.value = selected;
    }
  }

  function currentModel() {
    return localStorage.getItem(MODEL_STORAGE_KEY) || '';
  }

  async function loadModels() {
    var res = await fetch('/v1/models');
    var data = await res.json();
    allModels = data.data || [];
    var selected = currentModel();
    [
      'inspect-model',
      'template-model',
      'battery-model',
      'compare-model-a',
      'compare-model-b'
    ].forEach(function (id) {
      populateSelect($(id), allModels, selected);
    });
  }

  // ---------- tabs ----------

  var tabActivationHandlers = {};

  function onTabActivated(tab, fn) {
    tabActivationHandlers[tab] = fn;
  }

  function initTabs() {
    var buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        buttons.forEach(function (b) { b.classList.remove('active'); });
        document.querySelectorAll('.tab-panel').forEach(function (p) {
          p.classList.remove('active');
        });
        btn.classList.add('active');
        $('panel-' + btn.dataset.tab).classList.add('active');
        var handler = tabActivationHandlers[btn.dataset.tab];
        if (handler) handler();
      });
    });
  }

  // ---------- Inspect tab ----------

  function defaultInspectBody() {
    return {
      messages: [
        { role: 'user', content: 'Call the echo tool with message set to "hello".' }
      ],
      tools: [
        {
          type: 'function',
          function: {
            name: 'echo',
            description: 'Echo a message back.',
            parameters: {
              type: 'object',
              properties: { message: { type: 'string' } },
              required: ['message']
            }
          }
        }
      ],
      tool_choice: 'required',
      max_tokens: 256
    };
  }

  function renderInspectResult(container, data) {
    container.innerHTML = '';
    data.classifications.forEach(function (classification, i) {
      var block = document.createElement('div');
      if (data.classifications.length > 1) {
        var h = document.createElement('div');
        h.style.fontWeight = '700';
        h.style.marginBottom = '6px';
        h.textContent = 'Choice ' + i;
        block.appendChild(h);
      }
      block.appendChild(classificationBlock(classification));
      container.appendChild(block);
      if (i < data.classifications.length - 1) {
        container.appendChild(document.createElement('hr'));
      }
    });
    container.appendChild(rawToggle('Raw response', data.raw_response));
  }

  function initInspectTab() {
    $('inspect-body').value = JSON.stringify(defaultInspectBody(), null, 2);
    var tgt = makeTargetWidget('inspect', 'Target');
    $('inspect-target').appendChild(tgt.el);
    var recent = mountRecentRuns($('inspect-recent-body'), 'inspect');
    onTabActivated('inspect', recent.refresh);

    $('inspect-run').addEventListener('click', async function () {
      var statusEl = $('inspect-status');
      var resultEl = $('inspect-result');
      var target, headers;
      try {
        target = tgt.getTarget();
        headers = targetHeaders(target);
      } catch (e) {
        setStatus(statusEl, e.message, true);
        return;
      }
      var model = (target && target.model) || $('inspect-model').value;
      if (!model) {
        setStatus(statusEl, target ? 'Enter the model name for the custom endpoint.' : 'Select a model first.', true);
        return;
      }
      var body;
      try {
        body = JSON.parse($('inspect-body').value);
      } catch (e) {
        setStatus(statusEl, 'Request body is not valid JSON: ' + e.message, true);
        return;
      }
      body.model = model;

      var btn = $('inspect-run');
      btn.disabled = true;
      setStatus(statusEl, 'Running…');
      resultEl.innerHTML = '';
      try {
        var data = await postJson('/v1/conformance/inspect', body, headers);
        setStatus(statusEl, 'Done.');
        renderInspectResult(resultEl, data);
        recent.refresh();
      } catch (e) {
        setStatus(statusEl, e.message, true);
      } finally {
        btn.disabled = false;
      }
    });
  }

  // ---------- Resolved Template tab ----------

  function defaultTemplateBody() {
    return {
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'Call the echo tool with message set to "hello".' }
      ],
      tools: [
        {
          type: 'function',
          function: {
            name: 'echo',
            description: 'Echo a message back.',
            parameters: {
              type: 'object',
              properties: { message: { type: 'string' } },
              required: ['message']
            }
          }
        }
      ]
    };
  }

  function renderTemplateResult(container, data) {
    container.innerHTML = '';
    if (!data.resolved) {
      var banner = document.createElement('div');
      banner.className = 'banner banner-warn';
      banner.textContent =
        data.error ||
        'Server did not return a live-resolved prompt; showing raw template source only.';
      container.appendChild(banner);
    }
    if (data.prompt) {
      var pre = document.createElement('pre');
      pre.className = 'prompt-view';
      pre.textContent = data.prompt;
      container.appendChild(pre);
    } else if (data.template_source) {
      var srcLabel = document.createElement('div');
      srcLabel.className = 'field-label';
      srcLabel.textContent = 'Raw template source (unresolved)';
      container.appendChild(srcLabel);
      var srcPre = document.createElement('pre');
      srcPre.className = 'prompt-view';
      srcPre.textContent = data.template_source;
      container.appendChild(srcPre);
    } else if (data.resolved) {
      var empty = document.createElement('div');
      empty.className = 'banner banner-warn';
      empty.textContent = 'Server returned no prompt text.';
      container.appendChild(empty);
    }
  }

  function initTemplateTab() {
    $('template-body').value = JSON.stringify(defaultTemplateBody(), null, 2);
    var tgt = makeTargetWidget('template', 'Target');
    $('template-target').appendChild(tgt.el);
    var recent = mountRecentRuns($('template-recent-body'), 'resolve_template');
    onTabActivated('template', recent.refresh);

    $('template-run').addEventListener('click', async function () {
      var statusEl = $('template-status');
      var resultEl = $('template-result');
      var target, headers;
      try {
        target = tgt.getTarget();
        headers = targetHeaders(target);
      } catch (e) {
        setStatus(statusEl, e.message, true);
        return;
      }
      var model = (target && target.model) || $('template-model').value;
      if (!model) {
        setStatus(statusEl, target ? 'Enter the model name for the custom endpoint.' : 'Select a model first.', true);
        return;
      }
      var body;
      try {
        body = JSON.parse($('template-body').value);
      } catch (e) {
        setStatus(statusEl, 'Request body is not valid JSON: ' + e.message, true);
        return;
      }
      body.model = model;

      var btn = $('template-run');
      btn.disabled = true;
      setStatus(statusEl, 'Resolving…');
      resultEl.innerHTML = '';
      try {
        var data = await postJson('/v1/conformance/resolve-template', body, headers);
        setStatus(statusEl, data.resolved ? 'Resolved.' : 'Fallback (see below).');
        renderTemplateResult(resultEl, data);
        recent.refresh();
      } catch (e) {
        setStatus(statusEl, e.message, true);
      } finally {
        btn.disabled = false;
      }
    });
  }

  // ---------- Battery tab ----------

  function renderBatteryResult(container, report) {
    container.innerHTML = '';

    var summary = document.createElement('div');
    summary.className = 'badge ' + (report.overall_pass ? 'badge-pass' : 'badge-fail');
    summary.textContent = report.overall_pass ? 'All cases passed' : 'Some cases failed';
    container.appendChild(summary);

    var table = document.createElement('table');
    table.className = 'battery-table';
    table.innerHTML =
      '<thead><tr><th>Case</th><th>Result</th><th>Detail</th></tr></thead>';
    var tbody = document.createElement('tbody');
    report.cases.forEach(function (c) {
      var tr = document.createElement('tr');

      var caseTd = document.createElement('td');
      caseTd.textContent = c.case;
      tr.appendChild(caseTd);

      var resultTd = document.createElement('td');
      var badge = document.createElement('span');
      badge.className = 'badge ' + (c.pass ? 'badge-pass' : 'badge-fail');
      badge.textContent = c.pass ? 'Pass' : 'Fail';
      resultTd.appendChild(badge);
      tr.appendChild(resultTd);

      var detailTd = document.createElement('td');
      if (c.reason) {
        var reasonEl = document.createElement('div');
        reasonEl.className = 'case-reason';
        reasonEl.textContent = c.reason;
        detailTd.appendChild(reasonEl);
      }
      detailTd.appendChild(rawToggle('Classification', c.classification));
      tr.appendChild(detailTd);

      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    container.appendChild(table);
  }

  function initBatteryTab() {
    var tgt = makeTargetWidget('battery', 'Target');
    $('battery-target').appendChild(tgt.el);
    var recent = mountRecentRuns($('battery-recent-body'), 'battery');
    onTabActivated('battery', recent.refresh);

    $('battery-run').addEventListener('click', async function () {
      var statusEl = $('battery-status');
      var resultEl = $('battery-result');
      var target, headers;
      try {
        target = tgt.getTarget();
        headers = targetHeaders(target);
      } catch (e) {
        setStatus(statusEl, e.message, true);
        return;
      }
      var model = (target && target.model) || $('battery-model').value;
      if (!model) {
        setStatus(statusEl, target ? 'Enter the model name for the custom endpoint.' : 'Select a model first.', true);
        return;
      }
      var btn = $('battery-run');
      btn.disabled = true;
      setStatus(statusEl, 'Running battery (may take a while)…');
      resultEl.innerHTML = '';
      try {
        var data = await postJson(
          '/v1/conformance/battery/' + encodeURIComponent(model),
          {},
          headers
        );
        setStatus(statusEl, 'Done.');
        renderBatteryResult(resultEl, data);
        recent.refresh();
      } catch (e) {
        setStatus(statusEl, e.message, true);
      } finally {
        btn.disabled = false;
      }
    });
  }

  // ---------- Compare tab ----------

  function renderCompareResult(container, report) {
    container.innerHTML = '';

    var summary = document.createElement('div');
    summary.className = 'diff-summary';
    summary.textContent = summarizeCompare(report);
    container.appendChild(summary);

    [report.result_a, report.result_b].forEach(function (result) {
      var col = document.createElement('div');
      col.className = 'compare-col';
      var h = document.createElement('h4');
      h.textContent = result.model;
      col.appendChild(h);

      if (result.error) {
        var err = document.createElement('div');
        err.className = 'banner banner-error';
        err.textContent = result.error;
        col.appendChild(err);
      } else if (result.battery_case) {
        var badge = document.createElement('span');
        badge.className =
          'badge ' + (result.battery_case.pass ? 'badge-pass' : 'badge-fail');
        badge.textContent = result.battery_case.pass ? 'Pass' : 'Fail';
        col.appendChild(badge);
        if (result.battery_case.reason) {
          var reason = document.createElement('div');
          reason.className = 'case-reason';
          reason.style.marginTop = '8px';
          reason.textContent = result.battery_case.reason;
          col.appendChild(reason);
        }
        col.appendChild(classificationBlock(result.battery_case.classification));
      } else if (result.inspect) {
        result.inspect.classifications.forEach(function (c) {
          col.appendChild(classificationBlock(c));
        });
        col.appendChild(rawToggle('Raw response', result.inspect.raw_response));
      }

      container.appendChild(col);
    });
  }

  function summarizeCompare(report) {
    var a = report.result_a;
    var b = report.result_b;
    if (a.error || b.error) return 'One or both models errored — see below.';

    var locA = a.battery_case
      ? a.battery_case.classification.location
      : a.inspect && a.inspect.classifications[0] && a.inspect.classifications[0].location;
    var locB = b.battery_case
      ? b.battery_case.classification.location
      : b.inspect && b.inspect.classifications[0] && b.inspect.classifications[0].location;

    if (a.battery_case && b.battery_case) {
      if (a.battery_case.pass === b.battery_case.pass) {
        return a.battery_case.pass
          ? 'Both models passed this case.'
          : 'Both models failed this case.';
      }
      return a.model + ' ' + (a.battery_case.pass ? 'passed' : 'failed') +
        ', ' + b.model + ' ' + (b.battery_case.pass ? 'passed' : 'failed') + '.';
    }

    if (locA && locB) {
      return locA === locB
        ? 'Both models produced the same outcome: ' + humanizeLocation(locA) + '.'
        : a.model + ': ' + humanizeLocation(locA) + '  —  ' +
          b.model + ': ' + humanizeLocation(locB);
    }

    return 'Ran both models — see individual results below.';
  }

  function updateCompareModeUi() {
    var mode = $('compare-mode').value;
    $('compare-case-row').style.display = mode === 'battery_case' ? '' : 'none';
    $('compare-request-row').style.display = mode === 'custom_request' ? '' : 'none';
  }

  function initCompareTab() {
    $('compare-body').value = JSON.stringify(defaultInspectBody(), null, 2);
    $('compare-mode').addEventListener('change', updateCompareModeUi);
    updateCompareModeUi();

    var tgtA = makeTargetWidget('compare-a', 'Target A');
    var tgtB = makeTargetWidget('compare-b', 'Target B');
    $('compare-target-a').appendChild(tgtA.el);
    $('compare-target-b').appendChild(tgtB.el);
    var recent = mountRecentRuns($('compare-recent-body'), 'compare');
    onTabActivated('compare', recent.refresh);

    $('compare-run').addEventListener('click', async function () {
      var statusEl = $('compare-status');
      var resultEl = $('compare-result');
      var targetA, targetB, headers;
      try {
        targetA = tgtA.getTarget();
        targetB = tgtB.getTarget();
        headers = Object.assign(
          {},
          targetHeaders(targetA, '-a'),
          targetHeaders(targetB, '-b')
        );
      } catch (e) {
        setStatus(statusEl, e.message, true);
        return;
      }
      var modelA = (targetA && targetA.model) || $('compare-model-a').value;
      var modelB = (targetB && targetB.model) || $('compare-model-b').value;
      if (!modelA || !modelB) {
        setStatus(statusEl, 'Both sides need a model (select one, or enter it for a custom endpoint).', true);
        return;
      }

      var mode = $('compare-mode').value;
      var payload = { model_a: modelA, model_b: modelB, mode: mode };

      if (mode === 'battery_case') {
        payload.case = $('compare-case').value;
      } else {
        try {
          payload.request = JSON.parse($('compare-body').value);
        } catch (e) {
          setStatus(statusEl, 'Request body is not valid JSON: ' + e.message, true);
          return;
        }
        // ChatCompletionRequest.model is required at deserialization time even
        // though the server overwrites it per side — fill in a placeholder if
        // the user's edited JSON dropped it.
        if (!payload.request.model) {
          payload.request.model = modelA;
        }
      }

      var btn = $('compare-run');
      btn.disabled = true;
      setStatus(statusEl, 'Swapping models and running — this can take a while…');
      resultEl.innerHTML = '';
      try {
        var data = await postJson('/v1/conformance/compare', payload, headers);
        setStatus(statusEl, 'Done.');
        renderCompareResult(resultEl, data);
        recent.refresh();
      } catch (e) {
        setStatus(statusEl, e.message, true);
      } finally {
        btn.disabled = false;
      }
    });
  }

  // ---------- History tab ----------

  function passBadge(passed) {
    var span = document.createElement('span');
    if (passed === true) {
      span.className = 'badge badge-pass';
      span.textContent = 'Pass';
    } else if (passed === false) {
      span.className = 'badge badge-fail';
      span.textContent = 'Fail';
    } else {
      span.className = 'badge';
      span.textContent = '—';
    }
    return span;
  }

  function historyQueryString() {
    var params = [];
    var kind = $('history-kind').value;
    var model = $('history-model').value.trim();
    if (kind) params.push('kind=' + encodeURIComponent(kind));
    if (model) params.push('model=' + encodeURIComponent(model));
    params.push('limit=200');
    return '?' + params.join('&');
  }

  async function refreshHistory() {
    var statusEl = $('history-status');
    var resultEl = $('history-result');
    setStatus(statusEl, 'Loading…');
    try {
      var res = await fetch('/v1/conformance/history' + historyQueryString());
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var rows = await res.json();
      setStatus(statusEl, rows.length + ' run' + (rows.length === 1 ? '' : 's'));
      renderHistory(resultEl, rows);
    } catch (e) {
      setStatus(statusEl, e.message, true);
    }
  }

  function renderHistory(container, rows) {
    container.innerHTML = '';
    if (!rows.length) {
      container.textContent = 'No runs recorded yet.';
      return;
    }
    container.appendChild(buildHistoryTable(rows, true));
  }

  // Shared table builder for the History tab and the per-tab "Recent runs"
  // panels. `showKind` hides the Kind column when every row is the same kind.
  function buildHistoryTable(rows, showKind) {
    var table = document.createElement('table');
    table.className = 'battery-table';
    var cols = showKind
      ? '<th>When</th><th>Kind</th><th>Model</th><th>Summary</th><th>Result</th>'
      : '<th>When</th><th>Model</th><th>Summary</th><th>Result</th>';
    var span = showKind ? 5 : 4;
    var thead = document.createElement('thead');
    thead.innerHTML = '<tr>' + cols + '</tr>';
    table.appendChild(thead);
    var tbody = document.createElement('tbody');

    rows.forEach(function (row) {
      var tr = document.createElement('tr');
      tr.className = 'history-row';
      tr.style.cursor = 'pointer';

      var when = document.createElement('td');
      var d = new Date(row.run_at);
      when.textContent = isNaN(d.getTime()) ? row.run_at : d.toLocaleString();
      tr.appendChild(when);

      if (showKind) {
        var kind = document.createElement('td');
        kind.textContent = row.kind;
        tr.appendChild(kind);
      }

      var model = document.createElement('td');
      model.textContent = row.model_b ? row.model + ' ↔ ' + row.model_b : (row.model || '—');
      tr.appendChild(model);

      var summary = document.createElement('td');
      summary.textContent = row.summary;
      tr.appendChild(summary);

      var result = document.createElement('td');
      result.appendChild(passBadge(row.passed));
      tr.appendChild(result);

      var detailRow = document.createElement('tr');
      var detailCell = document.createElement('td');
      detailCell.colSpan = span;
      detailCell.style.display = 'none';
      detailRow.appendChild(detailCell);

      tr.addEventListener('click', async function () {
        if (detailCell.style.display === 'none') {
          detailCell.style.display = '';
          if (!detailCell.dataset.loaded) {
            detailCell.textContent = 'Loading…';
            try {
              var res = await fetch('/v1/conformance/history/' + row.id);
              if (!res.ok) throw new Error('HTTP ' + res.status);
              var full = await res.json();
              detailCell.textContent = '';
              detailCell.appendChild(jsonPre(full.detail));
              detailCell.dataset.loaded = '1';
            } catch (e) {
              detailCell.textContent = 'Failed to load detail: ' + e.message;
            }
          }
        } else {
          detailCell.style.display = 'none';
        }
      });

      tbody.appendChild(tr);
      tbody.appendChild(detailRow);
    });

    table.appendChild(tbody);
    return table;
  }

  // A compact, self-contained "Recent runs" panel scoped to one run kind,
  // for embedding under a tab's result area.
  function mountRecentRuns(container, kind) {
    container.innerHTML = '';
    var body = document.createElement('div');
    container.appendChild(body);

    async function refresh() {
      try {
        var res = await fetch(
          '/v1/conformance/history?limit=20&kind=' + encodeURIComponent(kind)
        );
        if (!res.ok) throw new Error('HTTP ' + res.status);
        var rows = await res.json();
        body.innerHTML = '';
        if (!rows.length) {
          body.textContent = 'No ' + kind + ' runs recorded yet.';
          return;
        }
        body.appendChild(buildHistoryTable(rows, false));
      } catch (e) {
        body.textContent = 'Could not load history: ' + e.message;
      }
    }

    return { refresh: refresh };
  }

  // ---------- test-target (custom endpoint) widget ----------

  // In-memory only — API keys are never written to localStorage or sent
  // anywhere except the outbound X-Conformance-Api-Key header.
  var apiKeyMemory = {};

  function makeTargetWidget(idPrefix, label) {
    var wrap = document.createElement('div');
    wrap.className = 'target-widget';

    var lsKey = 'ggs-conformance-target-' + idPrefix;
    var saved = {};
    try { saved = JSON.parse(localStorage.getItem(lsKey) || '{}'); } catch (e) { saved = {}; }

    var row = document.createElement('div');
    row.className = 'field-row';

    var legend = document.createElement('label');
    legend.textContent = label;
    row.appendChild(legend);

    var modeSel = document.createElement('select');
    modeSel.innerHTML =
      '<option value="local">GGUF Switchboard (local)</option>' +
      '<option value="custom">Custom OpenAI-compatible endpoint</option>';
    modeSel.value = saved.mode === 'custom' ? 'custom' : 'local';
    row.appendChild(modeSel);
    wrap.appendChild(row);

    var custom = document.createElement('div');
    custom.className = 'field-row target-custom';

    var baseUrl = document.createElement('input');
    baseUrl.type = 'text';
    baseUrl.placeholder = 'Base URL, e.g. https://api.openai.com/v1';
    baseUrl.value = saved.baseUrl || '';
    baseUrl.style.minWidth = '260px';

    var modelName = document.createElement('input');
    modelName.type = 'text';
    modelName.placeholder = 'Model name (as the endpoint expects)';
    modelName.value = saved.model || '';

    var apiKey = document.createElement('input');
    apiKey.type = 'password';
    apiKey.placeholder = 'API key (not stored)';
    apiKey.value = apiKeyMemory[idPrefix] || '';
    apiKey.autocomplete = 'off';

    custom.appendChild(baseUrl);
    custom.appendChild(modelName);
    custom.appendChild(apiKey);
    wrap.appendChild(custom);

    function sync() {
      custom.style.display = modeSel.value === 'custom' ? 'flex' : 'none';
    }
    function persist() {
      try {
        localStorage.setItem(lsKey, JSON.stringify({
          mode: modeSel.value,
          baseUrl: baseUrl.value.trim(),
          model: modelName.value.trim()
        }));
      } catch (e) { /* ignore */ }
      apiKeyMemory[idPrefix] = apiKey.value;
    }
    modeSel.addEventListener('change', function () { sync(); persist(); });
    [baseUrl, modelName, apiKey].forEach(function (el) {
      el.addEventListener('change', persist);
    });
    sync();

    return {
      el: wrap,
      // null => local; otherwise {baseUrl, model, apiKey}
      getTarget: function () {
        if (modeSel.value !== 'custom') return null;
        var b = baseUrl.value.trim();
        if (!b) throw new Error('Custom endpoint selected but Base URL is empty');
        return { baseUrl: b, model: modelName.value.trim(), apiKey: apiKey.value };
      }
    };
  }

  function targetHeaders(target, suffix) {
    if (!target) return {};
    suffix = suffix || '';
    var h = {};
    h['X-Conformance-Base-Url' + suffix] = target.baseUrl;
    if (target.model) h['X-Conformance-Model' + suffix] = target.model;
    if (target.apiKey) h['X-Conformance-Api-Key' + suffix] = target.apiKey;
    return h;
  }

  function initHistoryTab() {
    $('history-refresh').addEventListener('click', refreshHistory);
    $('history-kind').addEventListener('change', refreshHistory);
    $('history-model').addEventListener('keydown', function (e) {
      if (e.key === 'Enter') refreshHistory();
    });
    $('history-clear').addEventListener('click', async function () {
      if (!window.confirm('Delete the entire conformance run history?')) return;
      try {
        var res = await fetch('/v1/conformance/history', { method: 'DELETE' });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        refreshHistory();
      } catch (e) {
        setStatus($('history-status'), e.message, true);
      }
    });
    onTabActivated('history', refreshHistory);
  }

  // ---------- boot ----------

  document.addEventListener('DOMContentLoaded', async function () {
    initTabs();
    initInspectTab();
    initTemplateTab();
    initBatteryTab();
    initCompareTab();
    initHistoryTab();
    try {
      await loadModels();
    } catch (e) {
      console.warn('Failed to load models for conformance console:', e);
    }
  });
})();
