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
    if (m.kind) parts.push(m.kind);
    var ctx = m.context_size || m.max_context_length;
    if (ctx) parts.push('ctx ' + ctx);
    if (m.min_vram_gb) parts.push('~' + m.min_vram_gb + 'GB');
    return parts.join(' · ');
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

  async function postJson(path, body) {
    var res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
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
    models.forEach(function (m) {
      var opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = formatModelLabel(m);
      select.appendChild(opt);
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

    $('inspect-run').addEventListener('click', async function () {
      var statusEl = $('inspect-status');
      var resultEl = $('inspect-result');
      var model = $('inspect-model').value;
      if (!model) {
        setStatus(statusEl, 'Select a model first.', true);
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
        var data = await postJson('/v1/conformance/inspect', body);
        setStatus(statusEl, 'Done.');
        renderInspectResult(resultEl, data);
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

    $('template-run').addEventListener('click', async function () {
      var statusEl = $('template-status');
      var resultEl = $('template-result');
      var model = $('template-model').value;
      if (!model) {
        setStatus(statusEl, 'Select a model first.', true);
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
        var data = await postJson('/v1/conformance/resolve-template', body);
        setStatus(statusEl, data.resolved ? 'Resolved.' : 'Fallback (see below).');
        renderTemplateResult(resultEl, data);
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
    $('battery-run').addEventListener('click', async function () {
      var statusEl = $('battery-status');
      var resultEl = $('battery-result');
      var model = $('battery-model').value;
      if (!model) {
        setStatus(statusEl, 'Select a model first.', true);
        return;
      }
      var btn = $('battery-run');
      btn.disabled = true;
      setStatus(statusEl, 'Running battery (may take a while)…');
      resultEl.innerHTML = '';
      try {
        var data = await postJson(
          '/v1/conformance/battery/' + encodeURIComponent(model),
          {}
        );
        setStatus(statusEl, 'Done.');
        renderBatteryResult(resultEl, data);
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

    $('compare-run').addEventListener('click', async function () {
      var statusEl = $('compare-status');
      var resultEl = $('compare-result');
      var modelA = $('compare-model-a').value;
      var modelB = $('compare-model-b').value;
      if (!modelA || !modelB) {
        setStatus(statusEl, 'Select both models first.', true);
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
        var data = await postJson('/v1/conformance/compare', payload);
        setStatus(statusEl, 'Done.');
        renderCompareResult(resultEl, data);
      } catch (e) {
        setStatus(statusEl, e.message, true);
      } finally {
        btn.disabled = false;
      }
    });
  }

  // ---------- boot ----------

  document.addEventListener('DOMContentLoaded', async function () {
    initTabs();
    initInspectTab();
    initTemplateTab();
    initBatteryTab();
    initCompareTab();
    try {
      await loadModels();
    } catch (e) {
      console.warn('Failed to load models for conformance console:', e);
    }
  });
})();
