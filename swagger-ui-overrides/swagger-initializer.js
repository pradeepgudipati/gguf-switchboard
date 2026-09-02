window.onload = function() {
  const MODEL_STORAGE_KEY = 'gguf-switchboard-swagger-model';
  let selectedModel = localStorage.getItem(MODEL_STORAGE_KEY) || '';
  let allModels = [];
  const userEditedBodies = new WeakSet();

  const HIDE_FOR_EMBEDDING = new Set(['chat', 'completions', 'responses', 'audio']);

  function updateEndpointVisibility(kind) {
    const isEmbedding = kind === 'embedding';
    document.querySelectorAll('[id^="operations-tag-"]').forEach(function(section) {
      var tag = section.id.replace('operations-tag-', '');
      var shouldHide = isEmbedding
        ? HIDE_FOR_EMBEDDING.has(tag)
        : tag === 'embeddings';
      section.style.display = shouldHide ? 'none' : '';
    });
  }

  function kindForModel(modelId) {
    var m = allModels.find(function(m) { return m.id === modelId; });
    return m ? m.kind : '';
  }

  function isSwaggerPlaceholder(value) {
    return value === 'string' || value === null || value === undefined;
  }

  function isPlaceholderToolCall(call) {
    if (!call || typeof call !== 'object') return true;
    return isSwaggerPlaceholder(call.id) ||
      isSwaggerPlaceholder(call.type) ||
      (call.function && (
        isSwaggerPlaceholder(call.function.name) ||
        isSwaggerPlaceholder(call.function.arguments)
      ));
  }

  function isPlaceholderToolCalls(calls) {
    return !Array.isArray(calls) || calls.length === 0 ||
      calls.every(isPlaceholderToolCall);
  }

  function isPlaceholderTool(tool) {
    return !tool || typeof tool !== 'object' ||
      isSwaggerPlaceholder(tool.type) ||
      (tool.function && isSwaggerPlaceholder(tool.function.name));
  }

  function sanitizeRequestBody(body, url) {
    if (!body || typeof body !== 'object') return body;

    try {
      var path = new URL(url, window.location.origin).pathname;
    } catch (e) {
      path = url;
    }

    if (path === '/v1/chat/completions') {
      if (Array.isArray(body.messages)) {
        body.messages = body.messages
          .filter(function(msg) { return msg && msg.role; })
          .map(function(msg) {
            var cleaned = { role: msg.role };
            var content = msg.content;
            if (content == null || content === 'string') {
              if (msg.role === 'system') {
                content = 'You are a helpful assistant.';
              } else if (msg.role === 'assistant') {
                content = 'Hello!';
              } else {
                content = 'Say hello in one sentence.';
              }
            }
            cleaned.content = content;
            if (!isPlaceholderToolCalls(msg.tool_calls)) {
              cleaned.tool_calls = msg.tool_calls;
            }
            if (msg.tool_call_id && !isSwaggerPlaceholder(msg.tool_call_id)) {
              cleaned.tool_call_id = msg.tool_call_id;
            }
            if (msg.name && !isSwaggerPlaceholder(msg.name)) {
              cleaned.name = msg.name;
            }
            return cleaned;
          });
      }
      if (!Array.isArray(body.messages) || body.messages.length === 0) {
        body.messages = [{ role: 'user', content: 'Say hello in one sentence.' }];
      }

      ['logit_bias', 'response_format', 'tool_choice', 'user'].forEach(function(key) {
        if (isSwaggerPlaceholder(body[key])) delete body[key];
      });
      if (body.tools && Array.isArray(body.tools) && body.tools.every(isPlaceholderTool)) {
        delete body.tools;
      }
      if (typeof body.max_tokens === 'number' && body.max_tokens >= 1000000000) {
        body.max_tokens = 2048;
      }
      if (typeof body.n === 'number' && body.n > 1) {
        body.n = 1;
      }
      if (body.seed === 9007199254740991) {
        delete body.seed;
      }
    }

    if (path === '/v1/conformance/inspect') {
      if (Array.isArray(body.messages)) {
        body.messages = body.messages
          .filter(function(msg) { return msg && msg.role; })
          .map(function(msg) {
            var cleaned = { role: msg.role };
            var content = msg.content;
            if (content == null || content === 'string') {
              content = msg.role === 'user'
                ? 'Call the echo tool with message set to "hello".'
                : 'Hello!';
            }
            cleaned.content = content;
            return cleaned;
          });
      }
      if (!Array.isArray(body.messages) || body.messages.length === 0) {
        body.messages = [{ role: 'user', content: 'Call the echo tool with message set to "hello".' }];
      }
      if (isSwaggerPlaceholder(body.tool_choice)) delete body.tool_choice;
      if (body.tools && Array.isArray(body.tools) && body.tools.every(isPlaceholderTool)) {
        delete body.tools;
      }
    }

    if (path === '/v1/conformance/resolve-template') {
      if (Array.isArray(body.messages)) {
        body.messages = body.messages
          .filter(function(msg) { return msg && msg.role; })
          .map(function(msg) {
            var cleaned = { role: msg.role };
            var content = msg.content;
            if (content == null || content === 'string') {
              content = 'Say hello in one sentence.';
            }
            cleaned.content = content;
            return cleaned;
          });
      }
      if (!Array.isArray(body.messages) || body.messages.length === 0) {
        body.messages = [{ role: 'user', content: 'Say hello in one sentence.' }];
      }
      if (body.tools && Array.isArray(body.tools) && body.tools.every(isPlaceholderTool)) {
        delete body.tools;
      }
    }

    if (path === '/v1/completions') {
      if (isSwaggerPlaceholder(body.prompt)) {
        body.prompt = 'Say hello in one sentence.';
      }
      ['logit_bias', 'user'].forEach(function(key) {
        if (isSwaggerPlaceholder(body[key])) delete body[key];
      });
      if (isSwaggerPlaceholder(body.suffix)) delete body.suffix;
      if (typeof body.max_tokens === 'number' && body.max_tokens >= 1000000000) {
        body.max_tokens = 2048;
      }
    }

    if (path === '/v1/embeddings') {
      if (isSwaggerPlaceholder(body.input)) {
        body.input = 'The quick brown fox jumps over the lazy dog.';
      }
      if (isSwaggerPlaceholder(body.user)) delete body.user;
    }

    if (path === '/v1/responses') {
      if (isSwaggerPlaceholder(body.input)) {
        body.input = 'What is the capital of France?';
      }
      if (isSwaggerPlaceholder(body.instructions)) {
        body.instructions = 'Answer concisely in one sentence.';
      }
      if (isSwaggerPlaceholder(body.user)) delete body.user;
      if (typeof body.max_output_tokens === 'number' && body.max_output_tokens > 32768) {
        body.max_output_tokens = 512;
      }
      if (body.stream == null) {
        body.stream = false;
      }
    }

    if (path === '/v1/audio/transcriptions') {
      if (isSwaggerPlaceholder(body.file)) {
        body.file = 'sample.wav';
      }
      if (isSwaggerPlaceholder(body.response_format)) {
        body.response_format = 'json';
      }
      if (isSwaggerPlaceholder(body.language)) {
        body.language = 'en';
      }
      if (isSwaggerPlaceholder(body.prompt)) delete body.prompt;
    }

    if (path === '/v1/audio/speech') {
      if (isSwaggerPlaceholder(body.input)) {
        body.input = 'Hello from the GGUF Switchboard speech API.';
      }
      if (isSwaggerPlaceholder(body.voice)) {
        body.voice = 'alloy';
      }
      if (isSwaggerPlaceholder(body.response_format)) {
        body.response_format = 'mp3';
      }
    }

    return body;
  }

  function defaultRequestBody(path, model) {
    var resolvedModel = model || 'gemma-4-e4b';
    if (path === '/v1/chat/completions') {
      return {
        model: resolvedModel,
        messages: [{ role: 'user', content: 'Is Rust faster than Python for backend services? Explain briefly.' }],
        max_tokens: 2048,
        stream: false
      };
    }
    if (path === '/v1/completions') {
      return {
        model: resolvedModel,
        prompt: 'Say hello in one sentence.',
        max_tokens: 512
      };
    }
    if (path === '/v1/conformance/inspect') {
      return {
        model: resolvedModel,
        messages: [{ role: 'user', content: 'Call the echo tool with message set to "hello".' }],
        tools: [{
          type: 'function',
          function: {
            name: 'echo',
            parameters: {
              type: 'object',
              properties: { message: { type: 'string' } },
              required: ['message']
            }
          }
        }],
        tool_choice: 'required'
      };
    }
    if (path === '/v1/conformance/resolve-template') {
      return {
        model: resolvedModel,
        messages: [{ role: 'user', content: 'Say hello in one sentence.' }],
        tools: []
      };
    }
    if (path === '/v1/embeddings') {
      return {
        model: resolvedModel,
        input: 'The quick brown fox jumps over the lazy dog.'
      };
    }
    if (path === '/v1/responses') {
      return {
        model: resolvedModel,
        input: 'What is the capital of France?',
        instructions: 'Answer concisely in one sentence.',
        max_output_tokens: 512,
        stream: false
      };
    }
    if (path === '/v1/audio/transcriptions') {
      return {
        model: resolvedModel,
        file: 'sample.wav',
        response_format: 'json',
        language: 'en'
      };
    }
    if (path === '/v1/audio/speech') {
      return {
        model: resolvedModel,
        input: 'Hello from the GGUF Switchboard speech API.',
        voice: 'alloy',
        response_format: 'mp3'
      };
    }
    return null;
  }

  // Swagger UI's body/parameter editors are React-controlled inputs. Setting
  // `.value` directly (as a plain DOM mutation) paints the screen but never
  // reaches React's internal state or Swagger's Redux store — so the visible
  // editor can revert on the next re-render, and `requestInterceptor` becomes
  // the only thing keeping the outgoing request correct. Use the native
  // value-setter + a real `input` event so React's onChange fires and its
  // state actually updates.
  function setReactControlledValue(el, value) {
    if (!el) return;
    var proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    var setter = Object.getOwnPropertyDescriptor(proto, 'value') &&
      Object.getOwnPropertyDescriptor(proto, 'value').set;
    if (setter) {
      setter.call(el, value);
    } else {
      el.value = value;
    }
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }

  function markBodyEditor(textarea) {
    if (!textarea) return;
    textarea.addEventListener('input', function() {
      userEditedBodies.add(textarea);
    }, { once: false });
  }

  function initializeRequestBody(textarea, path, model) {
    if (!textarea) return;

    markBodyEditor(textarea);

    try {
      var json = JSON.parse(textarea.value || '{}');
      if (!json || typeof json !== 'object') return;

      var sanitized = sanitizeRequestBody(json, path);
      if (model && 'model' in sanitized) {
        sanitized.model = model;
      }
      if (!userEditedBodies.has(textarea)) {
        setReactControlledValue(textarea, JSON.stringify(sanitized, null, 2));
      } else if (model && 'model' in json) {
        json.model = model;
        setReactControlledValue(textarea, JSON.stringify(json, null, 2));
      }
    } catch (e) {
      if (userEditedBodies.has(textarea)) return;
      var fallback = defaultRequestBody(path, model);
      if (fallback) {
        setReactControlledValue(textarea, JSON.stringify(fallback, null, 2));
      }
    }
  }

  function updateModelFieldOnly(model) {
    if (!model) return;

    document.querySelectorAll('.opblock-body textarea, .body-param textarea').forEach(function(textarea) {
      try {
        const json = JSON.parse(textarea.value);
        if (!json || typeof json !== 'object' || !('model' in json)) return;
        json.model = model;
        setReactControlledValue(textarea, JSON.stringify(json, null, 2));
      } catch (e) {
        /* not JSON */
      }
    });

    document
      .querySelectorAll('input[data-param-name="model"], tr[data-param-name="model"] input')
      .forEach(function(input) {
        setReactControlledValue(input, model);
      });

    document
      .querySelectorAll(
        'input[data-param-name="model_id"], tr[data-param-name="model_id"] input'
      )
      .forEach(function(input) {
        setReactControlledValue(input, model);
      });
  }

  function pathFromOpblock(opblock) {
    if (!opblock) return '';
    var pathNode = opblock.querySelector('.opblock-summary-path');
    if (!pathNode) return '';
    return (pathNode.getAttribute('data-path') || pathNode.textContent || '').trim();
  }

  function initializeVisibleBodies(model) {
    document.querySelectorAll('.opblock.is-open textarea').forEach(function(textarea) {
      if (!textarea.closest('.opblock-body, .body-param')) return;
      var path = pathFromOpblock(textarea.closest('.opblock'));
      initializeRequestBody(textarea, path, model);
    });
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

  function formatModelCard(m) {
    if (!m || !m.id) return '';
    var lines = [];
    if (m.display_name) lines.push(m.display_name);
    lines.push(formatModelLabel(m));
    if (m.hf_repo) lines.push(m.hf_repo);
    if (m.description) lines.push(m.description);
    if (m.capabilities && m.capabilities.length) {
      lines.push('capabilities: ' + m.capabilities.join(', '));
    }
    return lines.join('\n');
  }

  const STATUS_POLL_MS = 3000;
  let statusPollHandle = null;

  function statusBadgeState(status) {
    if (!status || !status.loaded_model) {
      return { dotClass: 'status-dot-gray', text: 'No model loaded' };
    }
    var n = status.active_requests || 0;
    if (n > 0) {
      return {
        dotClass: 'status-dot-amber',
        text: status.loaded_model + ' \u00b7 processing ' + n + ' request' + (n === 1 ? '' : 's')
      };
    }
    return {
      dotClass: 'status-dot-green',
      text: status.loaded_model + ' \u00b7 serving (idle)'
    };
  }

  function renderStatusBadge(status) {
    const badge = document.getElementById('model-status-badge');
    if (!badge) return;
    const dot = badge.querySelector('.status-dot');
    const label = badge.querySelector('.status-label');
    const state = statusBadgeState(status);
    dot.className = 'status-dot ' + state.dotClass;
    label.textContent = state.text;
    badge.title = status
      ? 'Loaded model: ' + (status.loaded_model || '(none)') +
        '\nActive requests: ' + (status.active_requests || 0) +
        '\nUptime: ' + (status.uptime_secs || 0) + 's'
      : 'Status unavailable';
  }

  function pollStatus() {
    fetch('/status')
      .then(function(r) { return r.json(); })
      .then(function(data) { renderStatusBadge(data); })
      .catch(function(err) {
        console.warn('Failed to poll /status for Swagger UI badge:', err);
        const badge = document.getElementById('model-status-badge');
        if (!badge) return;
        badge.querySelector('.status-dot').className = 'status-dot status-dot-gray';
        badge.querySelector('.status-label').textContent = 'Status unavailable';
      });
  }

  // Currently-loaded / serving / processing indicator. Lives inside the
  // model-selector bar so it travels with it; the bar itself gets torn
  // down and rebuilt on "Refresh models", so this re-creates the badge
  // each time but only ever starts one polling interval.
  function injectStatusBadge(bar) {
    const badge = document.createElement('span');
    badge.id = 'model-status-badge';
    badge.className = 'model-status-badge';
    badge.title = 'Checking status\u2026';

    const dot = document.createElement('span');
    dot.className = 'status-dot status-dot-gray';

    const label = document.createElement('span');
    label.className = 'status-label';
    label.textContent = 'Checking\u2026';

    badge.appendChild(dot);
    badge.appendChild(label);
    bar.appendChild(badge);

    if (statusPollHandle) {
      pollStatus();
    } else {
      pollStatus();
      statusPollHandle = setInterval(pollStatus, STATUS_POLL_MS);
    }
  }

  function injectModelSelector(models) {
    if (document.getElementById('global-model-select')) return;
    allModels = models;

    const wrapper = document.querySelector('.topbar-wrapper');
    if (!wrapper) return;

    const bar = document.createElement('div');
    bar.className = 'model-selector-bar';

    injectStatusBadge(bar);

    const label = document.createElement('label');
    label.setAttribute('for', 'global-model-select');
    label.textContent = 'Model';

    const select = document.createElement('select');
    select.id = 'global-model-select';
    select.title = 'Selected model is applied to all API requests (like the auth token)';

    const empty = document.createElement('option');
    empty.value = '';
    empty.textContent = '(select a model)';
    select.appendChild(empty);

    models.forEach(function(m) {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = formatModelLabel(m);
      opt.title = formatModelCard(m);
      select.appendChild(opt);
    });

    if (selectedModel && models.some(function(m) { return m.id === selectedModel; })) {
      select.value = selectedModel;
    } else if (models.length > 0) {
      selectedModel = models[0].id;
      select.value = selectedModel;
      localStorage.setItem(MODEL_STORAGE_KEY, selectedModel);
    }

    select.addEventListener('change', function(e) {
      selectedModel = e.target.value;
      if (selectedModel) {
        localStorage.setItem(MODEL_STORAGE_KEY, selectedModel);
      } else {
        localStorage.removeItem(MODEL_STORAGE_KEY);
      }
      updateModelFieldOnly(selectedModel);
      updateEndpointVisibility(kindForModel(selectedModel));
    });

    const refreshBtn = document.createElement('button');
    refreshBtn.type = 'button';
    refreshBtn.id = 'refresh-models-btn';
    refreshBtn.textContent = 'Refresh models';
    refreshBtn.title = 'Rescan model directories and update the registry';
    refreshBtn.className = 'refresh-models-btn';
    refreshBtn.addEventListener('click', function() {
      refreshBtn.disabled = true;
      const prev = refreshBtn.textContent;
      refreshBtn.textContent = 'Refreshing…';
      fetch('/v1/models/refresh', { method: 'POST' })
        .then(function(r) {
          if (!r.ok) {
            return r.text().then(function(t) {
              throw new Error(t || ('HTTP ' + r.status));
            });
          }
          return r.json();
        })
        .then(function(data) {
          refreshBtn.textContent = 'Updated (' + (data.total || 0) + ')';
          return fetch('/v1/models')
            .then(function(r) { return r.json(); })
            .then(function(list) {
              const existing = document.getElementById('global-model-select');
              const barEl = existing && existing.closest('.model-selector-bar');
              if (barEl) barEl.remove();
              injectModelSelector(list.data || []);
            });
        })
        .catch(function(err) {
          console.warn('Model refresh failed:', err);
          refreshBtn.textContent = 'Refresh failed';
        })
        .finally(function() {
          setTimeout(function() {
            refreshBtn.textContent = prev;
            refreshBtn.disabled = false;
          }, 2000);
        });
    });

    const conformanceLink = document.createElement('a');
    conformanceLink.id = 'conformance-console-link';
    conformanceLink.href = './conformance.html';
    conformanceLink.target = '_blank';
    conformanceLink.rel = 'noopener';
    conformanceLink.className = 'conformance-console-link';
    conformanceLink.textContent = 'Conformance Console →';
    conformanceLink.title =
      'Open the tool-calling / chat-template conformance console in a new tab';

    bar.appendChild(label);
    bar.appendChild(select);
    bar.appendChild(refreshBtn);
    bar.appendChild(conformanceLink);
    wrapper.appendChild(bar);

    if (selectedModel) {
      updateModelFieldOnly(selectedModel);
      updateEndpointVisibility(kindForModel(selectedModel));
    }

    document.getElementById('swagger-ui').addEventListener('click', function(event) {
      var tryIt = event.target.closest('.btn.try-out__btn, .try-out__btn');
      if (!tryIt) return;
      setTimeout(function() {
        initializeVisibleBodies(selectedModel);
      }, 0);
    }, true);
  }

  function fetchModelsAndInject() {
    fetch('/v1/models')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        injectModelSelector(data.data || []);
        // ponytail: safety net for async Swagger UI rendering after onComplete
        setTimeout(function() {
          if (selectedModel) updateEndpointVisibility(kindForModel(selectedModel));
        }, 500);
      })
      .catch(function(err) {
        console.warn('Failed to load models for Swagger UI selector:', err);
      });
  }

  window.ui = SwaggerUIBundle({
    {{config}},
    requestInterceptor: function(request) {
      if (request.body) {
        try {
          const body =
            typeof request.body === 'string' ? JSON.parse(request.body) : request.body;
          if (body && typeof body === 'object') {
            if ('model' in body && selectedModel) {
              body.model = selectedModel;
            }
            sanitizeRequestBody(body, request.url);
            request.body = JSON.stringify(body);
          }
        } catch (e) {
          /* ignore */
        }
      } else if (selectedModel && request.url) {
        try {
          var url = new URL(request.url, window.location.origin);
          var defaultBody = defaultRequestBody(url.pathname, selectedModel);
          if (defaultBody) {
            request.body = JSON.stringify(defaultBody);
            request.headers = request.headers || {};
            request.headers['Content-Type'] = 'application/json';
          }
        } catch (e) {
          /* ignore */
        }
      }

      if (!selectedModel) return request;

      try {
        const url = new URL(request.url, window.location.origin);
        if (url.pathname.startsWith('/v1/models/') && url.pathname !== '/v1/models' && url.pathname !== '/v1/models/registry.json') {
          url.pathname = '/v1/models/' + encodeURIComponent(selectedModel);
          request.url = url.pathname + url.search;
        }
        if (url.pathname.startsWith('/v1/usage') && url.searchParams.has('model')) {
          url.searchParams.set('model', selectedModel);
          request.url = url.pathname + '?' + url.searchParams.toString();
        }
      } catch (e) {
        /* ignore */
      }

      return request;
    },
    onComplete: fetchModelsAndInject,
    presets: [
      SwaggerUIBundle.presets.apis,
      SwaggerUIStandalonePreset
    ],
    plugins: [
      SwaggerUIBundle.plugins.DownloadUrl
    ]
  });
};
