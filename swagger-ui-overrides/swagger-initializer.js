window.onload = function() {
  const MODEL_STORAGE_KEY = 'gguf-switchboard-swagger-model';
  let selectedModel = localStorage.getItem(MODEL_STORAGE_KEY) || '';
  const userEditedBodies = new WeakSet();
  const initializedBodies = new WeakSet();

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

  function markBodyEditor(textarea) {
    if (!textarea) return;
    textarea.addEventListener('input', function() {
      userEditedBodies.add(textarea);
    }, { once: false });
  }

  function initializeRequestBody(textarea, path, model) {
    if (!textarea || initializedBodies.has(textarea) || userEditedBodies.has(textarea)) {
      return;
    }

    markBodyEditor(textarea);

    try {
      var json = JSON.parse(textarea.value || '{}');
      if (!json || typeof json !== 'object') return;

      var sanitized = sanitizeRequestBody(json, path);
      if (model && 'model' in sanitized) {
        sanitized.model = model;
      }
      textarea.value = JSON.stringify(sanitized, null, 2);
      initializedBodies.add(textarea);
    } catch (e) {
      var fallback = defaultRequestBody(path, model);
      if (fallback) {
        textarea.value = JSON.stringify(fallback, null, 2);
        initializedBodies.add(textarea);
      }
    }
  }

  function updateModelFieldOnly(model) {
    if (!model) return;

    document.querySelectorAll('.opblock-body textarea, .body-param textarea').forEach(function(textarea) {
      if (userEditedBodies.has(textarea)) return;
      try {
        const json = JSON.parse(textarea.value);
        if (!json || typeof json !== 'object' || !('model' in json)) return;
        json.model = model;
        textarea.value = JSON.stringify(json, null, 2);
      } catch (e) {
        /* not JSON */
      }
    });

    document
      .querySelectorAll('input[data-param-name="model"], tr[data-param-name="model"] input')
      .forEach(function(input) {
        input.value = model;
      });

    document
      .querySelectorAll(
        'input[data-param-name="model_id"], tr[data-param-name="model_id"] input'
      )
      .forEach(function(input) {
        input.value = model;
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

  function injectModelSelector(models) {
    if (document.getElementById('global-model-select')) return;

    const wrapper = document.querySelector('.topbar-wrapper');
    if (!wrapper) return;

    const bar = document.createElement('div');
    bar.className = 'model-selector-bar';

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
      opt.textContent = m.id;
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
    });

    const registryLink = document.createElement('a');
    registryLink.id = 'registry-json-download';
    registryLink.href = '/v1/models/registry.json';
    registryLink.download = 'models.json';
    registryLink.textContent = 'models.json';
    registryLink.title = 'Download portable model registry JSON';
    registryLink.className = 'registry-json-link';

    bar.appendChild(label);
    bar.appendChild(select);
    bar.appendChild(registryLink);
    wrapper.appendChild(bar);

    if (selectedModel) {
      updateModelFieldOnly(selectedModel);
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
