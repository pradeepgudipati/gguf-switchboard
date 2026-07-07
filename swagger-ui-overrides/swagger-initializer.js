window.onload = function() {
  const MODEL_STORAGE_KEY = 'openai-runtime-swagger-model';
  let selectedModel = localStorage.getItem(MODEL_STORAGE_KEY) || '';

  function updateVisibleModelFields(model) {
    if (!model) return;

    document.querySelectorAll('textarea').forEach(function(textarea) {
      if (!textarea.closest('.opblock-body, .body-param')) return;
      try {
        const json = JSON.parse(textarea.value);
        if (json && typeof json === 'object' && 'model' in json) {
          json.model = model;
          textarea.value = JSON.stringify(json, null, 2);
          textarea.dispatchEvent(new Event('input', { bubbles: true }));
        }
      } catch (e) {
        /* not JSON */
      }
    });

    document
      .querySelectorAll('input[data-param-name="model"], tr[data-param-name="model"] input')
      .forEach(function(input) {
        input.value = model;
        input.dispatchEvent(new Event('input', { bubbles: true }));
      });

    document
      .querySelectorAll(
        'input[data-param-name="model_id"], tr[data-param-name="model_id"] input'
      )
      .forEach(function(input) {
        input.value = model;
        input.dispatchEvent(new Event('input', { bubbles: true }));
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
      updateVisibleModelFields(selectedModel);
    });

    bar.appendChild(label);
    bar.appendChild(select);
    wrapper.appendChild(bar);

    if (selectedModel) {
      updateVisibleModelFields(selectedModel);
    }

    let debounceTimer;
    const observer = new MutationObserver(function() {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(function() {
        if (selectedModel) updateVisibleModelFields(selectedModel);
      }, 200);
    });
    const root = document.getElementById('swagger-ui');
    if (root) {
      observer.observe(root, { childList: true, subtree: true });
    }
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
      if (!selectedModel) return request;

      if (request.body) {
        try {
          const body =
            typeof request.body === 'string' ? JSON.parse(request.body) : request.body;
          if (body && typeof body === 'object' && 'model' in body) {
            body.model = selectedModel;
            request.body = JSON.stringify(body);
          }
        } catch (e) {
          /* ignore */
        }
      }

      try {
        const url = new URL(request.url, window.location.origin);
        if (url.pathname.startsWith('/v1/models/') && url.pathname !== '/v1/models') {
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
