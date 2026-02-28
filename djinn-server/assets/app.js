/* djinn frontend — SSE streaming client + CLIP image-upload form
 *
 * Listens for form submission on #prompt-form, posts the prompt to
 * /complete/stream as JSON, and streams the response tokens back via
 * Server-Sent Events (SSE).  Completed tokens are appended to
 * #stream-output so the user sees text appear token-by-token.
 *
 * SSE event protocol (see djinn-server/src/complete/mod.rs):
 *   data: <token>          — one token of generated text
 *   event: error / data: … — generation error; displayed inline
 *   event: done / data:    — stream finished cleanly
 *
 * The #clip-form section submits a multipart upload (image + prompt) to
 * /ui/clip and renders the returned HTML fragment into #clip-response.
 */

(function () {
  'use strict';

  /**
   * Initialise the streaming form handler.
   * @param {Document} doc
   */
  function init(doc) {
    const form = doc.getElementById('prompt-form');
    const loading = doc.getElementById('loading');
    const responseSection = doc.getElementById('response');

    if (form && loading && responseSection) {
      form.addEventListener('submit', function (e) {
        handleSubmit(e, loading, responseSection);
      });
    }

    initClipForm(doc);
  }

  /**
   * Initialise the CLIP image-upload form.
   * @param {Document} doc
   */
  function initClipForm(doc) {
    const clipForm = doc.getElementById('clip-form');
    const dropzone = doc.getElementById('clip-dropzone');
    const fileInput = doc.getElementById('clip-image');
    const filenameEl = doc.getElementById('clip-filename');
    const previewEl = doc.getElementById('clip-preview');
    const clipLoading = doc.getElementById('clip-loading');
    const clipResponse = doc.getElementById('clip-response');
    const clipReset = doc.getElementById('clip-reset');

    if (!clipForm || !dropzone || !fileInput) return;

    // Open file picker when the dropzone is clicked (except on the input itself).
    dropzone.addEventListener('click', function (e) {
      if (e.target !== fileInput) {
        fileInput.click();
      }
    });

    // Keyboard accessibility for the dropzone.
    dropzone.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
      }
    });

    // Drag-and-drop events.
    dropzone.addEventListener('dragover', function (e) {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });
    dropzone.addEventListener('dragleave', function () {
      dropzone.classList.remove('drag-over');
    });
    dropzone.addEventListener('drop', function (e) {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
      var files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length > 0) {
        fileInput.files = files;
        showPreview(files[0], filenameEl, previewEl);
      }
    });

    // Update preview when a file is selected via the native picker.
    fileInput.addEventListener('change', function () {
      if (fileInput.files && fileInput.files.length > 0) {
        showPreview(fileInput.files[0], filenameEl, previewEl);
      }
    });

    // Reset preview on clear.
    if (clipReset) {
      clipReset.addEventListener('click', function () {
        if (filenameEl) { filenameEl.hidden = true; filenameEl.textContent = ''; }
        if (previewEl) { previewEl.hidden = true; previewEl.src = ''; }
        if (clipResponse) {
          clipResponse.innerHTML = '<p class="placeholder">CLIP similarity result will appear here.</p>';
        }
      });
    }

    // Submit: POST multipart form to /ui/clip, render HTML fragment response.
    clipForm.addEventListener('submit', function (e) {
      e.preventDefault();
      handleClipSubmit(clipForm, clipLoading, clipResponse);
    });
  }

  /**
   * Show a thumbnail preview and filename for the selected image.
   * @param {File} file
   * @param {HTMLElement} filenameEl
   * @param {HTMLImageElement} previewEl
   */
  function showPreview(file, filenameEl, previewEl) {
    if (filenameEl) {
      filenameEl.textContent = file.name;
      filenameEl.hidden = false;
    }
    if (previewEl && (file.type.startsWith('image/') || /\.(jpe?g|png|gif|webp|bmp|svg|tiff?)$/i.test(file.name))) {
      var reader = new FileReader();
      reader.onload = function (ev) {
        previewEl.src = ev.target.result;
        previewEl.hidden = false;
      };
      reader.readAsDataURL(file);
    }
  }

  /**
   * POST the CLIP form as multipart and insert the returned HTML into clipResponse.
   * @param {HTMLFormElement} form
   * @param {HTMLElement} loadingEl
   * @param {HTMLElement} responseEl
   */
  function handleClipSubmit(form, loadingEl, responseEl) {
    var formData = new FormData(form);

    if (loadingEl) loadingEl.style.display = 'flex';
    if (responseEl) responseEl.innerHTML = '';

    fetch('/ui/clip', {
      method: 'POST',
      body: formData,
    }).then(function (res) {
      return res.text();
    }).then(function (html) {
      if (responseEl) responseEl.innerHTML = html;
    }).catch(function (err) {
      if (responseEl) {
        responseEl.innerHTML = '<p class="error">' + err.message + '</p>';
      }
    }).finally(function () {
      if (loadingEl) loadingEl.style.display = '';
    });
  }

  /**
   * Handle form submission: POST to /complete/stream and render SSE tokens.
   * @param {SubmitEvent} e
   * @param {HTMLElement} loading
   * @param {HTMLElement} responseSection
   */
  function handleSubmit(e, loading, responseSection) {
    e.preventDefault();

    const prompt = document.getElementById('prompt').value.trim();
    if (!prompt) return;

    responseSection.innerHTML =
      '<section class="response-block">' +
        '<h3 class="response-prompt">Prompt</h3>' +
        '<pre class="response-prompt-text" id="stream-prompt"></pre>' +
        '<h3 class="response-output">Response</h3>' +
        '<div class="response-content" id="stream-output" style="white-space:pre-wrap"></div>' +
      '</section>';
    document.getElementById('stream-prompt').textContent = prompt;
    const output = document.getElementById('stream-output');

    loading.style.display = 'flex';

    fetchStream('/complete/stream', prompt, output).finally(function () {
      loading.style.display = '';
    });
  }

  /**
   * Open an SSE stream for the given prompt and pipe tokens into `outputEl`.
   * @param {string} url
   * @param {string} prompt
   * @param {HTMLElement} outputEl
   * @returns {Promise<void>}
   */
  function fetchStream(url, prompt, outputEl) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: prompt }),
    }).then(function (res) {
      if (!res.ok) {
        outputEl.innerHTML =
          '<p class="error">Server error: ' + res.status + ' ' + res.statusText + '</p>';
        return;
      }
      return readSseStream(res.body, outputEl);
    }).catch(function (err) {
      outputEl.innerHTML = '<p class="error">' + err.message + '</p>';
    });
  }

  /**
   * Read a raw SSE stream from `body` and append tokens to `outputEl`.
   * @param {ReadableStream} body
   * @param {HTMLElement} outputEl
   * @returns {Promise<void>}
   */
  function readSseStream(body, outputEl) {
    var reader = body.getReader();
    var decoder = new TextDecoder();
    var buf = '';

    function pump() {
      return reader.read().then(function (chunk) {
        if (chunk.done) return;

        buf += decoder.decode(chunk.value, { stream: true });

        var boundary;
        while ((boundary = buf.indexOf('\n\n')) !== -1) {
          var eventText = buf.slice(0, boundary);
          buf = buf.slice(boundary + 2);
          processEvent(eventText, outputEl);
        }

        return pump();
      });
    }

    return pump();
  }

  /**
   * Parse a single SSE event block and append its data to `outputEl`.
   * @param {string} eventText
   * @param {HTMLElement} outputEl
   */
  function processEvent(eventText, outputEl) {
    var isError = false;
    var dataLines = [];

    eventText.split('\n').forEach(function (line) {
      if (line.startsWith('data: ')) {
        dataLines.push(line.slice(6));
      } else if (line === 'event: error') {
        isError = true;
      }
      // 'event: done' — stream ends naturally when the channel closes
    });

    if (isError) {
      outputEl.textContent += '\n[Error: ' + dataLines.join('\n') + ']';
    } else if (dataLines.length) {
      outputEl.textContent += dataLines.join('\n');
    }
  }

  // Exported for unit testing in non-browser environments.
  if (typeof module !== 'undefined') {
    module.exports = { fetchStream: fetchStream, readSseStream: readSseStream, processEvent: processEvent };
  }

  if (typeof document !== 'undefined') {
    init(document);
  }
}());
