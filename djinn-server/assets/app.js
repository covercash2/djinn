/* djinn frontend — SSE streaming client
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

    if (!form || !loading || !responseSection) {
      return;
    }

    form.addEventListener('submit', function (e) {
      handleSubmit(e, loading, responseSection);
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
