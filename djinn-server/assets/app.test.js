import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createRequire } from 'module';

// Load app.js via CommonJS require so the module.exports guard is hit.
const require = createRequire(import.meta.url);
const { processEvent, readSseStream, fetchStream } = require('./app.js');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create a minimal mock of an HTMLElement with textContent and innerHTML. */
function makeOutputEl() {
  let text = '';
  return {
    get textContent() { return text; },
    set textContent(v) { text = v; },
    innerHTML: '',
  };
}

/** Build a ReadableStream from an array of Uint8Array chunks. */
function makeReadableStream(chunks) {
  let idx = 0;
  return new ReadableStream({
    pull(controller) {
      if (idx < chunks.length) {
        controller.enqueue(chunks[idx++]);
      } else {
        controller.close();
      }
    },
  });
}

/** Encode a string to Uint8Array. */
function enc(str) {
  return new TextEncoder().encode(str);
}

// ---------------------------------------------------------------------------
// processEvent
// ---------------------------------------------------------------------------

describe('processEvent', () => {
  it('appends a single data token to outputEl', () => {
    const el = makeOutputEl();
    processEvent('data: hello', el);
    expect(el.textContent).toBe('hello');
  });

  it('appends multiple data lines joined by newline', () => {
    const el = makeOutputEl();
    processEvent('data: foo\ndata: bar', el);
    expect(el.textContent).toBe('foo\nbar');
  });

  it('accumulates across multiple calls', () => {
    const el = makeOutputEl();
    processEvent('data: first ', el);
    processEvent('data: second', el);
    expect(el.textContent).toBe('first second');
  });

  it('renders an error event with bracketed prefix', () => {
    const el = makeOutputEl();
    processEvent('event: error\ndata: something broke', el);
    expect(el.textContent).toContain('[Error: something broke]');
  });

  it('done event is a no-op', () => {
    const el = makeOutputEl();
    processEvent('event: done\ndata: ', el);
    expect(el.textContent).toBe('');
  });

  it('ignores lines without a recognised prefix', () => {
    const el = makeOutputEl();
    processEvent('id: 42\nretry: 1000', el);
    expect(el.textContent).toBe('');
  });

  it('handles empty event text without throwing', () => {
    const el = makeOutputEl();
    expect(() => processEvent('', el)).not.toThrow();
    expect(el.textContent).toBe('');
  });
});

// ---------------------------------------------------------------------------
// readSseStream
// ---------------------------------------------------------------------------

describe('readSseStream', () => {
  it('streams a single token from one chunk', async () => {
    const el = makeOutputEl();
    const body = makeReadableStream([enc('data: token1\n\n')]);
    await readSseStream(body, el);
    expect(el.textContent).toBe('token1');
  });

  it('reassembles events split across multiple chunks', async () => {
    const el = makeOutputEl();
    // The event boundary (\n\n) straddles two chunks.
    const body = makeReadableStream([
      enc('data: tok'),
      enc('en1\n\n'),
    ]);
    await readSseStream(body, el);
    expect(el.textContent).toBe('token1');
  });

  it('handles multiple events in a single chunk', async () => {
    const el = makeOutputEl();
    const body = makeReadableStream([enc('data: a\n\ndata: b\n\n')]);
    await readSseStream(body, el);
    expect(el.textContent).toBe('ab');
  });

  it('handles a done event at the end of the stream', async () => {
    const el = makeOutputEl();
    const body = makeReadableStream([enc('data: hello\n\nevent: done\ndata: \n\n')]);
    await readSseStream(body, el);
    expect(el.textContent).toBe('hello');
  });

  it('renders an error event from the stream', async () => {
    const el = makeOutputEl();
    const body = makeReadableStream([enc('event: error\ndata: oops\n\n')]);
    await readSseStream(body, el);
    expect(el.textContent).toContain('[Error: oops]');
  });
});

// ---------------------------------------------------------------------------
// fetchStream
// ---------------------------------------------------------------------------

describe('fetchStream', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('pipes response body into outputEl on 200', async () => {
    const el = makeOutputEl();
    const body = makeReadableStream([enc('data: streamed\n\n')]);

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      body,
    }));

    await fetchStream('/complete/stream', 'hello', el);
    expect(el.textContent).toBe('streamed');
  });

  it('posts JSON with the prompt to the given URL', async () => {
    const el = makeOutputEl();
    const body = makeReadableStream([]);

    const mockFetch = vi.fn().mockResolvedValue({ ok: true, body });
    vi.stubGlobal('fetch', mockFetch);

    await fetchStream('/complete/stream', 'my prompt', el);

    expect(mockFetch).toHaveBeenCalledWith('/complete/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: 'my prompt' }),
    });
  });

  it('sets outputEl.innerHTML to an error paragraph on non-OK response', async () => {
    const el = makeOutputEl();

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
    }));

    await fetchStream('/complete/stream', 'hello', el);
    expect(el.innerHTML).toContain('Server error: 500');
  });

  it('sets outputEl.innerHTML to an error paragraph when fetch throws', async () => {
    const el = makeOutputEl();

    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('network down')));

    await fetchStream('/complete/stream', 'hello', el);
    expect(el.innerHTML).toContain('network down');
  });
});
