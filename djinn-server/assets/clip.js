/* djinn CLIP module — image-upload form handler
 *
 * Handles the #clip-form section: drag-and-drop or file picker, thumbnail
 * preview, and multipart POST to /ui/clip.  The returned HTML fragment is
 * injected into #clip-response.
 */

(function () {
  'use strict';

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

    // Track a file dropped onto the dropzone (used as fallback when
    // fileInput.files cannot be assigned).
    var droppedFile = null;

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
        droppedFile = files[0];
        // Try to assign via DataTransfer so the native input is in sync.
        // fileInput.files is read-only in some browsers; fall back to
        // tracking the file separately and appending it during submit.
        try {
          var dt = new DataTransfer();
          dt.items.add(droppedFile);
          fileInput.files = dt.files;
          droppedFile = null; // successfully assigned; no separate tracking needed
        } catch (err) {
          // Assignment not supported; droppedFile will be appended at submit.
        }
        showPreview(files[0], filenameEl, previewEl);
      }
    });

    // Update preview when a file is selected via the native picker.
    fileInput.addEventListener('change', function () {
      droppedFile = null; // clear drop-tracking when native picker is used
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
      handleClipSubmit(clipForm, clipLoading, clipResponse, droppedFile);
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
    if (previewEl) {
      var isImage = file.type.startsWith('image/') || /\.(jpe?g|png|gif|webp|bmp|svg|tiff?)$/i.test(file.name);
      if (isImage) {
        var reader = new FileReader();
        reader.onload = function (ev) {
          previewEl.src = ev.target.result;
          previewEl.hidden = false;
        };
        reader.readAsDataURL(file);
      } else {
        // Clear any stale thumbnail when the selected file is not an image.
        previewEl.src = '';
        previewEl.hidden = true;
      }
    }
  }

  /**
   * POST the CLIP form as multipart and insert the returned HTML into responseEl.
   * @param {HTMLFormElement} form
   * @param {HTMLElement} loadingEl
   * @param {HTMLElement} responseEl
   * @param {File|null} droppedFile  File tracked from drag-and-drop (may be null)
   */
  function handleClipSubmit(form, loadingEl, responseEl, droppedFile) {
    var formData = new FormData(form);

    // If the file was dropped and couldn't be assigned to fileInput.files,
    // append it manually so it is included in the multipart upload.
    if (droppedFile) {
      formData.set('image', droppedFile);
    }

    if (loadingEl) loadingEl.style.display = 'flex';
    if (responseEl) responseEl.innerHTML = '';

    fetch('/ui/clip', {
      method: 'POST',
      body: formData,
    }).then(function (res) {
      if (!res.ok) {
        return '<p class="error">Server error (' + res.status + '). Please try again.</p>';
      }
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

  // Exported for unit testing in non-browser environments.
  if (typeof module !== 'undefined') {
    module.exports = { initClipForm: initClipForm, showPreview: showPreview, handleClipSubmit: handleClipSubmit };
  }

  if (typeof document !== 'undefined') {
    initClipForm(document);
  }
}());
