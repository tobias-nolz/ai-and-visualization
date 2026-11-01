'use strict';

async function start() {
  const response = await fetch('/api/images');
  if (!response.ok) throw new Error('Could not load image manifest.');
  const data = await response.json();
  const get = id => document.getElementById(id);
  const saved = await fetch('/api/manual-review', {cache: 'no-store'});
  if (!saved.ok) throw new Error('Could not load the review file.');
  const state = await saved.json();
  const annotations = state.annotations || {};
  const cursor = state.cursor || {};
  for (const sequence of data.sequences) {
    for (const frame of sequence.frames) {
      if (frame.id in annotations && !['phantom_label', 'not_flagged'].includes(annotations[frame.id]?.status)) {
        throw new Error('The review file contains an invalid decision.');
      }
    }
  }
  const pageSize = 6;
  const pages = [];
  for (const [sequenceIndex, sequence] of data.sequences.entries()) {
    for (let start = 0; start < sequence.frames.length; start += pageSize) {
      pages.push({sequence, sequenceIndex, start, frames: sequence.frames.slice(start, start + pageSize)});
    }
  }
  let pageIndex = 0;
  let writeQueue = Promise.resolve();
  let revision = 0;

  function isAbsent(annotation) {
    return annotation?.absent ?? annotation?.status === 'phantom_label';
  }

  function save() {
    const entries = Object.values(annotations);
    const body = JSON.stringify({annotations, cursor: {imageId: pages[pageIndex].frames[0].id}});
    const progress = `${entries.length}/${data.imageCount} decisions · ${entries.filter(isAbsent).length} marked absent`;
    get('saveState').textContent = `${progress} · saving…`;
    const currentRevision = ++revision;
    writeQueue = writeQueue.then(async () => {
      let message = 'saved to local file';
      try {
        const result = await fetch('/api/manual-review', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body
        });
        if (!result.ok) throw new Error('Save failed.');
      } catch {
        message = 'SAVE FAILED. Keep this page open and check the server.';
      }
      if (currentRevision === revision) get('saveState').textContent = `${progress} · ${message}`;
    });
  }

  function toggleFrame(frame) {
    const absent = !isAbsent(annotations[frame.id]);
    annotations[frame.id] = {
      absent,
      status: absent && !['empty', 'motorcycle'].includes(frame.label) ? 'phantom_label' : 'not_flagged',
      changed_utc: new Date().toISOString()
    };
    render();
  }

  function render() {
    const page = pages[pageIndex];
    get('title').textContent = `Training C${page.sequence.camera} · sequence ${page.sequenceIndex + 1}/${data.sequences.length} · frames ${page.start + 1}–${page.start + page.frames.length}/${page.sequence.frames.length}`;
    get('title').title = page.sequence.id;
    get('previous').disabled = pageIndex === 0;
    get('next').setAttribute('aria-label', pageIndex === pages.length - 1
      ? 'Confirm final page' : 'Confirm page and continue');
    const container = get('frames');
    const columns = Math.min(page.frames.length, 3);
    container.style.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
    container.style.gridTemplateRows = `repeat(${Math.ceil(page.frames.length / columns)}, minmax(0, 1fr))`;
    container.replaceChildren();
    for (const frame of page.frames) {
      const button = document.createElement('button');
      button.className = 'frame';
      button.setAttribute('aria-pressed', String(isAbsent(annotations[frame.id])));
      button.setAttribute('aria-label', `Frame ${frame.index}, dataset label ${frame.label}: toggle animal absent`);
      const image = document.createElement('img');
      image.src = frame.url;
      image.alt = `Camera ${page.sequence.camera}, frame ${frame.index}`;
      image.draggable = false;
      const caption = document.createElement('div');
      caption.className = 'caption';
      const text = document.createElement('span');
      text.textContent = `${frame.index} · ${frame.time.slice(11, 19)} · dataset: ${frame.label}`;
      caption.append(text);
      button.append(image, caption);
      button.onclick = () => toggleFrame(frame);
      container.append(button);
    }
    save();
  }

  function nextPage() {
    for (const frame of pages[pageIndex].frames) {
      annotations[frame.id] = {
        ...annotations[frame.id],
        status: annotations[frame.id]?.status === 'phantom_label' ? 'phantom_label' : 'not_flagged',
        changed_utc: new Date().toISOString()
      };
    }
    if (pageIndex < pages.length - 1) pageIndex++;
    render();
  }

  function previousPage() {
    if (pageIndex > 0) pageIndex--;
    render();
  }

  const requested = new URLSearchParams(location.search).get('sequence');
  const firstImage = cursor.imageId;
  const requestedPage = pages.findIndex(page => requested
    ? page.sequence.id === requested : page.frames.some(frame => frame.id === firstImage));
  if (requestedPage >= 0) pageIndex = requestedPage;
  get('previous').onclick = previousPage;
  get('next').onclick = nextPage;
  document.addEventListener('keydown', event => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault();
      if (event.key === 'ArrowRight') nextPage();
      else previousPage();
    }
  });
  render();
}

start().catch(error => {
  document.getElementById('saveState').textContent = error.message;
});
