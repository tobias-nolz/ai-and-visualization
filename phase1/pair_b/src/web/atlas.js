'use strict';

let atlas;
const get = id => document.getElementById(id);
const names = {
  'empty': 'empty',
  'urocyon cinereoargenteus': 'gray fox',
  'dasyprocta punctata': 'agouti',
  'madoqua guentheri': 'dik-dik',
  'hystrix brachyura': 'porcupine',
  'crax rubra': 'curassow',
  'macaca sp': 'macaque',
  'tapirus terrestris': 'tapir',
  'giraffa camelopardalis': 'giraffe',
  'panthera onca': 'jaguar',
  'neofelis diardi': 'clouded leopard',
  'syncerus caffer': 'buffalo',
  'argusianus argus': 'great argus',
  'meleagris ocellata': 'ocellated turkey',
  'cephalophus nigrifrons': 'duiker',
  'sus scrofa': 'wild boar',
  'phacochoerus africanus': 'warthog',
  'crocuta crocuta': 'hyena',
  'leopardus pardalis': 'ocelot',
  'equus quagga': 'plains zebra',
  'lycaon pictus': 'African wild dog',
  'bos taurus': 'cattle',
  'loxodonta africana': 'African elephant'
};

let cameras = [];
let sequences = [];
let selectedSequence = null;
let selectedFrame = 0;
let framePage = 0;

function escapeHtml(value) {
  const entities = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};
  return String(value ?? '').replace(/[&<>"']/g, character => entities[character]);
}

function speciesName(label) {
  return names[label] || label;
}

function splitName(split) {
  return {train: 'TRAIN', test: 'FOREIGN', id_test: 'ID-test'}[split];
}

function frameIssues(frame) {
  return Boolean(frame.model_phantom || frame.training_phantom);
}

function sequenceIssues(sequence) {
  return sequence.frames.filter(frameIssues).length;
}

function cameraIssues(camera) {
  return (camera.model_phantoms || 0) + (camera.training_phantoms || 0);
}

function cameraIssueRate(camera) {
  return camera.n_images ? cameraIssues(camera) / camera.n_images : 0;
}

function sameCamera(camera, sequence) {
  return camera && sequence && camera.camera === sequence.camera && camera.split === sequence.split;
}

function selectCamera(camera) {
  const sequence = sequences.find(sequence => sameCamera(camera, sequence));
  selectSequence(sequence?.id);
}

function svgElement(tag, attributes, text) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text != null) element.textContent = text;
  return element;
}

function bindSelection(node, select) {
  node.onclick = select;
  node.onkeydown = event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      select();
    }
  };
}

function drawSequenceChart() {
  const svg = get('sequenceChart');
  const selected = selectedSequence?.split === 'train' ? selectedSequence.camera : null;
  const rows = atlas.cameras.filter(camera => camera.split === 'train' && camera.sequence_ends?.pairs &&
    (camera.sequence_ends.pairs >= 10 || camera.camera === selected)
  ).sort((a, b) => {
    return b.sequence_ends.last / b.sequence_ends.pairs
      - a.sequence_ends.last / a.sequence_ends.pairs || a.camera - b.camera;
  });
  const bottom = 80 + rows.length * 22;
  svg.setAttribute('viewBox', `0 0 320 ${bottom + 38}`);
  svg.replaceChildren();
  svg.append(svgElement('text', {x: 0, y: 20, 'font-size': 16}, 'First vs. last frame flags'));
  svg.append(svgElement('text', {x: 0, y: 42, fill: '#666', 'font-size': 11}, 'First (gray) / last (blue) · flagged frames (%)'));
  svg.append(svgElement('text', {x: 0, y: 60, fill: '#666', 'font-size': 10}, 'Sorted by last-frame rate · ≥10 pairs, plus selected'));
  const x = percent => 106 + percent * 2;
  for (const tick of [0, 25, 50, 75, 100]) {
    svg.append(svgElement('path', {d: `M${x(tick)} 72 V${bottom}`, stroke: '#eee'}));
    svg.append(svgElement('text', {x: x(tick), y: bottom + 20, 'text-anchor': 'middle', 'font-size': 10}, tick));
  }
  rows.forEach((camera, index) => {
    const counts = camera.sequence_ends;
    const first = counts.first / counts.pairs * 100;
    const last = counts.last / counts.pairs * 100;
    const y = 83 + index * 22;
    const active = camera.camera === selected;
    const row = svgElement('g', {class: 'node', tabindex: 0, role: 'button', 'aria-pressed': active,
      'aria-label': `Training C${camera.camera}: first ${first.toFixed(1)}%, last ${last.toFixed(1)}%, ${counts.pairs} pairs`});
    row.append(svgElement('rect', {x: 0, y: y - 10, width: 320, height: 21, fill: active ? '#e1eef7' : 'transparent'}));
    row.append(svgElement('text', {x: 2, y: y + 4, 'font-size': 11}, `C${camera.camera} · n=${counts.pairs}`));
    row.append(svgElement('path', {d: `M${x(first)} ${y} H${x(last)}`, stroke: '#ccc', 'stroke-width': 2}));
    row.append(svgElement('circle', {cx: x(first), cy: y, r: 4, fill: '#888'}));
    row.append(svgElement('circle', {cx: x(last), cy: y, r: 4, fill: '#004488'}));
    row.append(svgElement('title', {}, row.getAttribute('aria-label')));
    bindSelection(row, () => selectCamera(camera));
    svg.append(row);
  });
}

function drawMap() {
  const svg = get('cameraMap');
  svg.replaceChildren();
  const positioned = cameras.filter(camera => Number.isFinite(camera.openness) && Number.isFinite(camera.effective_species));
  const shown = positioned.filter(camera =>
    get('showAllCameras').checked || cameraIssues(camera) || sameCamera(camera, selectedSequence)
  );
  const unscoredFindings = cameras.filter(camera => camera.openness == null && cameraIssues(camera)).length;
  get('modelTotal').textContent = cameras.filter(camera => camera.model_phantoms > 0).length;
  get('trainingTotal').textContent = cameras.filter(camera => camera.training_phantoms > 0).length;

  const xmin = 0, xmax = 1;
  const ymin = 0, ymax = Math.max(...shown.map(camera => camera.effective_species));
  const x = value => 100 + (value - xmin) / (xmax - xmin || 1) * 1220;
  const y = value => 410 - (value - ymin) / (ymax - ymin || 1) * 310;

  for (let i = 0; i <= 4; i++) {
    const xv = xmin + (xmax - xmin) * i / 4;
    const yv = ymin + (ymax - ymin) * i / 4;
    svg.append(svgElement('path', {d: `M${x(xv)} 100 V410 M100 ${y(yv)} H1320`, fill: 'none', stroke: '#e8e8e8'}));
    svg.append(svgElement('text', {x: x(xv), y: 430, 'text-anchor': 'middle', 'font-size': 10}, `${Math.round(xv * 100)}%`));
    svg.append(svgElement('text', {x: 89, y: y(yv) + 3, 'text-anchor': 'end', 'font-size': 10}, yv.toFixed(1)));
  }
  svg.append(svgElement('path', {d: 'M100 90 V410 H1330', fill: 'none', stroke: '#999'}));
  svg.append(svgElement('text', {x: 710, y: 450, 'text-anchor': 'middle', 'font-size': 12}, 'Estimated background free of trees/shrubs (%)'));
  svg.append(svgElement('text', {transform: 'translate(30 255) rotate(-90)', 'text-anchor': 'middle', 'font-size': 12}, 'Species diversity (effective species)'));

  shown.sort((a, b) => cameraIssueRate(a) - cameraIssueRate(b));
  for (const camera of shown) {
    const train = camera.split === 'train';
    const count = cameraIssues(camera);
    const rate = cameraIssueRate(camera);
    const radius = Math.max(3.5, 16 * Math.sqrt(rate));
    const extent = train ? radius * Math.sqrt(Math.PI / 2) : radius;
    const node = svgElement('g', {
      class: 'node', transform: `translate(${x(camera.openness)},${y(camera.effective_species)})`, tabindex: 0, role: 'button',
      'aria-pressed': Boolean(sameCamera(camera, selectedSequence)),
      'aria-label': `Camera ${camera.camera}, ${splitName(camera.split)}: ${count}/${camera.n_images} images flagged (${(rate * 100).toFixed(1)}%)`
    });
    node.append(svgElement('circle', {r: Math.max(5, extent), fill: 'transparent'}));
    if (sameCamera(camera, selectedSequence)) {
      node.append(svgElement('circle', {r: extent + 3, fill: 'none', stroke: '#222'}));
    }
    if (train) {
      // Match the diamond area to the area of a test-camera circle.
      node.append(svgElement('path', {d: `M0 ${-extent} L${extent} 0 L0 ${extent} L${-extent} 0 Z`, fill: count ? '#004488' : '#ccc'}));
    } else {
      let fill = '#ccc';
      if (count) fill = '#DDAA33';
      node.append(svgElement('circle', {r: radius, fill}));
    }
    bindSelection(node, () => selectCamera(camera));
    node.onpointermove = event => {
      const tip = get('tooltip');
      tip.style.display = 'block';
      tip.style.left = Math.min(event.clientX + 13, innerWidth - 290) + 'px';
      tip.style.top = Math.min(event.clientY + 13, innerHeight - 160) + 'px';
      tip.textContent = `${splitName(camera.split)} C${camera.camera}: ${count}/${camera.n_images} images flagged (${(rate * 100).toFixed(1)}%)`;
      tip.textContent += `; clear background ${(camera.openness * 100).toFixed(1)}% (middle 50%: ${(camera.openness_q25 * 100).toFixed(0)}–${(camera.openness_q75 * 100).toFixed(0)}%; ${camera.openness_frames} ${camera.openness_basis} frames / ${camera.openness_sequences} sequences)`;
      tip.textContent += `; diversity ${camera.effective_species.toFixed(1)} effective species (${camera.species_count} observed, ${camera.animal_sequences} animal-labeled sequences; all splits at this camera)`;
      if (camera.animal_sequences < 10) tip.textContent += '; small sample';
      if (train) tip.textContent += `, ${camera.decision_count} candidates with saved decisions`;
    };
    node.onpointerleave = () => get('tooltip').style.display = 'none';
    svg.append(node);
  }
  svg.append(svgElement('text', {x: 20, y: 480, fill: '#666', 'font-size': 11},
    `${shown.length} shown · ${positioned.length}/${cameras.length} have both axes · circle = test · diamond = train` +
    (unscoredFindings ? ` · ${unscoredFindings} finding camera(s) lack a reliable daytime estimate` : '')));
}

function cameraSequences() {
  return sequences.filter(sequence => sameCamera(selectedSequence, sequence))
    .sort((a, b) => a.frames[0].t.localeCompare(b.frames[0].t) || a.id.localeCompare(b.id));
}

function selectSequence(id) {
  selectedSequence = sequences.find(sequence => sequence.id === id) || null;
  selectedFrame = 0;
  framePage = 0;
  get('tooltip').style.display = 'none';
  render();
}

function trainingReview(frame) {
  if (!frame.has_decision) return 'outside review set';
  return frame.training_phantom ? 'flagged' : 'not flagged';
}

function renderSequence() {
  const menu = get('caseSelect');
  menu.replaceChildren();
  const choices = cameraSequences();
  for (const [index, sequence] of choices.entries()) {
    const option = document.createElement('option');
    option.value = sequence.id;
    option.textContent = `${index + 1} of ${choices.length} · ${sequence.frames[0].t.slice(0, 16)} · ${sequenceIssues(sequence)} issues`;
    menu.append(option);
  }
  menu.disabled = !choices.length;
  menu.value = selectedSequence?.id || '';
  const position = choices.indexOf(selectedSequence);
  get('previousCase').disabled = position <= 0;
  get('nextCase').disabled = position < 0 || position === choices.length - 1;
  get('frames').replaceChildren();
  if (!selectedSequence) {
    get('caseTitle').textContent = 'No findings in this selection';
    for (const id of ['caseExplanation', 'sequenceMeta', 'framePage']) get(id).textContent = '';
    get('previousFrames').disabled = true;
    get('nextFrames').disabled = true;
    return;
  }

  const sequence = selectedSequence;
  const train = sequence.split === 'train';
  get('caseTitle').textContent = `${splitName(sequence.split)} · C${sequence.camera} · ${sequenceIssues(sequence)} issues in this sequence`;
  get('caseExplanation').textContent = train
    ? 'A training-label issue is a frame marked empty in the review but labeled as an animal in the dataset.'
    : 'Gold shows an animal prediction where the dataset says empty. Select a camera to inspect its sequences.';
  get('predictionRowLabel').textContent = train ? 'REVIEW' : 'MODEL SAYS';
  get('predictionRowLabel').className = train ? 'train-ink' : 'model-ink';
  get('datasetRowLabel').className = train ? 'train-ink' : 'dataset-ink';
  const start = framePage * 10;
  const page = sequence.frames.slice(start, start + 10);
  get('frames').innerHTML = page.map((frame, offset) => {
    const index = start + offset;
    const prediction = train ? trainingReview(frame) : speciesName(frame.pred);
    const confidence = train || frame.c == null ? '' : Math.round(frame.c * 100) + '%';
    const image = `/images/${frame.id}`;
    const labelClass = frame.training_phantom ? 'tflag' : '';
    return `
      <button class="frame ${index === selectedFrame ? 'selected' : ''}" data-frame="${index}" aria-label="Inspect frame ${index + 1}">
        <div class="photo">
          <img src="${escapeHtml(image)}" loading="lazy" alt="Camera ${sequence.camera}, frame ${index + 1}">
          <span class="frameindex">${index + 1} / ${escapeHtml((frame.t.split(' ')[1] || frame.t).slice(0, 8))}</span>
        </div>
        <div class="track ${frame.model_phantom ? 'mflag' : ''}">
          <span>${escapeHtml(prediction)}</span><small class="conf">${confidence}</small>
        </div>
        <div class="track ${labelClass}">
          <span>${escapeHtml(speciesName(frame.label))}</span>
        </div>
      </button>`;
  }).join('');
  for (const button of document.querySelectorAll('[data-frame]')) {
    button.onclick = () => openFrame(Number(button.dataset.frame));
  }
  get('sequenceMeta').textContent = `${sequence.frames[0].t.slice(0, 10)} / ${sequence.id}`;
  get('framePage').textContent = `Frames ${start + 1}–${Math.min(start + 10, sequence.frames.length)} / ${sequence.frames.length}`;
  get('previousFrames').disabled = framePage === 0;
  get('nextFrames').disabled = start + 10 >= sequence.frames.length;
}

function openFrame(index) {
  selectedFrame = index;
  const frame = selectedSequence.frames[index];
  const train = selectedSequence.split === 'train';
  get('largeImage').src = `/images/${frame.id}`;
  get('largeImage').alt = `Camera ${selectedSequence.camera}, frame ${index + 1}`;
  get('zoomTitle').textContent = `C${selectedSequence.camera} / frame ${index + 1} of ${selectedSequence.frames.length}`;
  const result = train ? trainingReview(frame) : `Model: ${speciesName(frame.pred)}`;
  get('largeCaption').textContent = `${result} · Dataset: ${speciesName(frame.label)}`;
  renderSequence();
  get('frameOverlay').showModal();
}

function moveSequence(step) {
  const choices = cameraSequences();
  const next = choices[choices.indexOf(selectedSequence) + step];
  if (next) selectSequence(next.id);
}

function render() {
  renderSequence();
  drawMap();
  drawSequenceChart();
}

async function start() {
  const response = await fetch('/results/atlas_data.json', {cache: 'no-store'});
  if (!response.ok) throw new Error('Could not load atlas data.');
  atlas = await response.json();
  if (atlas.training) {
    atlas.cameras.push(...atlas.training.cameras);
    atlas.sequences.push(...atlas.training.sequences);
  }
  cameras = atlas.cameras;
  sequences = atlas.sequences.sort((a, b) => sequenceIssues(b) - sequenceIssues(a) || a.camera - b.camera);
  selectedSequence = sequences.find(sequence => sequenceIssues(sequence) > 0) || null;
  get('caseSelect').onchange = () => selectSequence(get('caseSelect').value);
  get('showAllCameras').onchange = drawMap;
  get('previousCase').onclick = () => moveSequence(-1);
  get('nextCase').onclick = () => moveSequence(1);
  get('previousFrames').onclick = () => { framePage--; renderSequence(); };
  get('nextFrames').onclick = () => { framePage++; renderSequence(); };
  get('closeFrame').onclick = () => get('frameOverlay').close();
  render();
}

start().catch(error => {
  get('caseTitle').textContent = error.message;
});
