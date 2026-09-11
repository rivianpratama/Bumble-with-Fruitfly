// FlyScroll observation chamber.
//
// The Three.js scene (scene.js, motion.js, swipe.js) is adapted from
// mattyhempstead/fly-wirehead. FlyScroll's Python session already drives the feed
// and the simulation, so this module is a pure spectator: it takes the telemetry
// that app.js polls from /state and turns it into display. Nothing here alters the
// neural model or chooses the next reel.
//
// What moves the fly (all measured, see session.py "wirehead"):
//   electrode glow  ← interest (tonic novelty-MBON interest; the scroll decision)
//   wings / legs    ← MN9 + DNp09 motor rate, plus phasic sensory novelty so the
//                     fly visibly reacts to new content
//   head turn       ← DNa02 right-minus-left
// FlyScroll injects no reward current, so PAM11 is charted honestly and may be 0.
import { createLab } from './scene.js';
import { SWIPE_SECONDS, sampleSwipe } from './swipe.js';

const $ = (selector) => document.querySelector(selector);
const canvas = $('#scene');
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const STALE_MS = 3000;

// The phone screen: a composited portrait canvas. On a scroll the outgoing frame
// slides up under the incoming one, on the same timeline as the foreleg swipe.
const feed = document.createElement('canvas');
feed.width = 360; feed.height = 640;
const previous = document.createElement('canvas');
previous.width = 360; previous.height = 640;
const fctx = feed.getContext('2d');
let current = null;
let pendingSrc = '';
let transition = 1;
let hasPrevious = false;
let lastReelId = '';

function placeholder(title, subtitle) {
  fctx.fillStyle = '#080d10'; fctx.fillRect(0, 0, 360, 640);
  fctx.textAlign = 'center';
  fctx.fillStyle = '#c6f355'; fctx.font = '22px monospace'; fctx.fillText(title, 180, 305);
  fctx.fillStyle = '#81978b'; fctx.font = '13px monospace'; fctx.fillText(subtitle, 180, 336);
}
placeholder('FLYSCROLL', 'waiting for the first frame…');

function setFrame(dataUrl) {
  if (!dataUrl || dataUrl === pendingSrc) return;
  pendingSrc = dataUrl;
  const img = new Image();
  img.onload = () => { if (pendingSrc === dataUrl) current = img; };
  img.src = dataUrl;
}

function beginSwipe() {
  if (reduced) return;
  previous.getContext('2d').drawImage(feed, 0, 0);
  hasPrevious = true;
  transition = 0;
}

function swipeProgress() {
  return hasPrevious && !reduced ? transition : 1;
}

function drawFeed(dt) {
  if (hasPrevious && !reduced) transition = Math.min(1, transition + dt / SWIPE_SECONDS);
  const p = sampleSwipe(swipeProgress()).screen;
  fctx.fillStyle = '#000'; fctx.fillRect(0, 0, 360, 640);
  if (p < 1) fctx.drawImage(previous, 0, -p * 640);
  if (!current) return;
  fctx.save();
  fctx.translate(0, (1 - p) * 640);
  fctx.beginPath(); fctx.rect(0, 0, 360, 640); fctx.clip();
  // Cover-fit whatever aspect the frame arrives in.
  const iw = current.naturalWidth || 360, ih = current.naturalHeight || 640;
  const s = Math.max(360 / iw, 640 / ih), w = iw * s, h = ih * s;
  fctx.drawImage(current, (360 - w) / 2, (640 - h) / 2, w, h);
  fctx.restore();
}

// motion.js input. Its field names are fly-wirehead's: `pam11Hz` is the reward /
// glow channel and `motorHz` the flutter channel. FlyScroll feeds them the mapped
// measured signals described at the top of this file.
const motionState = { time: 0, paused: true, pam11Hz: 0, motorHz: 0, turnHz: 0 };
let latest = null;
let lastStateAt = 0;
let sceneFailed = false;
let lab = null;
let view = 0;

try {
  lab = createLab(canvas, feed);
} catch (error) {
  sceneFailed = true;
  console.error(error);
  $('#scene-error').hidden = false;
}

const isLive = () => Boolean(latest) && performance.now() - lastStateAt < STALE_MS;

function fmtClock(seconds) {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

// Auto-scaled multi-series plot over simulated time (same labelled detail scale
// fly-wirehead uses, so genuine small changes stay legible without noise).
const SERIES = [
  { key: 'interest', color: '#c6f355', width: 2 },
  { key: 'phasic', color: '#6ec8ff', width: 1.2 },
  { key: 'pam11_hz', color: '#ffb74a', width: 1.2 },
];

function drawChart() {
  const element = $('#hero-chart');
  if (!element) return;
  const rect = element.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.max(1, Math.round(rect.width * dpr));
  const H = Math.max(1, Math.round(rect.height * dpr));
  if (element.width !== W || element.height !== H) { element.width = W; element.height = H; }
  const c = element.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;
  c.clearRect(0, 0, w, h);

  const raw = Array.isArray(latest?.wirehead?.history) ? latest.wirehead.history : [];
  const points = raw.filter((e) => Number.isFinite(e?.sim_ms)).slice(-120);
  const range = $('#chart-range');
  if (!points.length) { range.textContent = 'WAITING FOR DATA'; return; }
  let lo = Infinity, hi = -Infinity;
  for (const p of points) for (const s of SERIES) {
    const v = Number(p[s.key]);
    if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  }
  const minimum = Math.max(0, Math.floor((lo - 5) / 5) * 5);
  const maximum = Math.max(minimum + 20, Math.ceil((hi + 5) / 5) * 5);
  range.textContent = `${minimum}–${maximum} Hz · AUTO SCALE`;

  const left = 32, right = w - 5, top = 7, bottom = h - 6;
  const y = (v) => bottom - (v - minimum) / (maximum - minimum) * (bottom - top);
  const start = points[0].sim_ms, span = Math.max(1, points.at(-1).sim_ms - start);
  const x = (p) => points.length === 1 ? right : left + (p.sim_ms - start) / span * (right - left);
  c.font = '11px monospace'; c.textAlign = 'left'; c.textBaseline = 'middle';
  for (let i = 0; i <= 2; i++) {
    const v = minimum + (maximum - minimum) * i / 2;
    c.strokeStyle = 'rgba(105,143,104,0.25)'; c.lineWidth = 0.7;
    c.beginPath(); c.moveTo(left, y(v)); c.lineTo(right, y(v)); c.stroke();
    c.fillStyle = '#a3baa3'; c.fillText(`${Number(v.toFixed(1))}`, 0, y(v));
  }
  for (const s of SERIES) {
    c.beginPath();
    let started = false;
    for (const p of points) {
      const v = Number(p[s.key]);
      if (!Number.isFinite(v)) continue;
      if (started) c.lineTo(x(p), y(v)); else { c.moveTo(x(p), y(v)); started = true; }
    }
    c.strokeStyle = s.color; c.lineWidth = s.width; c.lineJoin = 'round'; c.stroke();
  }
  // Fill under the primary series and mark its latest value.
  const primary = SERIES[0];
  const last = points.at(-1);
  c.beginPath();
  points.forEach((p, i) => i ? c.lineTo(x(p), y(Number(p[primary.key]) || 0)) : c.moveTo(x(p), y(Number(p[primary.key]) || 0)));
  c.lineTo(x(last), bottom); c.lineTo(x(points[0]), bottom); c.closePath();
  const fill = c.createLinearGradient(0, top, 0, bottom);
  fill.addColorStop(0, 'rgba(198,243,85,0.19)'); fill.addColorStop(1, 'rgba(198,243,85,0.02)');
  c.fillStyle = fill; c.fill();
  c.beginPath(); c.arc(x(last), y(Number(last[primary.key]) || 0), 3, 0, Math.PI * 2);
  c.fillStyle = '#e4ffaa'; c.fill();
}

function updateLabels() {
  const s = latest, wh = s?.wirehead, live = isLive();
  const interest = Number(s?.interest?.interest);
  $('#hero-interest').textContent = Number.isFinite(interest) ? interest.toFixed(1) : '—';
  $('#hero-interest-sub').textContent = s
    ? `${s.reel?.title || 'content'} · ${s.interest?.reason || 'watching'}`
    : 'tonic novelty · the fly scrolls when it fades';
  $('#spike-value').textContent = s ? Number(s.total_spikes || 0).toLocaleString() : '—';
  $('#sample-label').textContent = s
    ? `${Number(s.active_neurons || 0).toLocaleString()} active · ${(Number(s.neural_ms || 0) / 1000).toFixed(1)} s neural time`
    : 'Waiting for a sample';
  $('#hero-pam11').textContent = Number.isFinite(wh?.pam11_hz) ? wh.pam11_hz.toFixed(1) : '—';
  if (wh?.pam11_cells != null) $('#hero-pam11-cells').textContent = wh.pam11_cells;
  $('#session-time').textContent = fmtClock(motionState.time);
  const platform = s?.platform && s.platform !== 'local' ? s.platform.toUpperCase() : (s ? 'LOCAL CONTENT' : '');
  $('#top-state').textContent = live
    ? `BRAIN CONNECTED${platform ? ' · ' + platform : ''}`
    : (s ? 'BRAIN STALE' : 'BRAIN CONNECTING');
  $('#model-detail').textContent = s
    ? `MaleCNS v1.0 · ${s.scale} graph${s.cropped ? ' (cropped)' : ''}${s.capture_backend ? ' · ' + s.capture_backend : ''}`
    : 'Loading the MaleCNS graph…';
  document.body.classList.toggle('disconnected', !live);
}

// app.js polls /state once for the whole page and hands each sample here. Labels
// and the chart update from here too, so they stay live even when the tab is
// hidden and requestAnimationFrame is paused.
window.FlyChamber = {
  update(s) {
    if (!s || s.status !== 'running') return;
    const reelId = s.reel?.id || '';
    if (s.scrolled || (lastReelId && reelId && reelId !== lastReelId)) beginSwipe();
    lastReelId = reelId;
    latest = s;
    lastStateAt = performance.now();
    const wh = s.wirehead || {};
    const interest = Number(s.interest?.interest) || 0;
    const phasic = Number(s.interest?.phasic_novelty) || 0;
    motionState.pam11Hz = interest;                                  // glow ← interest
    motionState.motorHz = (Number(wh.motor_hz) || 0) + 0.5 * phasic; // flutter ← motor DNs + novelty
    motionState.turnHz = Number(wh.turn_hz) || 0;                    // head ← DNa02 R − L
    if (s.frame_jpeg) setFrame(s.frame_jpeg);
    drawChart();
    updateLabels();
  },
};

// Orbit and camera. The fly alone decides when to scroll, so no skip gestures.
let pointer = null;
canvas.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  pointer = { id: e.pointerId, x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', (e) => {
  if (!pointer || pointer.id !== e.pointerId) return;
  lab?.orbit(e.clientX - pointer.x, e.clientY - pointer.y);
  pointer.x = e.clientX; pointer.y = e.clientY;
});
for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
  canvas.addEventListener(type, () => { pointer = null; });
}
document.addEventListener('keydown', (e) => {
  if (e.altKey || e.ctrlKey || e.metaKey) return;
  if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || '')) return;
  if (e.code === 'KeyC' && lab) view = lab.setView(view + 1);
  if (e.code === 'KeyF') {
    const wrap = $('#chamber');
    const request = document.fullscreenElement ? document.exitFullscreen() : wrap.requestFullscreen();
    request?.catch?.(() => {});
  }
});
window.addEventListener('resize', drawChart);

let previousFrameAt = null;
function frame(now) {
  if (sceneFailed) return;
  // Bound the step so a suspended tab never produces a giant jump.
  const dt = previousFrameAt === null ? 0 : Math.max(0, Math.min((now - previousFrameAt) / 1000, 0.05));
  previousFrameAt = now;
  if (!document.hidden) {
    try {
      const live = isLive();
      motionState.paused = !live;
      if (live) motionState.time += dt;
      drawFeed(dt);
      lab.render(motionState.time, dt, motionState, swipeProgress());
    } catch (error) {
      console.error(error);
      sceneFailed = true;
      $('#scene-error').textContent = 'The 3D scene stopped. Reload to try again.';
      $('#scene-error').hidden = false;
      return;
    }
  }
  requestAnimationFrame(frame);
}

updateLabels();
drawChart();
requestAnimationFrame(frame);
