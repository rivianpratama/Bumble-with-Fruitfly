const cloud = window.FlyCloud(document.getElementById("cloud"));
// Rotation-speed slider for the wiring plate. Value is turns-per-minute; the
// cloud spins in radians/second. Remembered per browser (best-effort).
(function wireSpin() {
  const slider = document.getElementById("spin-speed");
  const readout = document.getElementById("spin-readout");
  if (!slider || !cloud.setSpin) return;
  try {
    const saved = localStorage.getItem("flyscroll.spinRpm");
    if (saved !== null) slider.value = saved;
  } catch (_) {}
  const apply = () => {
    const rpm = Number(slider.value) || 0;
    cloud.setSpin((rpm / 60) * 2 * Math.PI);
    if (readout) readout.textContent = `${rpm.toFixed(1)}/min`;
    try { localStorage.setItem("flyscroll.spinRpm", String(rpm)); } catch (_) {}
  };
  slider.addEventListener("input", apply);
  apply();
})();
let anatomyReady = false;
let shortsMode = false;
let lastReelKey = "";
let knownReelTicks = new Set();
let reelLogInitialized = false;
let eyeUV = null;
// Bumble swipe verdict: track like/pass counts so a new swipe triggers a zoom.
let prevLikes = null, prevPasses = null;
function showSwipeVerdict(kind) {
  const el = document.getElementById("swipe-verdict");
  if (!el) return;
  el.querySelector(".mark").textContent = kind === "like" ? "✓" : "✕";
  el.classList.remove("show", "like", "pass");
  void el.offsetWidth;  // reflow so the CSS animation restarts on every swipe
  el.classList.add("show", kind);
}

function decodeArray(b64, Type) {
  const raw = atob(b64 || "");
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Type(bytes.buffer);
}

function setEyeAnatomy(a) {
  if (!a.eye_uv_b64 || !a.eye_columns) return;
  eyeUV = decodeArray(a.eye_uv_b64, Float32Array);
}

function drawFlyEye(payload) {
  if (!eyeUV || !payload?.rgb_b64 || !payload?.activity_b64) return;
  const canvas = document.getElementById("fly-eye");
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || 560;
  const h = canvas.clientHeight || 210;
  canvas.width = Math.max(1, Math.floor(w * dpr));
  canvas.height = Math.max(1, Math.floor(h * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const rgb = decodeArray(payload.rgb_b64, Uint8Array);
  const activity = decodeArray(payload.activity_b64, Uint8Array);
  const n = Math.min(payload.columns || 0, eyeUV.length / 2, rgb.length / 3, activity.length);
  if (!n) return;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    minX = Math.min(minX, eyeUV[i * 2]);
    maxX = Math.max(maxX, eyeUV[i * 2]);
    minY = Math.min(minY, eyeUV[i * 2 + 1]);
    maxY = Math.max(maxY, eyeUV[i * 2 + 1]);
  }
  const pad = 13;
  const scale = Math.min(
    (w - pad * 2) / Math.max(1e-6, maxX - minX),
    (h - pad * 2) / Math.max(1e-6, maxY - minY),
  );
  const ox = (w - (maxX - minX) * scale) * 0.5;
  const oy = (h - (maxY - minY) * scale) * 0.5;
  const radius = Math.max(1.35, Math.min(4.0, scale * 0.012));

  ctx.globalCompositeOperation = "lighter";
  for (let i = 0; i < n; i++) {
    const a = activity[i] / 255;
    if (a < 0.10) continue;
    const x = ox + (eyeUV[i * 2] - minX) * scale;
    const y = oy + (eyeUV[i * 2 + 1] - minY) * scale;
    ctx.fillStyle = `rgba(76,205,255,${0.06 + a * 0.34})`;
    ctx.beginPath();
    ctx.arc(x, y, radius * (1.5 + a * 2.2), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalCompositeOperation = "source-over";
  for (let i = 0; i < n; i++) {
    const x = ox + (eyeUV[i * 2] - minX) * scale;
    const y = oy + (eyeUV[i * 2 + 1] - minY) * scale;
    const a = activity[i] / 255;
    const gain = 0.48 + 0.75 * a;
    const r = Math.min(255, Math.round(rgb[i * 3] * gain));
    const g = Math.min(255, Math.round(rgb[i * 3 + 1] * gain));
    const b = Math.min(255, Math.round(rgb[i * 3 + 2] * gain));
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  const keys = [
    ["on", "ON"], ["off", "OFF"], ["flicker", "flicker"],
    ["motion_up", "↑ motion"], ["motion_down", "↓ motion"],
    ["expansion", "loom"], ["small_object", "small object"],
    ["chromatic", "chromatic"],
  ];
  const features = payload.features || {};
  document.getElementById("eye-features").innerHTML = keys.map(([key, label]) => {
    const value = Number(features[key] || 0);
    const width = Math.min(100, (1 - Math.exp(-value * 5)) * 100);
    return `<span class="eye-feature">${label} ${value.toFixed(3)}` +
      `<span class="eye-feature-track"><i style="width:${width.toFixed(1)}%"></i></span></span>`;
  }).join("");
}

function cssVar(name, fallback) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function drawSeries(canvas, series, opts = {}) {
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || canvas.width;
  const cssH = opts.height || canvas.clientHeight || canvas.height;
  canvas.width = Math.max(1, Math.floor(cssW * dpr));
  canvas.height = Math.max(1, Math.floor(cssH * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = cssW;
  const h = cssH;
  ctx.clearRect(0, 0, w, h);

  const pad = opts.pad || { l: 4, r: 4, t: 6, b: 6 };
  const plotW = Math.max(1, w - pad.l - pad.r);
  const plotH = Math.max(1, h - pad.t - pad.b);
  const color = opts.color || cssVar("--accent", "#c6f355");
  const skipColor = opts.skipColor || cssVar("--warn", "#ff6b4a");
  const marks = opts.marks || [];
  const plotted = [
    { data: series, color },
    ...(opts.overlays || []),
  ].filter((item) => item.data && item.data.length >= 2);

  ctx.strokeStyle = "rgba(232,242,234,0.08)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t + plotH);
  ctx.lineTo(pad.l + plotW, pad.t + plotH);
  ctx.stroke();

  if (!plotted.length) return;

  let minT = Infinity;
  let maxT = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (const item of plotted) {
    for (const p of item.data) {
      if (p.t < minT) minT = p.t;
      if (p.t > maxT) maxT = p.t;
      if (p.v < minV) minV = p.v;
      if (p.v > maxV) maxV = p.v;
    }
  }
  if (opts.ymin != null) minV = Math.min(minV, opts.ymin);
  if (opts.ymax != null) maxV = Math.max(maxV, opts.ymax);
  if (!(maxT > minT)) maxT = minT + 1;
  if (!(maxV > minV)) {
    minV -= 1;
    maxV += 1;
  }

  const xAt = (t) => pad.l + ((t - minT) / (maxT - minT)) * plotW;
  const yAt = (v) => pad.t + plotH - ((v - minV) / (maxV - minV)) * plotH;

  for (const item of plotted) {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = item.lineWidth || 1.5;
    ctx.beginPath();
    item.data.forEach((p, i) => {
      const x = xAt(p.t);
      const y = yAt(p.v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  const primary = plotted[0].data;
  ctx.beginPath();
  primary.forEach((p, i) => {
    const x = xAt(p.t);
    const y = yAt(p.v);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(xAt(primary[primary.length - 1].t), pad.t + plotH);
  ctx.lineTo(xAt(primary[0].t), pad.t + plotH);
  ctx.closePath();
  ctx.globalAlpha = 0.12;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.globalAlpha = 1;

  for (const m of marks) {
    if (m.t < minT || m.t > maxT) continue;
    const x = xAt(m.t);
    ctx.strokeStyle = skipColor;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = skipColor;
    ctx.beginPath();
    ctx.arc(x, pad.t + 4, 2.4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawDualSpark(canvas, interest, novelty, skipT) {
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || 180;
  const cssH = 42;
  canvas.width = Math.max(1, Math.floor(cssW * dpr));
  canvas.height = Math.max(1, Math.floor(cssH * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const seriesList = [
    { data: interest, color: cssVar("--accent", "#c6f355") },
    { data: novelty, color: "#6ec8ff" },
  ].filter((s) => s.data && s.data.length > 1);

  if (!seriesList.length) return;

  let minT = Infinity;
  let maxT = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (const s of seriesList) {
    for (const p of s.data) {
      if (p.t < minT) minT = p.t;
      if (p.t > maxT) maxT = p.t;
      if (p.v < minV) minV = p.v;
      if (p.v > maxV) maxV = p.v;
    }
  }
  if (!(maxT > minT)) maxT = minT + 1;
  if (!(maxV > minV)) {
    minV -= 1;
    maxV += 1;
  }
  const xAt = (t) => ((t - minT) / (maxT - minT)) * cssW;
  const yAt = (v) => cssH - 2 - ((v - minV) / (maxV - minV)) * (cssH - 4);

  for (const s of seriesList) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    s.data.forEach((p, i) => {
      const x = xAt(p.t);
      const y = yAt(p.v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  if (skipT != null && skipT >= minT) {
    const x = xAt(skipT);
    ctx.strokeStyle = cssVar("--warn", "#ff6b4a");
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, cssH);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function enableShortsLayout() {
  if (shortsMode) return;
  shortsMode = true;
  document.getElementById("layout").classList.add("shorts-mode");
  document.getElementById("charts").hidden = false;
  document.getElementById("left-analytics").hidden = false;
  document.getElementById("watch-dist-stat").hidden = false;
}

function renderReelCards(reels) {
  const host = document.getElementById("reel-cards");
  if (!reels) return;
  const reelKey = reels.map((r) => r.tick).join(",");
  if (reelKey === lastReelKey) return;
  lastReelKey = reelKey;
  host.innerHTML = "";
  for (const r of reels.slice(0, 24)) {
    const card = document.createElement("article");
    card.className = "reel-card";
    if (reelLogInitialized && !knownReelTicks.has(r.tick)) {
      card.classList.add("reel-card-enter");
    }
    const hero = document.createElement("div");
    hero.className = "reel-hero";
    const watched = (r.watch_seconds ?? 0).toFixed(2);
    const duration = r.duration_seconds
      ? `${watched} / ${Number(r.duration_seconds).toFixed(2)}s`
      : `${watched}s`;
    // Bumble cards show the profile's average interest (Hz); others show watch %.
    const hz = r.mean_interest_hz;
    const big = hz != null ? `${Number(hz).toFixed(1)}` : `${Math.round(r.watch_pct ?? 0)}%`;
    const sub = hz != null
      ? `Hz avg · ${watched}s · ${r.reason || "skip"}`
      : `${duration} · ${r.reason || "skip"}`;
    hero.innerHTML = `${big}<small>${sub}</small>`;
    const body = document.createElement("div");
    const meta = document.createElement("div");
    meta.className = "reel-card-meta";
    const label = r.label ? `<span class="bumble-label">${r.label}</span>` : "";
    meta.innerHTML = `<strong>${r.title || r.reel_id || "content"}</strong>${label}`;
    const c = document.createElement("canvas");
    body.appendChild(meta);
    body.appendChild(c);
    card.appendChild(hero);
    card.appendChild(body);
    host.appendChild(card);
    requestAnimationFrame(() => drawDualSpark(c, r.interest, r.novelty, r.skip_t));
  }
  knownReelTicks = new Set(reels.map((r) => r.tick));
  reelLogInitialized = true;
}

function renderBumble(b) {
  const panel = document.getElementById("bumble-panel");
  panel.hidden = !b;
  if (!b) return;
  document.getElementById("bumble-decision").textContent = b.last_decision || "deciding…";
  // The swipe reads the recent spike window; show that vs the threshold, with
  // the whole-profile mean in parentheses for context.
  const signal = Number((b.window_interest ?? b.mean_interest) || 0).toFixed(1);
  const mean = Number(b.mean_interest || 0).toFixed(1);
  const threshold = Number(b.like_threshold || 0).toFixed(1);
  document.getElementById("bumble-mean").textContent =
    `${signal} / ${threshold} Hz · ${b.threshold_mode || ""} (mean ${mean})`;
  const cap = b.max_swipes ? ` / ${b.max_swipes}` : "";
  document.getElementById("bumble-counts").textContent =
    `${b.likes ?? 0} · ${b.passes ?? 0} · ${b.swipes ?? 0}${cap}`;
  document.getElementById("bumble-photo").textContent =
    `${(b.profile_scrolls ?? 0) + 1}/${(b.max_profile_scrolls ?? 0) + 1} · ${Number(b.dwell_seconds || 0).toFixed(1)}s`;
  const device = document.getElementById("bumble-device");
  device.textContent = b.device_lost ? "phone unreachable" : "device ok";
  device.style.color = b.device_lost ? "var(--warn)" : "";
  document.getElementById("bumble-dry").hidden = !b.dry_run;
}

async function loadAnatomy() {
  const res = await fetch("/anatomy", { cache: "no-store" });
  if (!res.ok) return;
  const a = await res.json();
  setEyeAnatomy(a);
  if (!a.points || !a.xyz_b64) {
    document.getElementById("cloud-label").textContent = "no soma coordinates";
    return;
  }
  cloud.setAnatomy(a);
  anatomyReady = true;
  const edgeBit = a.edges ? ` · ${a.edges.toLocaleString()} edges` : "";
  document.getElementById("cloud-label").textContent =
    `${a.points.toLocaleString()} somata${edgeBit} · phasic sparks`;
}

async function pull() {
  const res = await fetch("/state", { cache: "no-store" });
  if (!res.ok) return;
  const s = await res.json();
  if (s.status !== "running") return;

  if (!anatomyReady) await loadAnatomy();

  // Charts and the reel log are useful for local reels too; one dashboard for all.
  enableShortsLayout();

  // The 3D observation chamber (chamber.js) is a module and may load after
  // the first poll; it shares this single /state poller.
  window.FlyChamber?.update(s);

  const img = document.getElementById("frame");
  if (s.frame_jpeg) img.src = s.frame_jpeg;

  document.getElementById("reel-title").textContent = s.reel?.title || "—";
  document.getElementById("reel-sub").textContent =
    `${s.reel?.id || ""} · tick ${s.tick} · ${s.interest?.reason || ""}` +
    (s.capture_backend ? ` · ${s.capture_backend}` : "");

  const privacy = s.privacy || {};
  document.getElementById("privacy-badge").textContent = privacy.enabled
    ? `privacy: faces ${privacy.faces || 0} · text ${privacy.text_regions || 0}`
    : (privacy.error ? `privacy off · ${privacy.error}` : "privacy off");

  const interest = s.interest || {};
  const pct = Math.max(0, Math.min(100, ((interest.interest || 0) / 120) * 100));
  document.getElementById("interest-bar").style.width = `${pct}%`;
  const scrollError = document.getElementById("scroll-error");
  scrollError.hidden = !s.scroll_error;
  scrollError.textContent = s.scroll_error || "";
  document.getElementById("interest").textContent = `${interest.interest ?? "—"} Hz`;
  document.getElementById("phasic").textContent = `${interest.phasic_novelty ?? "—"} Hz`;
  document.getElementById("dwell").textContent =
    `${interest.low_novelty_seconds ?? "—"} / ${interest.boredom_dwell_seconds ?? "—"} s` +
    (interest.low_novelty_samples != null
      ? ` · samples ${interest.low_novelty_samples}/${interest.minimum_low_samples}`
      : "") +
    (interest.boredom_bar != null ? ` · bar ${interest.boredom_bar}` : "");
  document.getElementById("active").textContent = `${s.active_neurons}`;
  renderBumble(s.bumble);

  if (anatomyReady && s.cloud?.activity_b64) {
    cloud.setActivity(s.cloud.activity_b64, s.cloud.points);
  }
  drawFlyEye(s.fly_eye);

  if (s.analytics) {
    const a = s.analytics;
    const marks = (a.scroll_events || []).map((e) => ({ t: e.t, reason: e.reason }));
    // Canvas heights come from the CSS grid so the dashboard fits one screen.
    drawSeries(document.getElementById("chart-interest"), a.live_interest, {
      color: cssVar("--accent", "#c6f355"),
      overlays: [{ data: a.live_novelty, color: "#6ec8ff", lineWidth: 1.35 }],
      marks,
      ymin: 0,
    });
    const outcomes = a.outcome_counts || {};
    const bored = Number(outcomes.bored || 0);
    const full = Number(outcomes.full_watch || 0);
    const outcomeMax = Math.max(1, bored, full);
    document.getElementById("outcome-bored").textContent = `${bored}`;
    document.getElementById("outcome-full").textContent = `${full}`;
    document.getElementById("outcome-total").textContent = `${bored + full} content`;
    document.getElementById("outcome-bored-bar").style.width =
      `${(100 * bored / outcomeMax).toFixed(1)}%`;
    document.getElementById("outcome-full-bar").style.width =
      `${(100 * full / outcomeMax).toFixed(1)}%`;
    const avgSeries = (a.avg_watch_history || []).map((p) => ({ t: p.n, v: p.avg }));
    drawSeries(document.getElementById("chart-avg"), avgSeries, {
      color: "#e2b0ff",
      ymin: 0,
      ymax: a.max_watch_seconds || 5,
    });
    const avg = a.avg_watch_seconds;
    document.getElementById("avg-watch").textContent =
      avg == null ? "—" : `${avg.toFixed(2)} s · n=${a.watches}`;
    const dist = a.watch_time_distribution;
    document.getElementById("watch-dist-v").textContent = dist
      ? `p50 ${dist.p50}s · p10–p90 ${dist.p10}–${dist.p90}`
      : "—";
    renderReelCards(a.reels || []);
  }

  // Bumble only: fire the ✓/✕ zoom when the like or pass count ticks up.
  const bumble = s.bumble;
  if (bumble) {
    if (prevLikes !== null && bumble.likes > prevLikes) showSwipeVerdict("like");
    else if (prevPasses !== null && bumble.passes > prevPasses) showSwipeVerdict("pass");
    prevLikes = bumble.likes;
    prevPasses = bumble.passes;
  } else {
    prevLikes = prevPasses = null;
  }
}

async function pollLoop() {
  try {
    await pull();
  } catch (_) {
    // A later poll can recover from startup or transient HTTP failures.
  }
  setTimeout(pollLoop, 200);
}

loadAnatomy().catch(() => {});
pollLoop();
