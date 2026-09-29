/* Rocket Tracker app: talks to the local JSON API in app.py. No build step, no framework. */
'use strict';

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v == null || !isFinite(v)) ? '–' : Number(v).toFixed(d);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : v; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: 'POST',
    headers: body instanceof Blob ? {} : { 'Content-Type': 'application/json' },
    body: body instanceof Blob ? body : JSON.stringify(body),
  };
  const r = await fetch(path, { cache: 'no-store', ...opts });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

function toast(text) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = text;
  $('toasts').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

/* ---------- geometry ---------- */
const R_EARTH = 6371000, RAD = Math.PI / 180;
function distance(a, b) {
  const dLat = (b[0] - a[0]) * RAD, dLon = (b[1] - a[1]) * RAD;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a[0] * RAD) * Math.cos(b[0] * RAD) * Math.sin(dLon / 2) ** 2;
  return 2 * R_EARTH * Math.asin(Math.sqrt(h));
}
function bearing(a, b) {
  const y = Math.sin((b[1] - a[1]) * RAD) * Math.cos(b[0] * RAD);
  const x = Math.cos(a[0] * RAD) * Math.sin(b[0] * RAD) - Math.sin(a[0] * RAD) * Math.cos(b[0] * RAD) * Math.cos((b[1] - a[1]) * RAD);
  return (Math.atan2(y, x) / RAD + 360) % 360;
}
const COMPASS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
const compass = (deg) => COMPASS[Math.round(deg / 45) % 8];
const fmtDist = (m) => m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${fmt(m)} m`;
function fmtTime(s) {
  if (s == null || !isFinite(s)) return '–';
  s = Math.max(0, s);
  return s >= 60 ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}` : `${s.toFixed(0)} s`;
}

/* ---------- tiny SVG charts ---------- */
function niceTicks(min, max, count = 4) {
  if (!(max > min)) max = min + 1;
  const raw = (max - min) / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => (max - min) / s <= count) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}
let chartSeq = 0;
function lineChart(el, { series = [], xLabel = '', yLabel = '', xMin, xMax, yMin = 0, yMax, markers = [], extra = '' }) {
  const W = el.clientWidth || 360, H = el.clientHeight || 150;
  const m = { l: 44, r: 10, t: 8, b: 24 };
  const pts = series.flatMap((s) => s.points);
  if (!pts.length) { el.innerHTML = `<svg viewBox="0 0 ${W} ${H}"><text x="${W / 2}" y="${H / 2}" text-anchor="middle">No data yet</text></svg>`; return null; }
  const x0 = xMin ?? Math.min(...pts.map((p) => p[0])), x1 = xMax ?? Math.max(...pts.map((p) => p[0]), x0 + 1);
  const y0 = yMin ?? Math.min(...pts.map((p) => p[1]));
  const y1 = yMax ?? Math.max(...pts.map((p) => p[1]), y0 + 1) * 1.06;
  const X = (v) => m.l + (v - x0) / (x1 - x0 || 1) * (W - m.l - m.r);
  const Y = (v) => H - m.b - (v - y0) / (y1 - y0 || 1) * (H - m.t - m.b);
  let g = '';
  for (const t of niceTicks(y0, y1)) g += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${m.l - 6}" y="${Y(t) + 3}" text-anchor="end">${t}</text>`;
  for (const t of niceTicks(x0, x1, 5)) g += `<text x="${X(t)}" y="${H - 8}" text-anchor="middle">${t}</text>`;
  g += `<text x="${W - m.r}" y="${H - m.b - 4}" text-anchor="end">${esc(xLabel)}</text>`;
  g += `<text x="${m.l + 4}" y="${m.t + 9}">${esc(yLabel)}</text>`;
  const clip = `clip${++chartSeq}`;
  g += `<clipPath id="${clip}"><rect x="${m.l}" y="${m.t - 4}" width="${W - m.l - m.r + 4}" height="${H - m.t - m.b + 8}"/></clipPath><g clip-path="url(#${clip})">`;
  for (const s of series) {
    if (s.points.length < 2) continue;
    const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''} stroke-linejoin="round"/>`;
  }
  g += '</g>';
  for (const mk of markers) g += `<circle cx="${X(mk[0])}" cy="${Y(mk[1])}" r="${mk[3] || 3.5}" fill="${mk[2]}" stroke="var(--panel)" stroke-width="1.5"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${g}${extra ? extra(X, Y) : ''}</svg>`;
  return { X, Y };
}

/* ---------- map ---------- */
const map = L.map('map', { zoomControl: true, preferCanvas: false }).setView([37.3954, -121.8518], 14);
const tileOpts = { maxZoom: 19, maxNativeZoom: 18 };
const baseLayers = {
  Satellite: L.tileLayer('/tiles/satellite/{z}/{x}/{y}', { ...tileOpts, attribution: 'Imagery © Esri' }),
  Map: L.tileLayer('/tiles/osm/{z}/{x}/{y}', { ...tileOpts, attribution: '© OpenStreetMap contributors' }),
};
baseLayers[store.get('rt-base', 'Satellite')]?.addTo(map) || baseLayers.Satellite.addTo(map);
L.control.layers(baseLayers).addTo(map);
map.on('baselayerchange', (e) => store.set('rt-base', e.name));
L.control.scale({ imperial: false }).addTo(map);

const canvas = L.canvas({ padding: 0.3 });
const layers = {
  padForecast: L.polygon([], { color: '#b58cff', weight: 2, dashArray: '5 6', fillOpacity: 0.08 }).addTo(map),
  padPath: L.polyline([], { color: '#b58cff', weight: 1.5, dashArray: '2 6', opacity: 0.8 }).addTo(map),
  history: L.polyline([], { color: '#ff5d5d', weight: 1, opacity: 0.25 }).addTo(map),
  ellipse: L.polygon([], { color: '#ff5d5d', weight: 2, fillOpacity: 0.12 }).addTo(map),
  samples: L.layerGroup().addTo(map),
  gps: L.layerGroup().addTo(map),
  path: L.polyline([], { color: '#ffb547', weight: 2.5, dashArray: '7 7' }).addTo(map),
  track: L.polyline([], { color: '#4fa3ff', weight: 3.5 }).addTo(map),
};
const padMarker = L.circleMarker([0, 0], { radius: 7, color: '#fff', weight: 2, fillColor: '#b58cff', fillOpacity: 1 }).bindTooltip('Launch pad');
const landMarker = L.circleMarker([0, 0], { radius: 7, color: '#fff', weight: 2, fillColor: '#ff5d5d', fillOpacity: 1 }).bindTooltip('Predicted landing');
const rocketMarker = L.marker([0, 0], { icon: L.divIcon({ className: '', html: '<div class="rocket-icon"></div>', iconSize: [18, 18], iconAnchor: [9, 9] }), zIndexOffset: 1000 });
const truthMarker = L.marker([0, 0], { icon: L.divIcon({ className: '', html: '<div style="font-size:22px;color:#fff;text-shadow:0 0 4px #000">★</div>', iconSize: [22, 22], iconAnchor: [11, 13] }) }).bindTooltip('Actual landing (simulated)');
const meMarker = L.marker([0, 0], { icon: L.divIcon({ className: '', html: '<div class="me-icon"></div>', iconSize: [14, 14], iconAnchor: [7, 7] }) }).bindTooltip('You');

let S = null;            // latest state
let cfg = null;          // latest config
let me = null;           // [lat, lon] from geolocation
let follow = false, fittedFor = null, pickingSite = false;
let lastEventCount = 0, lastRunStarted = null;

function recolor() {
  layers.track.setStyle({ color: cssVar('--track') });
  layers.path.setStyle({ color: cssVar('--pred') });
  layers.ellipse.setStyle({ color: cssVar('--land') });
  layers.history.setStyle({ color: cssVar('--land') });
  layers.padForecast.setStyle({ color: cssVar('--pad') });
  layers.padPath.setStyle({ color: cssVar('--pad') });
}

function fitAll() {
  if (!S) return;
  const pts = [[S.site.lat, S.site.lon]];
  const p = S.prediction || S.pad_forecast;
  if (p) pts.push(...p.ellipse_95.polygon, ...p.path.map((q) => [q[0], q[1]]));
  if (S.track) pts.push(...S.track.map((q) => [q[0], q[1]]));
  if (me) pts.push(me);
  map.fitBounds(L.latLngBounds(pts).pad(0.15), { maxZoom: 17 });
}

function drawMap(s) {
  const site = [s.site.lat, s.site.lon];
  padMarker.setLatLng(site).addTo(map);
  const launched = s.launched;
  const pf = !launched ? s.pad_forecast : null;
  layers.padForecast.setLatLngs(pf ? pf.ellipse_95.polygon : []);
  layers.padPath.setLatLngs(pf ? pf.path.map((q) => [q[0], q[1]]) : []);

  const p = s.prediction;
  layers.track.setLatLngs(launched ? s.track.map((q) => [q[0], q[1]]) : []);
  layers.path.setLatLngs(p ? p.path.map((q) => [q[0], q[1]]) : []);
  layers.ellipse.setLatLngs(p ? p.ellipse_95.polygon : []);
  layers.history.setLatLngs(p ? (s.landing_history || []).map((h) => [h[1], h[2]]) : []);
  layers.samples.clearLayers();
  if (p) p.samples.forEach((q) => L.circleMarker(q, { renderer: canvas, radius: 1.8, stroke: false, fillColor: cssVar('--land'), fillOpacity: 0.55 }).addTo(layers.samples));
  layers.gps.clearLayers();
  (s.gps || []).forEach((q) => L.circleMarker(q, { renderer: canvas, radius: 2, stroke: false, fillColor: cssVar('--gps'), fillOpacity: 0.7 }).addTo(layers.gps));
  if (p) landMarker.setLatLng(p.landing).addTo(map); else landMarker.remove();
  if (launched && s.state) {
    rocketMarker.setLatLng([s.state.lat, s.state.lon]).addTo(map);
    if (follow) map.panTo([s.state.lat, s.state.lon], { animate: false });
  } else rocketMarker.remove();
  if (s.run.truth_landing) truthMarker.setLatLng(s.run.truth_landing).addTo(map); else truthMarker.remove();

  const key = `${s.run.started_at || 'idle'}:${launched}`;
  if (fittedFor !== key && (p || pf)) { fitAll(); fittedFor = key; }
}

/* ---------- top bar ---------- */
function drawTopbar(s) {
  $('rocketName').textContent = s.rocket;
  const phase = s.state ? s.state.phase : 'pad';
  const order = s.phases;
  const idx = order.indexOf(phase);
  $('timeline').innerHTML = order.map((ph, i) => `<li class="${i < idx ? 'done' : i === idx ? 'now' : ''}">${ph}</li>`).join('');
  const st = s.run.state;
  const pill = $('runState');
  pill.textContent = st === 'finished' ? 'landed · finished' : st;
  pill.className = `pill ${st}`;
  const active = ['running', 'paused', 'starting'].includes(st);
  $('pauseBtn').disabled = !active || s.run.source === 'follow';
  $('stopBtn').disabled = !active;
  $('speedTop').disabled = !active || s.run.source === 'follow';
  $('pauseBtn').textContent = st === 'paused' ? 'Resume' : 'Pause';
  if (active && document.activeElement !== $('speedTop')) $('speedTop').value = String(s.run.speed);
  $('startCard').hidden = active;
  $('setupLocked').hidden = !active;
  $('setupForm').querySelectorAll('input,select,textarea,button').forEach((e) => { e.disabled = active; });
  ['saveCfg', 'resetCfg'].forEach((id) => { $(id).disabled = active; });
  $('importCfg').disabled = active;
}

/* ---------- flight tab ---------- */
function drawFlight(s) {
  const launched = s.launched;
  const p = launched ? s.prediction : s.pad_forecast;
  const site = [s.site.lat, s.site.lon];
  $('landingTitle').textContent = launched ? 'Predicted landing' : 'If launched now';
  $('landingBadge').textContent = launched ? (s.state.phase === 'landed' ? 'landed' : 'live') : 'forecast';
  $('landingBadge').className = `badge ${launched ? 'live' : ''}`;
  if (p) {
    const d = distance(site, p.landing), b = bearing(site, p.landing);
    $('lDist').textContent = fmtDist(d);
    $('lBear').textContent = `${fmt(b)}° ${compass(b)}`;
    const eta = launched ? p.time_to_landing - (s.state.t - p.t) : p.time_to_landing;
    $('lEta').textContent = launched && s.state.phase === 'landed' ? 'Down' : fmtTime(eta);
    $('lEtaSub').textContent = launched ? (s.state.phase === 'landed' ? `at T+${fmt(s.state.t)} s` : 'until touchdown') : 'flight time';
    $('lEll').textContent = `${fmt(p.ellipse_95.semi_major_m * 2)}×${fmt(p.ellipse_95.semi_minor_m * 2)}`;
    $('lApogee').textContent = `${fmt(p.apogee_agl)} m AGL`;
    $('lCoord').textContent = `${p.landing[0].toFixed(5)}, ${p.landing[1].toFixed(5)}`;
    $('mapsLink').href = `https://www.google.com/maps/search/?api=1&query=${p.landing[0]},${p.landing[1]}`;
    if (me) {
      const dm = distance(me, p.landing), bm = bearing(me, p.landing);
      $('lFromMe').textContent = `${fmtDist(dm)} toward ${fmt(bm)}° ${compass(bm)}`;
    }
  }
  if (s.run.truth_landing && p) {
    $('truthRow').hidden = false;
    $('lTruth').textContent = `${fmtDist(distance(s.run.truth_landing, p.landing))} from prediction`;
  } else $('truthRow').hidden = true;

  // Altitude vs time, with the predicted rest of the flight.
  const tr = launched ? s.track : [];
  const now = launched ? s.state.t : 0;
  const future = p ? p.path.map((q) => [(launched ? p.t : 0) + q[3], q[2]]) : [];
  lineChart($('chartAlt'), {
    series: [{ points: tr.map((q) => [q[3], q[2]]), color: cssVar('--track') },
      { points: future, color: launched ? cssVar('--pred') : cssVar('--pad'), dash: '5 5' }],
    xLabel: 'seconds', yLabel: 'm AGL', xMin: 0,
    markers: launched ? [[now, s.state.alt_agl, cssVar('--text')]] : [],
  });
  lineChart($('chartSide'), {
    series: [{ points: tr.map((q) => [q[5], q[2]]), color: cssVar('--track') },
      { points: p ? p.path.map((q) => [q[4], q[2]]) : [], color: launched ? cssVar('--pred') : cssVar('--pad'), dash: '5 5' }],
    xLabel: 'm from pad', yLabel: 'm AGL', xMin: 0,
    markers: p ? [[distance(site, p.landing), 0, cssVar('--land')]] : [],
  });
  const hist = launched ? (s.landing_history || []) : [];
  lineChart($('chartUnc'), {
    series: [{ points: hist.map((h) => [h[0], h[3] * 2]), color: cssVar('--land') }],
    xLabel: 'seconds', yLabel: 'm', xMin: 0,
  });
  if (launched) {
    const st = s.state;
    $('hudT').textContent = `${fmt(st.t, 1)} s`;
    $('hudAlt').textContent = `${fmt(st.alt_agl)} m`;
    $('hudSpd').textContent = `${fmt(st.speed)} m/s`;
    $('hudVz').textContent = `${st.velocity[2] >= 0 ? '▲' : '▼'} ${fmt(Math.abs(st.velocity[2]))} m/s`;
    $('fWind').textContent = `${fmt(Math.hypot(...st.wind_bias), 1)} m/s (${fmt(st.wind_bias[0], 1)} E, ${fmt(st.wind_bias[1], 1)} N)`;
    $('fDrag').textContent = `${fmt(st.drag_scale, 2)}×`;
    $('fSigma').textContent = `${fmt(Math.hypot(st.pos_sigma[0], st.pos_sigma[1]), 1)} m horiz · ${fmt(st.pos_sigma[2], 1)} m vert`;
    $('fRej').textContent = st.rejected;
  } else {
    ['hudT', 'hudAlt', 'hudSpd', 'hudVz', 'fWind', 'fDrag', 'fSigma', 'fRej'].forEach((id) => { $(id).textContent = '–'; });
    $('hudT').textContent = s.run.state === 'running' ? 'waiting' : 'on pad';
  }
}

/* ---------- weather tab ---------- */
function drawWeather(s) {
  const w = s.weather, prof = s.weather_profile;
  const names = { 'open-meteo': 'Open-Meteo (live)', 'standard-atmosphere': 'Standard atmosphere' };
  $('wxSource').textContent = names[w.source] || w.source;
  $('wxValid').textContent = w.valid_time ? w.valid_time.replace('T', ' ').slice(0, 16) + ' UTC' : 'n/a';
  $('wxFetched').textContent = w.fetched_at ? new Date(w.fetched_at).toLocaleTimeString() : 'n/a';
  $('wxEvery').textContent = `every ${fmt(w.refresh_s / 60, 1)} min · ${w.updates} update(s)`;
  $('wxErr').hidden = !w.error;
  $('wxErr').textContent = w.error ? `Live fetch failed, using ${w.source}: ${w.error}` : '';
  $('wxNow').textContent = s.state ? `${fmt(s.state.forecast_wind.speed, 1)} m/s from ${fmt(s.state.forecast_wind.from_deg)}°` : '–';

  const siteAlt = s.site.alt;
  const levels = prof.levels.map((l) => ({ ...l, agl: l.alt_msl - siteAlt, speed: Math.hypot(l.wind_east, l.wind_north) }));
  const apogee = (s.prediction || s.pad_forecast || {}).apogee_agl || 3000;
  const top = Math.max(1500, apogee * 1.4);
  const shown = levels.filter((l) => l.agl <= top * 1.2);
  const rows = shown.length ? shown : levels.slice(0, 5);
  const rocketAlt = s.state ? s.state.alt_agl : null;
  lineChart($('chartWind'), {
    series: [{ points: rows.map((l) => [l.speed, l.agl]), color: cssVar('--accent') }],
    xLabel: 'm/s', yLabel: 'm AGL', xMin: 0, yMin: 0, yMax: top,
    extra: (X, Y) => rows.filter((l) => l.agl <= top).map((l) => {
      const ang = Math.atan2(-l.wind_north, l.wind_east) / RAD; // screen angle of the blow-to direction
      return `<g transform="translate(${X(l.speed)},${Y(l.agl)}) rotate(${ang})"><path d="M-9,0 L7,0 M2,-4 L7,0 L2,4" stroke="var(--text)" stroke-width="1.6" fill="none"/></g>`;
    }).join('') + (rocketAlt != null ? `<line x1="44" x2="100%" y1="${Y(rocketAlt)}" y2="${Y(rocketAlt)}" stroke="${cssVar('--track')}" stroke-dasharray="4 4"/><text x="48" y="${Y(rocketAlt) - 4}" fill="${cssVar('--track')}">rocket</text>` : ''),
  });
  $('wxTable').tBodies[0].innerHTML = rows.map((l) => {
    const from = (Math.atan2(-l.wind_east, -l.wind_north) / RAD + 360) % 360;
    return `<tr><td>${fmt(l.alt_msl)} m</td><td>${fmt(l.speed, 1)} m/s</td><td>${fmt(from)}° ${compass(from)}</td><td>${fmt(l.temperature_c, 1)} °C</td><td>${fmt(l.pressure_hpa)} hPa</td></tr>`;
  }).join('');
}

/* ---------- log tab ---------- */
function drawLog(s) {
  $('events').innerHTML = s.events.slice().reverse().map((e) => `<li><time>T+${fmt(e[0], 1)}s</time><span>${esc(e[1])}</span></li>`).join('') || '<li><span class="hint">No events yet</span></li>';
  const runKey = s.run.started_at || null;
  if (runKey !== lastRunStarted) { lastRunStarted = runKey; lastEventCount = s.events.length; return; }
  if (s.events.length > lastEventCount) s.events.slice(lastEventCount).forEach((e) => toast(e[1]));
  lastEventCount = s.events.length;
}

/* ---------- polling ---------- */
let polling = false;
async function tick() {
  if (polling) return;
  polling = true;
  try {
    S = await api('/api/state');
    drawTopbar(S);
    drawMap(S);
    drawFlight(S);
    drawWeather(S);
    drawLog(S);
    if (S.run.state === 'error') { $('startErr').hidden = false; $('startErr').textContent = S.run.error; }
  } catch (e) {
    $('runState').textContent = 'offline';
    $('runState').className = 'pill error';
  } finally { polling = false; }
}

/* ---------- run controls ---------- */
let source = 'sim';
$('sourceSeg').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  source = b.dataset.src;
  $('sourceSeg').querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
  document.querySelectorAll('[data-hint]').forEach((el) => { el.hidden = el.dataset.hint !== source; });
  $('flipLon').closest('label').hidden = source === 'sim';
  if (source === 'replay') loadFiles();
});
$('flipLon').closest('label').hidden = true;
$('followPath').value = store.get('rt-follow', '');

async function loadFiles(select) {
  const r = await api('/api/files');
  $('dataDir').textContent = `Logs live in ${r.dir}`;
  $('fileSelect').innerHTML = r.files.length
    ? r.files.map((f) => `<option value="${esc(f.name)}">${esc(f.name)} (${fmt(f.size / 1024)} KB)</option>`).join('')
    : '<option value="">No logs yet, upload one</option>';
  if (select) $('fileSelect').value = select;
}
$('uploadInput').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    const r = await api(`/api/upload?name=${encodeURIComponent(f.name)}`, f);
    toast(`Uploaded ${r.name}`);
    await loadFiles(r.name);
  } catch (err) { toast(`Upload failed: ${err.message}`); }
  e.target.value = '';
});
$('startBtn').addEventListener('click', async () => {
  $('startErr').hidden = true;
  const req = { source, speed: +$('speedStart').value, offline: $('weatherMode').value === 'offline', flip_lon: $('flipLon').checked };
  if (source === 'replay') req.path = $('fileSelect').value;
  if (source === 'follow') { req.path = $('followPath').value.trim(); store.set('rt-follow', req.path); }
  if (source !== 'sim' && !req.path) { $('startErr').hidden = false; $('startErr').textContent = 'Choose a log file first.'; return; }
  try { await api('/api/run', req); follow = false; setFollow(false); tick(); } catch (err) { $('startErr').hidden = false; $('startErr').textContent = err.message; }
});
function togglePause() {
  if (!S || $('pauseBtn').disabled) return;
  api('/api/control', { action: S.run.state === 'paused' ? 'resume' : 'pause' }).then(tick).catch((e) => toast(e.message));
}
$('pauseBtn').addEventListener('click', togglePause);
$('stopBtn').addEventListener('click', () => api('/api/control', { action: 'stop' }).then(tick).catch((e) => toast(e.message)));
$('speedTop').addEventListener('change', (e) => api('/api/control', { speed: +e.target.value }).catch((err) => toast(err.message)));
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(document.activeElement.tagName)) { e.preventDefault(); togglePause(); }
});

/* ---------- map tools ---------- */
function setFollow(on) { follow = on; $('followBtn').setAttribute('aria-pressed', String(on)); }
$('followBtn').addEventListener('click', () => setFollow(!follow));
$('fitBtn').addEventListener('click', fitAll);
map.on('dragstart', () => setFollow(false));
let watchId = null;
$('meBtn').addEventListener('click', () => {
  if (!navigator.geolocation) { toast('Location is not available in this browser'); return; }
  if (watchId != null) { if (me) map.panTo(me); return; }
  watchId = navigator.geolocation.watchPosition((pos) => {
    const first = !me;
    me = [pos.coords.latitude, pos.coords.longitude];
    meMarker.setLatLng(me).addTo(map);
    if (first) { map.panTo(me); $('meBtn').setAttribute('aria-pressed', 'true'); }
  }, (err) => { toast(`Location: ${err.message}`); watchId = null; }, { enableHighAccuracy: true });
});
$('copyBtn').addEventListener('click', () => {
  navigator.clipboard?.writeText($('lCoord').textContent).then(() => toast('Coordinates copied'), () => toast('Copy failed'));
});

/* ---------- weather ---------- */
$('wxRefresh').addEventListener('click', async () => {
  $('wxRefresh').disabled = true;
  try {
    const r = await api('/api/weather/refresh', {});
    toast(r.ok ? 'Weather updated' : `Weather not updated: ${r.error}`);
  } catch (e) { toast(e.message); }
  $('wxRefresh').disabled = false;
  tick();
});

/* ---------- setup form ---------- */
const SECTIONS = [
  ['rocket', 'Rocket'], ['site', 'Launch site'], ['weather', 'Weather'],
  ['sensors', 'Sensors'], ['predictor', 'Prediction'], ['sim', 'Simulation (how the “real” flight differs)'],
];
const UNITS = [['_mps2', 'm/s²'], ['_kg', 'kg'], ['_hz', 'Hz'], ['_deg', '°'], ['_n', 'N'], ['_m', 'm'], ['_s', 's']];
const LABELS = {
  cd: 'Cd', agl: 'AGL', gps: 'GPS', sigma: 'σ', h: 'horizontal', v: 'vertical', dir: 'direction', tau: 'time constant',
};
function labelFor(key) {
  let unit = '';
  for (const [suf, u] of UNITS) if (key.endsWith(suf)) { unit = u; key = key.slice(0, -suf.length); break; }
  const words = key.split('_').map((w) => LABELS[w] || w);
  const text = words.join(' ').replace(/^./, (c) => c.toUpperCase());
  return unit ? `${text} (${unit})` : text;
}
function buildForm(c) {
  const form = $('setupForm');
  form.innerHTML = SECTIONS.map(([sec, title]) => {
    const fields = Object.entries(c[sec]).map(([k, v]) => {
      const id = `cfg-${sec}-${k}`;
      if (k === 'thrust_curve') {
        return `<label class="field full"><span>Thrust curve [[t, N], …] (optional, overrides average thrust)</span>
          <input id="${id}" data-sec="${sec}" data-key="${k}" data-type="json" value='${esc(JSON.stringify(v))}'></label>`;
      }
      if (k === 'provider') {
        return `<label class="field"><span>Weather source</span><select id="${id}" data-sec="${sec}" data-key="${k}">
          <option value="open-meteo"${v === 'open-meteo' ? ' selected' : ''}>Open-Meteo (live)</option>
          <option value="standard"${v === 'standard' ? ' selected' : ''}>Standard atmosphere</option></select></label>`;
      }
      const type = typeof v === 'number' ? 'number' : 'text';
      const cls = k === 'name' ? 'field full' : 'field';
      return `<label class="${cls}"><span>${esc(labelFor(k))}</span><input id="${id}" type="${type}" step="any" data-sec="${sec}" data-key="${k}" data-type="${type}" value="${esc(v)}"></label>`;
    }).join('');
    const extra = sec === 'site' ? `<div class="full" style="display:flex;gap:6px;margin-top:10px">
      <button type="button" class="btn small" id="pickSite">Pick on map</button>
      <button type="button" class="btn small" id="siteFromMe">Use my location</button></div>` : '';
    return `<fieldset><legend>${esc(title)}</legend><div class="form-grid">${fields}${extra}</div></fieldset>`;
  }).join('');
  $('pickSite').addEventListener('click', () => {
    pickingSite = true;
    map.getContainer().style.cursor = 'crosshair';
    toast('Click the map where the pad is');
  });
  $('siteFromMe').addEventListener('click', () => {
    navigator.geolocation?.getCurrentPosition((pos) => {
      $('cfg-site-latitude').value = pos.coords.latitude.toFixed(6);
      $('cfg-site-longitude').value = pos.coords.longitude.toFixed(6);
      if (pos.coords.altitude != null) $('cfg-site-altitude_m').value = pos.coords.altitude.toFixed(0);
      toast('Site set from your location, save to apply');
    }, (err) => toast(`Location: ${err.message}`));
  });
}
map.on('click', (e) => {
  if (!pickingSite) return;
  pickingSite = false;
  map.getContainer().style.cursor = '';
  $('cfg-site-latitude').value = e.latlng.lat.toFixed(6);
  $('cfg-site-longitude').value = e.latlng.lng.toFixed(6);
  toast('Pad moved, set its altitude and save');
});
function readForm() {
  const out = {};
  for (const el of $('setupForm').querySelectorAll('[data-sec]')) {
    const sec = el.dataset.sec, key = el.dataset.key;
    out[sec] = out[sec] || {};
    let v = el.value;
    if (el.dataset.type === 'number') {
      if (v === '' || !isFinite(+v)) throw new Error(`${labelFor(key)} needs a number`);
      v = +v;
    } else if (el.dataset.type === 'json') {
      try { v = JSON.parse(v || '[]'); } catch { throw new Error('Thrust curve must be JSON like [[0, 800], [1.5, 600]]'); }
    }
    out[sec][key] = v;
  }
  return out;
}
async function loadConfig() { cfg = await api('/api/config'); buildForm(cfg); }
async function saveConfig(data, msg) {
  $('setupErr').hidden = true;
  try {
    cfg = await api('/api/config', data);
    buildForm(cfg);
    fittedFor = null;
    toast(msg);
    tick();
  } catch (e) { $('setupErr').hidden = false; $('setupErr').textContent = e.message; }
}
$('saveCfg').addEventListener('click', () => {
  try { saveConfig(readForm(), 'Setup saved'); } catch (e) { $('setupErr').hidden = false; $('setupErr').textContent = e.message; }
});
$('resetCfg').addEventListener('click', async () => {
  if (!confirm('Reset every setting to the defaults?')) return;
  try { cfg = await api('/api/config/reset', {}); buildForm(cfg); fittedFor = null; toast('Defaults restored'); tick(); } catch (e) { toast(e.message); }
});
$('exportCfg').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${(cfg.rocket.name || 'rocket').replace(/\W+/g, '_')}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
$('importCfg').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try { await saveConfig(JSON.parse(await f.text()), `Loaded ${f.name}`); } catch (err) { $('setupErr').hidden = false; $('setupErr').textContent = `Could not import: ${err.message}`; }
  e.target.value = '';
});

/* ---------- tabs & theme ---------- */
document.querySelector('.tabs').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  document.querySelectorAll('.tabs button').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
  document.querySelectorAll('.tab-body').forEach((el) => { el.hidden = el.dataset.body !== b.dataset.tab; });
  store.set('rt-tab', b.dataset.tab);
  if (S) { drawFlight(S); drawWeather(S); }
});
const savedTab = store.get('rt-tab', 'flight');
document.querySelector(`.tabs button[data-tab="${savedTab}"]`)?.click();
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  store.set('rt-theme', t);
  recolor();
  if (S) tick();
}
applyTheme(store.get('rt-theme', window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'));
$('themeBtn').addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light'));
window.addEventListener('resize', () => { if (S) { drawFlight(S); drawWeather(S); } });

loadConfig();
tick();
setInterval(tick, 500);
