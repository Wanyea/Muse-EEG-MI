const logEl = document.getElementById("log");
const liveEl = document.getElementById("live");
const predOutEl = document.getElementById("predOut");
const trainOutEl = document.getElementById("trainOut");
const cmInfoEl = document.getElementById("cmInfo");
const sessionsTbody = document.querySelector("#sessionsTable tbody");

// Training overlay refs
const overlay = document.getElementById("trainingOverlay");
const trainingCueEl = document.getElementById("trainingCue");
const trainingCountdownEl = document.getElementById("trainingCountdown");
const trainingStatusEl = document.getElementById("trainingStatus");
const trainingProgressEl = document.getElementById("trainingProgress");
const overlayStopBtn = document.getElementById("overlayStopBtn");

function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

async function post(url, body = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return await res.json();
}

async function get(url) {
  const res = await fetch(url);
  return await res.json();
}

function selectedClasses() {
  return Array.from(document.querySelectorAll(".cls"))
    .filter(cb => cb.checked)
    .map(cb => cb.value);
}

function arrowFor(label) {
  return ({UP:"↑",DOWN:"↓",LEFT:"←",RIGHT:"→"}[label] || "•");
}

function showOverlay() {
  overlay.style.display = "block";
  document.body.style.overflow = "hidden";
}
function hideOverlay() {
  overlay.style.display = "none";
  document.body.style.overflow = "auto";
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function beep(freq=880, durationMs=120, gain=0.05) {
  // Consistent auditory cue for phase changes
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.connect(g); g.connect(ctx.destination);
    o.type = "sine";
    o.frequency.value = freq;
    g.gain.value = gain;
    o.start();
    setTimeout(() => { o.stop(); ctx.close(); }, durationMs);
  } catch (e) {}
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && overlay.style.display === "block") {
    blockRunning = false;
    hideOverlay();
    log("[UI] Exited training overlay.");
  }
});

overlayStopBtn.onclick = async () => {
  blockRunning = false;
  const out = await post("/block/stop", {});
  log("[UI] block_stop: " + JSON.stringify(out));
  trainingStatusEl.textContent = "Stopped.";
  trainingCueEl.textContent = "•";
  setTimeout(() => hideOverlay(), 600);
};

// ---------- Socket.IO ----------
const socket = io();
socket.on("connect", () => log("[UI] Socket connected"));
socket.on("muse_data", (pkt) => {
  const eeg = pkt.eeg || {};
  const acc = pkt.acc || {};
  const gyro = pkt.gyro || {};
  liveEl.innerHTML = `
    <div><b>EEG</b> TP9:${(eeg.tp9 ?? 0).toFixed(2)} AF7:${(eeg.af7 ?? 0).toFixed(2)} AF8:${(eeg.af8 ?? 0).toFixed(2)} TP10:${(eeg.tp10 ?? 0).toFixed(2)}</div>
    <div><b>ACC</b> x:${(acc.x ?? 0).toFixed(2)} y:${(acc.y ?? 0).toFixed(2)} z:${(acc.z ?? 0).toFixed(2)}</div>
    <div><b>GYRO</b> x:${(gyro.x ?? 0).toFixed(2)} y:${(gyro.y ?? 0).toFixed(2)} z:${(gyro.z ?? 0).toFixed(2)}</div>
    <div><b>Battery</b> ${(pkt.battery ?? 0).toFixed(1)}%</div>
  `;
});

// ---------- Session Controls ----------
document.getElementById("startSessionBtn").onclick = async () => {
  const subject_id = document.getElementById("subjectId").value.trim();
  const out = await post("/session/start", { subject_id });
  log("[UI] start_session: " + JSON.stringify(out));
};

document.getElementById("endSessionBtn").onclick = async () => {
  const out = await post("/session/end", {});
  log("[UI] end_session: " + JSON.stringify(out));
};

document.getElementById("exportBtn").onclick = async () => {
  const out = await post("/export", {});
  log("[UI] export: " + JSON.stringify(out));
};

// ---------- Block / Training Mode ----------
let blockRunning = false;

document.getElementById("runBlockBtn").onclick = async () => {
  if (blockRunning) return;
  blockRunning = true;

  const classes = selectedClasses();
  const n_trials = parseInt(document.getElementById("nTrials").value, 10);

  const out = await post("/block/start", { n_trials, classes });
  if (out.error) {
    log("[UI] block_start error: " + JSON.stringify(out));
    blockRunning = false;
    return;
  }

  showOverlay();

  trainingStatusEl.textContent = "Get ready…";
  trainingCueEl.textContent = "•";
  trainingProgressEl.textContent = `Trial 0/${n_trials}`;
  trainingCountdownEl.textContent = "";

  // Countdown (3..2..1)
  for (let i = 3; i >= 1; i--) {
    trainingCountdownEl.textContent = `${i}`;
    beep(660, 120);
    await sleep(700);
  }
  trainingCountdownEl.textContent = "";
  beep(880, 140);

  await loopBlockOverlay(n_trials);

  // End-of-block screen
  if (overlay.style.display === "block") {
    trainingCountdownEl.textContent = "";
    trainingCueEl.textContent = "✓";
    trainingStatusEl.textContent = "Block complete.";
    beep(880, 120); setTimeout(() => beep(990, 120), 170);
    setTimeout(() => hideOverlay(), 1200);
  }

  blockRunning = false;
};

document.getElementById("stopBlockBtn").onclick = async () => {
  blockRunning = false;
  const out = await post("/block/stop", {});
  log("[UI] block_stop: " + JSON.stringify(out));
  trainingCueEl.textContent = "•";
  trainingStatusEl.textContent = "Stopped.";
  hideOverlay();
};

async function loopBlockOverlay(n_trials) {
  let trialIndex = 0;

  while (blockRunning) {
    const step = await post("/block/next", {});
    if (step.done) break;
    if (step.error) {
      trainingStatusEl.textContent = step.error;
      trainingCueEl.textContent = "•";
      break;
    }

    trialIndex = step.trial_index ?? (trialIndex + 1);
    trainingProgressEl.textContent = `Trial ${trialIndex}/${n_trials}`;

    // Expected server payload:
    // step.schedule = [{phase:"rest", duration_s:1.5},{phase:"imagery", duration_s:3.0, label:"LEFT"}, {phase:"rest", duration_s:1.5}]
    const schedule = step.schedule || [
      { phase: "rest", duration_s: 1.5 },
      { phase: "imagery", duration_s: step.total_s || 3.0, label: step.label },
    ];

    for (let i = 0; i < schedule.length && blockRunning; i++) {
      const seg = schedule[i];
      const phase = seg.phase || "imagery";
      const durMs = Math.max(0, (seg.duration_s || 0) * 1000);

      // Phase change cues
      if (phase === "rest") {
        // Fixation dot during rest to minimize saccades
        trainingCueEl.textContent = "•";
        trainingStatusEl.textContent = "Rest (fixate)";
        beep(440, 90);
      } else if (phase === "imagery") {
        trainingCueEl.textContent = arrowFor(seg.label || step.label);
        trainingStatusEl.textContent = "Imagine movement";
        beep(880, 90);
      } else {
        trainingCueEl.textContent = "•";
        trainingStatusEl.textContent = phase;
        beep(550, 90);
      }

      await sleep(durMs);
    }

    // Prediction (returns {status:not_trained} until trained)
    const pred = await post("/predict_last", {});
    predOutEl.textContent = JSON.stringify(pred, null, 2);
  }
}

// ---------- Sessions / Training ----------
async function refreshSessions() {
  const data = await get("/sessions");
  const sessions = data.sessions || [];
  sessionsTbody.innerHTML = "";

  for (const s of sessions) {
    const cc = s.class_counts || {};
    const classPills = Object.entries(cc)
      .map(([k,v]) => `<span class="pill">${k}:${v}</span>`)
      .join("");

    const last = s.last_train || {};
    const acc = (last.accuracy != null) ? Number(last.accuracy).toFixed(3) : "-";
    const bacc = (last.balanced_accuracy != null) ? Number(last.balanced_accuracy).toFixed(3) : "-";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><input type="checkbox" class="sessSel" value="${s.session_id}"></td>
      <td>${s.session_id}</td>
      <td>${s.n_exported_trials ?? 0}</td>
      <td>${classPills}</td>
      <td>${acc}</td>
      <td>${bacc}</td>
    `;
    sessionsTbody.appendChild(tr);
  }

  log(`[UI] sessions refreshed: ${sessions.length}`);
}

document.getElementById("refreshSessionsBtn").onclick = refreshSessions;

document.getElementById("trainSelectedBtn").onclick = async () => {
  const session_ids = Array.from(document.querySelectorAll(".sessSel"))
    .filter(cb => cb.checked)
    .map(cb => cb.value);

  const classes = selectedClasses();
  if (session_ids.length === 0) {
    log("[UI] No sessions selected.");
    return;
  }
  if (classes.length < 2) {
    log("[UI] Select at least 2 classes.");
    return;
  }

  const out = await post("/train_sessions", { session_ids, classes });
  log("[UI] train_sessions: " + JSON.stringify(out.combined || out));

  trainOutEl.textContent = JSON.stringify(out, null, 2);

  if (out.combined && out.combined.confusion_matrix) {
    drawConfusionMatrix(out.combined);
  }

  refreshSessions();
};

function drawConfusionMatrix(result) {
  const cm = result.confusion_matrix;
  const labels = result.classes;

  cmInfoEl.textContent = `Combined model — Acc=${Number(result.accuracy).toFixed(3)}  BAcc=${Number(result.balanced_accuracy).toFixed(3)}  Trials=${result.n_trials}`;

  const data = [{
    z: cm,
    x: labels,
    y: labels,
    type: "heatmap",
    hoverongaps: false
  }];

  const layout = {
    margin: { t: 20 },
    xaxis: { title: "Predicted" },
    yaxis: { title: "True", autorange: "reversed" }
  };

  Plotly.newPlot("cmPlot", data, layout, {displayModeBar:false});
}

// initial load
refreshSessions();
