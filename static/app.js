const logEl = document.getElementById("log");
const liveEl = document.getElementById("live");
const cueEl = document.getElementById("cue");
const sessionIdEl = document.getElementById("sessionId");
const predOutEl = document.getElementById("predOut");
const refreshSessionsBtn = document.getElementById("refreshSessionsBtn");
const sessionsDropdown = document.getElementById("sessionsDropdown");
const trainSelectedBtn = document.getElementById("trainSelectedBtn");

async function refreshSessions() {
  const res = await fetch("/sessions");
  const data = await res.json();
  sessionsDropdown.innerHTML = "";
  (data.sessions || []).forEach(s => {
    const opt = document.createElement("option");
    opt.value = s.session_id;
    opt.textContent = s.session_id;
    sessionsDropdown.appendChild(opt);
  });
  log("[UI] sessions refreshed: " + (data.sessions || []).length);
}

refreshSessionsBtn.onclick = refreshSessions;

trainSelectedBtn.onclick = async () => {
  const session_id = sessionsDropdown.value;
  if (!session_id) {
    log("[UI] No session selected.");
    return;
  }
  const out = await post("/train_session", { session_id });
  log("[UI] train_session: " + JSON.stringify(out));
};

refreshSessions();

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

const socket = io();

socket.on("connect", () => log("[UI] Socket connected"));
socket.on("muse_data", (pkt) => {
  const eeg = pkt.eeg || {};
  const acc = pkt.acc || {};
  const gyro = pkt.gyro || {};
  liveEl.innerHTML = `
    <div><b>EEG</b> TP9:${eeg.tp9?.toFixed(2)} AF7:${eeg.af7?.toFixed(2)} AF8:${eeg.af8?.toFixed(2)} TP10:${eeg.tp10?.toFixed(2)}</div>
    <div><b>ACC</b> x:${acc.x?.toFixed(2)} y:${acc.y?.toFixed(2)} z:${acc.z?.toFixed(2)}</div>
    <div><b>GYRO</b> x:${gyro.x?.toFixed(2)} y:${gyro.y?.toFixed(2)} z:${gyro.z?.toFixed(2)}</div>
    <div><b>Battery</b> ${pkt.battery?.toFixed(1)}%</div>
  `;
});

document.getElementById("startSessionBtn").onclick = async () => {
  const subject_id = document.getElementById("subjectId").value.trim();
  const out = await post("/session/start", { subject_id });
  log("[UI] start_session: " + JSON.stringify(out));
  if (out.session_id) sessionIdEl.textContent = out.session_id;
};

document.getElementById("endSessionBtn").onclick = async () => {
  const out = await post("/session/end", {});
  log("[UI] end_session: " + JSON.stringify(out));
};

document.getElementById("exportBtn").onclick = async () => {
  const out = await post("/export", {});
  log("[UI] export: " + JSON.stringify(out));
};

document.getElementById("trainBtn").onclick = async () => {
  const out = await post("/train", {});
  log("[UI] train: " + JSON.stringify(out));
};

let blockRunning = false;

document.getElementById("runBlockBtn").onclick = async () => {
  if (blockRunning) return;
  blockRunning = true;
  cueEl.textContent = "•";
  const out = await post("/block/start", { n_trials: 20 }); // 20 randomized trials
  log("[UI] block_start: " + JSON.stringify(out));
  loopBlock();
};

document.getElementById("stopBlockBtn").onclick = async () => {
  blockRunning = false;
  const out = await post("/block/stop", {});
  log("[UI] block_stop: " + JSON.stringify(out));
  cueEl.textContent = "•";
};

async function loopBlock() {
  while (blockRunning) {
    const step = await post("/block/next", {});
    if (step.error) {
      log("[UI] " + JSON.stringify(step));
      cueEl.textContent = "•";
      blockRunning = false;
      break;
    }

    // step has phases; show cue during cue+imagery
    cueEl.textContent = step.display;

    // wait total time for trial (server already scheduled markers)
    await sleep(step.total_s * 1000);

    // fetch last prediction if available
    const pred = await post("/predict_last", {});
    predOutEl.textContent = JSON.stringify(pred, null, 2);
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}