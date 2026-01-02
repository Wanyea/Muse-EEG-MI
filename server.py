"""
Muse MI Trainer Server (v1)

- Your OpenMuse BLE pipeline
- Live Socket.IO streaming to UI
- Session/trial management + raw EEG recorder
- Cue runner + export + train + online prediction

Run:
  python server.py --address <BLE_ADDR> --preset p1041
Then open:
  http://localhost:5115
"""

import os
import time
import json
import asyncio
import argparse
import threading
import random
from collections import deque

import numpy as np
import pandas as pd

import bleak
from pythonosc.udp_client import SimpleUDPClient

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS

from OpenMuse.decode import parse_message, make_timestamps
from OpenMuse.muse import MuseS
from OpenMuse.utils import get_utc_timestamp

from mi.recorder import MuseRecorder
from mi.protocol import DEFAULT_PROTOCOL
from mi.model import MIModel


# -------------------------
# SHUTDOWN HANDLER
# -------------------------
import signal
shutdown_event = threading.Event()

def handle_signal(signum, frame):
    print(f"[OpenMuse] Signal {signum} received. Shutting down...")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# -------------------------
# CONFIG
# -------------------------
OSC_IP = "127.0.0.1"
OSC_PORT = 5000

WEBSOCKET_PORT = 5115

EEG_CHANNELS  = ["tp9", "af7", "af8", "tp10"]
ACC_CHANNELS  = ["x", "y", "z"]
GYRO_CHANNELS = ["x", "y", "z"]

DEFAULT_FS = 256  # Muse S EEG is commonly 256 Hz; adjust if your OpenMuse preset differs.
DATA_DIR = os.path.join(os.getcwd(), "sessions")

# -------------------------
# FLASK / SOCKET.IO
# -------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# -------------------------
# GLOBAL STATE
# -------------------------
client_osc: SimpleUDPClient | None = None

history = deque()  # last 120s packets for UI snapshots

latest_packet = {
    "timestamp": None,
    "eeg":  {ch: 0.0 for ch in EEG_CHANNELS},
    "acc":  {ch: 0.0 for ch in ACC_CHANNELS},
    "gyro": {ch: 0.0 for ch in GYRO_CHANNELS},
    "ppg":  [0.0] * 16,
    "battery": 0.0
}

LAST_EMIT = 0.0
EMIT_INTERVAL = 1.0 / 30.0

# Recorder + Model
recorder = MuseRecorder(EEG_CHANNELS, fs=DEFAULT_FS, buffer_seconds=300)
model = MIModel(fs=DEFAULT_FS, channels=EEG_CHANNELS)

current_session_id = None
current_trial_id = None

# Block runner state
block_active = False
block_queue = []
protocol = DEFAULT_PROTOCOL.copy()

# last predicted trial
last_prediction = {}

# -------------------------
# UTILS: UI emit + record, list & vaLidate sessions
# -------------------------
def _record_and_emit():
    global LAST_EMIT
    ts = time.time()
    latest_packet["timestamp"] = ts

    history.append((ts, json.loads(json.dumps(latest_packet))))
    cutoff = ts - 120.0
    while history and history[0][0] < cutoff:
        history.popleft()

    if ts - LAST_EMIT >= EMIT_INTERVAL:
        LAST_EMIT = ts
        socketio.emit("muse_data", latest_packet)

def _list_session_dirs(data_dir: str):
    if not os.path.exists(data_dir):
        return []
    items = []
    for name in sorted(os.listdir(data_dir), reverse=True):
        p = os.path.join(data_dir, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "session_meta.json")):
            items.append({"session_id": name, "path": p})
    return items

def _validate_session_dir(session_dir: str):
    export_index = os.path.join(session_dir, "export_index.json")
    if not os.path.exists(export_index):
        return False, "export_index.json not found. You must Export Trials for that session first."
    return True, ""

# -------------------------
# DECODE + OSC + UPDATE
# -------------------------
async def decode_and_send(data: bytearray, samples_sent: dict):
    global client_osc

    msg = f"{get_utc_timestamp()}\t{MuseS.EEG_UUID}\t{data.hex()}"
    decoded = parse_message(msg)

    # EEG
    if "EEG" in decoded:
        arr, *_ = make_timestamps(decoded["EEG"], None, None, None, None)

        # Guard: sometimes OpenMuse yields empty arrays
        if arr is None or not hasattr(arr, "shape") or arr.size == 0:
            return
        if arr.ndim != 2:
            return
        if arr.shape[0] == 0:
            return

        # Some pipelines may output only samples (no timestamp column)
        # Expected: (N, 1 + C)  where first column is timestamps
        if arr.shape[1] < 2:
            # Not enough columns for timestamp + channels
            return

        ts = arr[:, 0].astype(float)
        eeg = arr[:, 1:].astype(float)

        # If channel count doesn't match expected, skip (or adapt)
        if eeg.shape[1] != len(EEG_CHANNELS):
            # If it’s bigger, take first 4. If smaller, skip.
            if eeg.shape[1] > len(EEG_CHANNELS):
                eeg = eeg[:, :len(EEG_CHANNELS)]
            else:
                return

        samples_sent["EEG"] += len(eeg)

        # Convert timestamps to unix if they don't look like unix
        if np.nanmean(ts) < 1e9:
            now = time.time()
            ts = np.linspace(now - (len(eeg) / recorder.fs), now, len(eeg), endpoint=False)

        recorder.add_eeg_samples(eeg, ts)

        # Update latest_packet + OSC
        for row in eeg:
            row_list = row.tolist()
            if client_osc:
                client_osc.send_message("/muse/eeg", row_list)
            latest_packet["eeg"] = dict(zip(EEG_CHANNELS, row_list))

    # ACC/GYRO
    if "ACCGYRO" in decoded:
        arr, *_ = make_timestamps(decoded["ACCGYRO"], None, None, None, None)
        ag = arr[:, 1:]
        samples_sent["ACCGYRO"] += len(ag)
        for row in ag:
            row_list = row.tolist()
            acc = row_list[:3]
            gyro = row_list[3:6]
            if client_osc:
                client_osc.send_message("/muse/acc",  acc)
                client_osc.send_message("/muse/gyro", gyro)
            latest_packet["acc"]  = {"x": acc[0],  "y": acc[1],  "z": acc[2]}
            latest_packet["gyro"] = {"x": gyro[0], "y": gyro[1], "z": gyro[2]}

    # OPTICS / PPG
    if "OPTICS" in decoded:
        arr, *_ = make_timestamps(decoded["OPTICS"], None, None, None, None)
        opt = arr[:, 1:]
        samples_sent["OPTICS"] += len(opt)
        for row in opt:
            row_list = row.tolist()
            if client_osc:
                client_osc.send_message("/muse/ppg", row_list)
            latest_packet["ppg"] = row_list

    # BATTERY
    if "BATTERY" in decoded:
        arr, *_ = make_timestamps(decoded["BATTERY"], None, None, None, None)
        bat = arr[:, 1:]
        samples_sent["BATTERY"] += len(bat)
        for row in bat:
            val = float(row[0])
            if client_osc:
                client_osc.send_message("/muse/batt", [val])
            latest_packet["battery"] = val

    _record_and_emit()

# -------------------------
# BLE loop
# -------------------------
async def stream_muse_ble(address: str, preset: str = "p1041", verbose: bool = True):
    print(f"[OpenMuse] Connecting to {address} ...")

    samples_sent = {"EEG": 0, "ACCGYRO": 0, "OPTICS": 0, "BATTERY": 0}

    try:
        client_ble = bleak.BleakClient(address, timeout=15.0)
        await client_ble.__aenter__()
    except Exception as e:
        print(f"[OpenMuse] BLE connection failed: {e}")
        return

    print(f"[OpenMuse] Connected. Device: {client_ble.name}")

    queue: asyncio.Queue[bytearray] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    async def decode_worker():
        while not shutdown_event.is_set():
            try:
                data = await queue.get()
                await decode_and_send(data, samples_sent)
            except asyncio.CancelledError:
                return
            except Exception as e:
                print("[OpenMuse][ERROR] decode_worker:", e)

    def handle_data(_, data: bytearray):
        if not shutdown_event.is_set():
            loop.call_soon_threadsafe(queue.put_nowait, data)

    try:
        callbacks = {MuseS.EEG_UUID: handle_data}
        await MuseS.connect_and_initialize(client_ble, preset, callbacks, verbose=verbose)
    except Exception as e:
        print(f"[OpenMuse] Initialization failed: {e}")
        await client_ble.__aexit__(None, None, None)
        return

    print("[OpenMuse] Streaming... (Ctrl+C to stop)")
    worker = asyncio.create_task(decode_worker())

    try:
        while not shutdown_event.is_set():
            if not client_ble.is_connected:
                print("[OpenMuse] BLE client disconnected.")
                break

            await asyncio.sleep(0.05)
    except KeyboardInterrupt:
        shutdown_event.set()
        raise

    print("[OpenMuse] Closing BLE connection...")
    try:
        await client_ble.__aexit__(None, None, None)
    except Exception:
        pass

    worker.cancel()
    try:
        await worker
    except Exception:
        pass

    print("[OpenMuse] Stream stopped.", samples_sent)

def start_ble_background(address: str, preset: str):
    """Entry point for BLE loop in a thread. Auto-reconnect on disconnect."""
    while not shutdown_event.is_set():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(stream_muse_ble(address, preset))
        except Exception as e:
            print("[OpenMuse] BLE loop error:", e)
        finally:
            try:
                loop.close()
            except Exception:
                pass

        if shutdown_event.is_set():
            break

        print("[OpenMuse] Disconnected. Reconnecting in 3 seconds...")
        time.sleep(3)

# -------------------------
# ROUTES: UI
# -------------------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# -------------------------
# ROUTES: Session/Trials
# -------------------------
@app.route("/session/start", methods=["POST"])
def session_start():
    global current_session_id
    payload = request.json or {}
    subject_id = payload.get("subject_id", "subject0")

    montage = {
        "system": "MuseS",
        "channels": EEG_CHANNELS,
        "labels_10_20": {"tp9": "TP9", "af7": "AF7", "af8": "AF8", "tp10": "TP10"},
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    current_session_id = recorder.start_session(DATA_DIR, subject_id, montage, extra_meta={"preset": "p1041"})
    return jsonify({"session_id": current_session_id, "session_dir": recorder.session_dir})

@app.route("/session/end", methods=["POST"])
def session_end():
    recorder.end_session()
    return jsonify({"ok": True})

# -------------------------
# Block Runner
# -------------------------
@app.route("/block/start", methods=["POST"])
def block_start():
    global block_active, block_queue
    if not recorder.session_dir:
        return jsonify({"error": "Start a session first (/session/start)."}), 400

    payload = request.json or {}
    n_trials = int(payload.get("n_trials", 20))
    labels = protocol["classes"]
    block_queue = [random.choice(labels) for _ in range(n_trials)]
    block_active = True
    return jsonify({"ok": True, "n_trials": n_trials})

@app.route("/block/stop", methods=["POST"])
def block_stop():
    global block_active, block_queue
    block_active = False
    block_queue = []
    return jsonify({"ok": True})

@app.route("/block/next", methods=["POST"])
def block_next():
    """
    Server schedules timestamps for trial phases. UI waits total_s.
    """
    global block_active, block_queue, last_prediction

    if not block_active:
        return jsonify({"error": "Block not active."}), 400
    if len(block_queue) == 0:
        block_active = False
        return jsonify({"error": "Block finished."}), 200

    label = block_queue.pop(0)

    # phase scheduling (server-side)
    fixation_s = float(protocol["fixation_s"])
    cue_s = float(protocol["cue_s"])
    imagery_s = float(protocol["imagery_s"])
    relax_s = float(protocol["relax_s"])
    jitter = float(protocol["jitter_s"]) * (random.random() * 2.0 - 1.0)

    now = time.time()
    cue_on = now + fixation_s
    imagery_on = cue_on + cue_s
    imagery_off = imagery_on + imagery_s
    relax_off = imagery_off + relax_s + jitter

    tr = recorder.start_trial(label, cue_on, imagery_on, imagery_off, relax_off)

    # UI display
    display = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}.get(label, "?")

    return jsonify({
        "trial_id": tr.trial_id,
        "label": label,
        "display": display,
        "total_s": fixation_s + cue_s + imagery_s + relax_s + max(0.0, jitter),
    })

# -----------------------------------------
# Export + Train + Predict + Session Handler
# -----------------------------------------
@app.route("/export", methods=["POST"])
def export_trials():
    if not recorder.session_dir:
        return jsonify({"error": "Start a session first."}), 400
    out = recorder.export_trials(protocol["epoch_offset_s"], protocol["epoch_len_s"])
    return jsonify(out)

@app.route("/train", methods=["POST"])
def train():
    payload = request.json or {}
    session_id = payload.get("session_id")

    if session_id:
        session_dir = os.path.join(DATA_DIR, session_id)
    else:
        session_dir = recorder.session_dir

    if not session_dir:
        return jsonify({"error": "No active session. Provide session_id."}), 400
    if not os.path.isdir(session_dir):
        return jsonify({"error": f"Session dir not found: {session_dir}"}), 404

    ok, msg = _validate_session_dir(session_dir)
    if not ok:
        return jsonify({"error": msg}), 400

    out = model.train_from_session_dir(session_dir)
    out["trained_on_session_dir"] = session_dir
    return jsonify(out)

@app.route("/predict_last", methods=["POST"])
def predict_last():
    """
    Predict the most recent valid trial (if exported), OR predict by slicing buffer using last trial timing.
    For v1: slice last trial window directly from recorder.buffer and run predict if model is fit.
    """
    global last_prediction
    if not model.is_fit:
        return jsonify({"status": "not_trained", "message": "Model not trained yet. Run Export Trials, then Train Model."}), 200

    if not recorder.trials:
        return jsonify({"status": "no_trials", "message": "No trials recorded yet."}), 200

    tr = recorder.trials[-1]
    start = tr.imagery_on_ts + protocol["epoch_offset_s"]
    end = start + protocol["epoch_len_s"]

    # slice from buffer
    buf = list(recorder.buffer)
    ts = np.array([b[0] for b in buf], dtype=float)
    x = np.array([b[1] for b in buf], dtype=float)

    idx = np.where((ts >= start) & (ts <= end))[0]
    if len(idx) < 10:
        return jsonify({"error": f"Not enough samples for prediction ({len(idx)})."}), 200

    df = pd.DataFrame(x[idx, :], columns=EEG_CHANNELS)
    df.insert(0, "timestamp", ts[idx])

    pred = model.predict_epoch(df)
    pred["trial_id"] = tr.trial_id
    pred["label_true"] = tr.label
    last_prediction = pred
    return jsonify(pred)

@app.route("/sessions", methods=["GET"])
def sessions_list():
    return jsonify({"sessions": _list_session_dirs(DATA_DIR)})

@app.route("/train_session", methods=["POST"])
def train_session():
    payload = request.json or {}
    session_id = payload.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    session_dir = os.path.join(DATA_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({"error": f"Session not found: {session_id}"}), 404

    ok, msg = _validate_session_dir(session_dir)
    if not ok:
        return jsonify({"error": msg, "session_id": session_id}), 400

    out = model.train_from_session_dir(session_dir)
    out["trained_on_session_id"] = session_id
    return jsonify(out)

# -------------------------
# Simple aggregate endpoints (kept)
# -------------------------
@app.route("/aggregate_eeg", methods=["POST"])
def aggregate_eeg():
    seconds = float(request.json.get("seconds", 10))
    now = time.time()
    cutoff = now - seconds

    window = [p["eeg"] for ts, p in history if ts >= cutoff]
    if not window:
        return jsonify({"error": "No EEG data"}), 404

    avg = {ch: sum(w[ch] for w in window) / len(window) for ch in EEG_CHANNELS}
    return jsonify({"average_eeg": avg})

@app.route("/rank_eeg", methods=["POST"])
def rank_eeg():
    seconds = float(request.json.get("seconds", 10))
    now = time.time()
    cutoff = now - seconds

    window = [p["eeg"] for ts, p in history if ts >= cutoff]
    if not window:
        return jsonify({"error": "No EEG data"}), 404

    sums = {ch: sum(w[ch] for w in window) for ch in EEG_CHANNELS}
    ranking = sorted(sums.items(), key=lambda x: x[1], reverse=True)
    return jsonify({"ranking": ranking})

@socketio.on("connect")
def on_connect():
    print("[OpenMuse] UI connected")

# -------------------------
# ENTRY POINT
# -------------------------
def main():
    global client_osc

    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--preset", default="p1041")
    parser.add_argument("--osc_ip", default=OSC_IP)
    parser.add_argument("--osc_port", type=int, default=OSC_PORT)
    parser.add_argument("--ws_port", type=int, default=WEBSOCKET_PORT)
    args = parser.parse_args()

    client_osc = SimpleUDPClient(args.osc_ip, args.osc_port)

    print(f"[OpenMuse] OSC out to {args.osc_ip}:{args.osc_port}")
    print(f"[OpenMuse] Web server on http://localhost:{args.ws_port}")

    ble_thread = threading.Thread(
        target=start_ble_background,
        args=(args.address, args.preset),
        daemon=True
    )
    ble_thread.start()

    socketio.run(app, host="0.0.0.0", port=args.ws_port)

if __name__ == "__main__":
    main()