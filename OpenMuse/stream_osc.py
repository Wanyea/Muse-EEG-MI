"""
OpenMuse → OSC direct streaming (no LSL)
Fully asynchronous decoder pipeline with graceful shutdown.

Streams:
  /muse/eeg
  /muse/acc
  /muse/gyro
  /muse/ppg
  /muse/batt
"""

import asyncio
import bleak
from pythonosc.udp_client import SimpleUDPClient

from .decode import parse_message, make_timestamps
from .muse import MuseS
from .utils import get_utc_timestamp


# Globals for OSC
OSC_IP = "127.0.0.1"
OSC_PORT = 5000
client = None

# Shutdown event
shutdown_event = asyncio.Event()


async def stream_osc(address: str, preset: str = "p1041", verbose: bool = True):
    """
    Main streaming coroutine.
    """
    global client

    print(f"Connecting to {address} ...")

    # Stats counters (OpenMuse style)
    samples_sent = {
        "EEG": 0,
        "ACCGYRO": 0,
        "OPTICS": 0,
        "BATTERY": 0,
    }

    # BLE connect
    try:
        client_ble = bleak.BleakClient(address, timeout=15.0)
        await client_ble.__aenter__()
    except Exception as e:
        print(f"BLE connection failed: {e}")
        return

    print(f"Connected. Device: {client_ble.name}")

    # ---------- Data Queue + Async Worker ----------

    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    async def decode_worker():
        """Processes BLE packets asynchronously."""
        while not shutdown_event.is_set():
            try:
                data = await queue.get()
                await decode_and_send(data)
            except asyncio.CancelledError:
                return
            except Exception as e:
                print("[ERROR] in decode_worker:", e)

    async def decode_and_send(data: bytearray):
        """Decode using OpenMuse pipeline and send OSC."""
        msg = f"{get_utc_timestamp()}\t{MuseS.EEG_UUID}\t{data.hex()}"
        decoded = parse_message(msg)

        # EEG
        if "EEG" in decoded:
            arr, *_ = make_timestamps(decoded["EEG"], None, None, None, None)
            eeg = arr[:, 1:]
            samples_sent["EEG"] += len(eeg)
            for row in eeg:
                client.send_message("/muse/eeg", row.tolist())

        # ACC/GYRO
        if "ACCGYRO" in decoded:
            arr, *_ = make_timestamps(decoded["ACCGYRO"], None, None, None, None)
            ag = arr[:, 1:]
            samples_sent["ACCGYRO"] += len(ag)
            for row in ag:
                client.send_message("/muse/acc", row[:3].tolist())
                client.send_message("/muse/gyro", row[3:6].tolist())

        # OPTICS
        if "OPTICS" in decoded:
            arr, *_ = make_timestamps(decoded["OPTICS"], None, None, None, None)
            opt = arr[:, 1:]
            samples_sent["OPTICS"] += len(opt)
            for row in opt:
                client.send_message("/muse/ppg", row.tolist())

        # BATTERY
        if "BATTERY" in decoded:
            arr, *_ = make_timestamps(decoded["BATTERY"], None, None, None, None)
            bat = arr[:, 1:]
            samples_sent["BATTERY"] += len(bat)
            for row in bat:
                client.send_message("/muse/batt", [float(row[0])])

    # ---------- BLE Notification Callback ----------
    def handle_data(_, data: bytearray):
        """Synchronous BLE callback (required by Bleak)."""
        if not shutdown_event.is_set():
            loop.call_soon_threadsafe(queue.put_nowait, data)

    # ---------- Initialize Muse ----------
    try:
        callbacks = {MuseS.EEG_UUID: handle_data}

        await MuseS.connect_and_initialize(
            client_ble,
            preset,
            callbacks,
            verbose=verbose
        )
    except Exception as e:
        print(f"Initialization failed: {e}")
        await client_ble.__aexit__(None, None, None)
        return

    print("Streaming... (Ctrl+C to stop)")

    # Start worker task
    worker = asyncio.create_task(decode_worker())

    # ---------- Main Loop ----------
    try:
        while not shutdown_event.is_set():
            if not client_ble.is_connected:
                print("Client disconnected.")
                shutdown_event.set()
                break
            await asyncio.sleep(0.05)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
        shutdown_event.set()

    # ---------- Cleanup ----------
    print("Stopping notifications...")
    try:
        await client_ble.stop_notify(MuseS.EEG_UUID)
    except:
        pass

    print("Closing BLE connection...")
    try:
        await client_ble.__aexit__(None, None, None)
    except:
        pass

    # Stop worker
    worker.cancel()
    try:
        await worker
    except:
        pass

    # ---------- Final Stats Summary ----------
    print(
        "Stream stopped. "
        f"EEG: {samples_sent['EEG']} samples, "
        f"ACCGYRO: {samples_sent['ACCGYRO']} samples, "
        f"OPTICS: {samples_sent['OPTICS']} samples, "
        f"BATTERY: {samples_sent['BATTERY']} samples"
    )


# ---------- CLI Entrypoint ----------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--preset", default="p1041")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    global client
    client = SimpleUDPClient(args.ip, args.port)

    try:
        asyncio.run(stream_osc(args.address, args.preset))
    except KeyboardInterrupt:
        print("Exit requested.")


if __name__ == "__main__":
    main()
