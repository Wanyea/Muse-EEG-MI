import asyncio
import os
import time
from typing import Iterable, Optional

import bleak

from .backends import _run
from .muse import MuseS
from .utils import get_utc_timestamp


async def _record_async(
    address: str,
    duration: float,
    outfile: str,
    preset: str = "p1041",
    subscribe_chars: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> None:
    if subscribe_chars is None:
        subscribe_chars = list(MuseS.DATA_CHARACTERISTICS)

    notified = 0
    stream_started = asyncio.Event()

    # Ensure output directory exists
    try:
        outdir = os.path.dirname(os.path.abspath(outfile))
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
    except Exception:
        pass

    # Open output file in text mode and append
    f = open(outfile, "a", encoding="utf-8")

    # Raw recording only
    def _callback(uuid: str):
        def inner(_, data: bytearray):
            nonlocal notified
            notified += 1
            # Log timestamp, char UUID, and hex payload
            ts = get_utc_timestamp()
            line = f"{ts}\t{uuid}\t{data.hex()}\n"
            f.write(line)
            # Raw recording only; no decoding or viewing
            if not stream_started.is_set():
                stream_started.set()

        return inner

    if verbose:
        print(f"Connecting to {address} ...")

    try:
        async with bleak.BleakClient(address, timeout=15.0) as client:
            if verbose:
                print("Connected. Subscribing and configuring ...")

            # Build callbacks dict for all data characteristics
            data_callbacks = {uuid: _callback(uuid) for uuid in subscribe_chars}

            # Use shared connection routine
            await MuseS.connect_and_initialize(client, preset, data_callbacks, verbose)

            # Streaming is now active (callbacks are registered and device is configured)
            # Data will start flowing asynchronously

            if verbose:
                print(f"Recording for {duration} seconds to {outfile} ...")

            start = time.time()
            try:
                while time.time() - start < duration:
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                pass
            finally:
                # Try to stop streaming gracefully
                await MuseS.stop_streaming(client, verbose)

            # Unsubscribe
            for cuuid in subscribe_chars:
                try:
                    await client.stop_notify(cuuid)
                except Exception:
                    pass
            # Also try to stop control notify
            try:
                await client.stop_notify(MuseS.CONTROL_UUID)
            except Exception:
                pass

    finally:
        try:
            f.flush()
            f.close()
        except Exception:
            pass
    if verbose:
        print(f"Done. Wrote {notified} notifications to {outfile}.")


def record(
    address: str,
    duration: float = 30.0,
    outfile: str = "muse_record.txt",
    preset: str = "p1041",
    verbose: bool = True,
) -> None:
    """
    Connect to a Muse device, stream notifications, and append raw packets to a text file.

    Each line written: ISO8601 UTC timestamp, characteristic UUID, hex payload.

    Parameters
    - address: Device MAC address (Windows) or identifier (platform-dependent).
    - duration: Recording duration in seconds.
    - outfile: Path to output text file.
    - preset: Preset string to send (e.g., "p1031" or "p21").
    - verbose: Print progress messages.
    """
    if not address:
        raise ValueError("address must be a non-empty string")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if not isinstance(outfile, str) or not outfile:
        raise ValueError("outfile must be a non-empty path string")

    chars = list(MuseS.DATA_CHARACTERISTICS)
    return _run(
        _record_async(
            address=address,
            duration=duration,
            outfile=outfile,
            preset=preset,
            subscribe_chars=chars,
            verbose=verbose,
        )
    )
