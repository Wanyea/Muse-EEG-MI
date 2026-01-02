from .backends import BleakBackend


def find_devices(timeout=10, verbose=True):
    """Scan for Muse devices via Bluetooth Low Energy (BLE). Call 'find' from the terminal."""
    backend = BleakBackend()
    if verbose:
        print(f"Searching for Muses (max. {timeout} seconds)...")
    devices = backend.scan(timeout=timeout)
    muses = []
    for d in devices:
        name = d.get("name")
        try:
            if isinstance(name, str) and "muse" in name.lower():
                muses.append(d)
        except Exception:
            continue

    if verbose:
        if muses:
            for m in muses:
                print(f'Found device {m["name"]}, MAC Address {m["address"]}')
        else:
            print("No Muses found. Ensure the device is on and Bluetooth is enabled.")

    return muses
