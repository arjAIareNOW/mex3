"""fdbench --reset-cmd for FDA: a fresh session between trials.

    python tools/fda_reset.py                            Elkit on this machine: touches runs/.fda_reset
    python tools/fda_reset.py http://<pi-ip>:8792/reset  Elkit on the Pi: asks it over HTTP (assistant_entry.py)
"""
import sys
import time
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
    import urllib.request
    try:
        urllib.request.urlopen(sys.argv[1], timeout=5).read()       # returns once the fresh session is done
    except OSError as e:
        print(f"fda_reset: {sys.argv[1]} not reachable ({e})", file=sys.stderr)
        sys.exit(1)
else:
    p = Path(__file__).resolve().parents[1] / "runs" / ".fda_reset"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(time.time()), encoding="utf-8")
    time.sleep(0.3)          # the agent checks every 0.1 s: the reset is done before the next clip plays
