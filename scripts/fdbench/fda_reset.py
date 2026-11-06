"""fdbench --reset-cmd for FDA: a fresh session between trials (assistant_entry.py watches this file)."""
import time
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "runs" / ".fda_reset"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(str(time.time()), encoding="utf-8")
time.sleep(0.3)          # the agent checks every 0.1 s: the reset is done before the next clip plays
