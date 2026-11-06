# fd-benchmark files for FDA

`fd-benchmark/` is its own repository (not part of this one). After cloning it into `fda/fd-benchmark`, copy:

    cp scripts/fdbench/assistant_entry.py fd-benchmark/assistant_entry.py
    cp scripts/fdbench/fda_reset.py      fd-benchmark/tools/fda_reset.py

- `assistant_entry.py` runs the Pipecat agent for the benchmark, never saves room data (test clips set alarms),
  and starts a fresh session when `runs/.fda_reset` changes.
- `fda_reset.py` is the `--reset-cmd`: each trial is its own conversation, as in the papers.

Laptop (local):

    .venv\Scripts\python.exe -m fdbench run --student fda-pc --rounds 3 --connect direct --where local `
      --judge-provider custom --judge-url http://127.0.0.1:8092/v1 --judge-model qwen2.5-3b --no-utmos --yes `
      --recordings k --reset-cmd ".venv\Scripts\python.exe tools\fda_reset.py"

(free local judge first: `python -m fda judge`)

With the Pi (`--where remote`), the reset runs on the laptop, but the file it touches must change on the Pi:
the Pi watches its own `fd-benchmark/runs/.fda_reset`. Until that is wired up, the Pi run has no per-trial reset.
