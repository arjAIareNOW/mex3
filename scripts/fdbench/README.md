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

- `assistant_entry.py` also answers `GET http://<pi>:8792/reset` (port: env `FDA_RESET_PORT`) and logs a
  `device:` line (RPi model, RAM, OS) for the latency report.

With the Pi (`--where remote`): Elkit runs on the Pi, fdbench on the laptop, and the reset goes over HTTP.

    # Pi (fd-benchmark/.env has FDBENCH_URL=ws://LAPTOP-IP:8791, see fd-benchmark docs/GUIDE.md)
    ~/ME3/.venv/bin/python launch_assistant.py --url ws://LAPTOP-IP:8791

    # laptop
    .venv\Scripts\python.exe -m fdbench run --student fda-pc --rounds 3 --connect direct --where remote `
      --judge-provider custom --judge-url http://127.0.0.1:8092/v1 --judge-model qwen2.5-3b --no-utmos --yes `
      --recordings k --reset-cmd ".venv\Scripts\python.exe tools\fda_reset.py http://PI-IP:8792/reset"

Then copy Elkit's log of that run from the Pi to the laptop and run
`python scripts/latency_report.py fd-benchmark/runs/fda-pc/<run> --log pi-assistant.log --cooling "..." --network "..."`,
then `python scripts/me_slides.py` (the deck fills Mean / p95 only from a Raspberry Pi report).
