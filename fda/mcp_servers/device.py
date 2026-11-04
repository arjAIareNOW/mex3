"""A read-only health and security check of the machine FDA runs on (the Pi; a smaller check on Windows/macOS).

Looks at: services open to the network, waiting updates, SSH settings, world-writable system files, disk, memory,
temperature and throttling. It only READS: every command is listed in the report, nothing is changed. Fixes are
suggested as commands for the owner to run. Each run writes reports/audit-<time>.md and says what changed since
the previous report.
"""
from __future__ import annotations

import json
import logging
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger("fda.mcp.device")
SEV = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
# ports that are fine to have open on a home device (FDA's own display, ssh is checked separately)
_EXPECTED = {22, 8080, 8100, 8101, 8102, 8103, 5353}


def _run(cmd: list[str], timeout: float = 25) -> tuple[str, str]:
    """(output, the command as text). Never raises."""
    shown = " ".join(cmd)
    if not shutil.which(cmd[0]):
        return "", shown + "  (not available)"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        return r.stdout, shown
    except (OSError, subprocess.TimeoutExpired) as e:
        return "", f"{shown}  ({type(e).__name__})"


class Device:
    def __init__(self, reports: Path):
        self.reports = reports
        self.reports.mkdir(parents=True, exist_ok=True)

    def check(self) -> dict:
        import psutil
        findings: list[dict] = []
        ran: list[str] = []

        def add(sev, title, detail, fix=""):
            findings.append({"severity": sev, "title": title, "detail": detail, "fix": fix})

        system = platform.system()
        # ---- open to the network
        try:
            listen = {}
            for c in psutil.net_connections(kind="inet"):
                if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.ip in ("0.0.0.0", "::", ""):
                    name = ""
                    try:
                        name = psutil.Process(c.pid).name() if c.pid else ""
                    except (psutil.Error, OSError):
                        pass
                    listen[c.laddr.port] = name
            ran.append("psutil.net_connections (listening sockets)")
            risky = {6379: "Redis", 27017: "MongoDB", 3306: "MySQL", 5432: "PostgreSQL", 23: "Telnet", 21: "FTP",
                     5900: "VNC", 11211: "Memcached", 9200: "Elasticsearch", 2375: "Docker API"}
            for port, name in sorted(listen.items()):
                if port in risky:
                    add("critical" if port in (6379, 2375, 23, 11211) else "high",
                        f"{risky[port]} is open to the whole network (port {port})",
                        f"{name or risky[port]} listens on all interfaces.",
                        f"Bind it to 127.0.0.1 or block port {port} in the firewall.")
            others = [f"{p} ({n})" if n else str(p) for p, n in sorted(listen.items())
                      if p not in risky and p not in _EXPECTED and p < 49152]
            if others:
                add("low", f"{len(others)} other services are reachable from the network",
                    "Ports: " + ", ".join(others[:12]), "Close the ones you do not use.")
        except Exception as e:  # noqa: BLE001
            ran.append(f"listening sockets: could not read ({e})")

        # ---- disk, memory, temperature
        disk = psutil.disk_usage(str(Path.home().anchor or "/"))
        ran.append("psutil.disk_usage /")
        if disk.percent >= 95:
            add("high", f"Disk is {disk.percent:.0f}% full", f"{disk.free / 1e9:.1f} GB free.", "Delete old files or logs.")
        elif disk.percent >= 85:
            add("medium", f"Disk is {disk.percent:.0f}% full", f"{disk.free / 1e9:.1f} GB free.", "Free some space soon.")
        mem = psutil.virtual_memory()
        ran.append("psutil.virtual_memory")
        if mem.percent >= 90:
            add("medium", f"Memory is {mem.percent:.0f}% used", f"{mem.available / 1e9:.1f} GB available.",
                "Close programs you do not need.")
        temp = None
        try:
            for vals in (psutil.sensors_temperatures() or {}).values() if hasattr(psutil, "sensors_temperatures") else []:
                for v in vals:
                    temp = max(temp or 0, v.current)
        except Exception:  # noqa: BLE001
            pass
        out, cmd = _run(["vcgencmd", "get_throttled"])
        if out:
            ran.append(cmd)
            m = re.search(r"0x([0-9a-f]+)", out)
            flags = int(m.group(1), 16) if m else 0
            if flags & 0x1:
                add("high", "The Pi is short of power right now (under-voltage)", out.strip(),
                    "Use the official 5V 5A USB-C power supply.")
            elif flags & 0x10000:
                add("medium", "The Pi had too little power since it started", out.strip(), "Use a stronger power supply.")
            if flags & 0x4 or flags & 0x40000:
                add("medium", "The Pi slowed itself down (throttled) because of heat or power", out.strip(),
                    "Add a fan or the active cooler.")
        if temp is not None and temp >= 80:
            add("high" if temp >= 85 else "medium", f"CPU is hot: {temp:.0f} °C", "The Pi slows down above 80-85 °C.",
                "Add a fan or the active cooler.")

        # ---- Linux (the Pi): updates, SSH, world-writable files
        if system == "Linux":
            out, cmd = _run(["apt", "list", "--upgradable"], timeout=40)
            ran.append(cmd)
            pk = [ln.split("/")[0] for ln in out.splitlines() if "/" in ln and "upgradable" in ln]
            if pk:
                sec = [p for p in pk if re.search(r"openssl|openssh|libssl|linux-image|raspberrypi-kernel|sudo|curl", p)]
                add("high" if sec else "medium", f"{len(pk)} updates are waiting" + (f", including {', '.join(sec[:3])}"
                                                                                   if sec else ""),
                    ", ".join(pk[:15]), "sudo apt update && sudo apt full-upgrade")
            sshd = Path("/etc/ssh/sshd_config")
            if sshd.exists():
                ran.append("read /etc/ssh/sshd_config")
                conf = sshd.read_text(errors="replace")
                for f in Path("/etc/ssh/sshd_config.d").glob("*.conf") if Path("/etc/ssh/sshd_config.d").exists() else []:
                    conf += "\n" + f.read_text(errors="replace")

                def opt(name):
                    m = re.findall(rf"^\s*{name}\s+(\S+)", conf, re.M | re.I)
                    return m[0].lower() if m else None          # sshd uses the first value it finds
                if opt("PermitRootLogin") in ("yes",):
                    add("high", "SSH lets root log in", "PermitRootLogin yes", "Set PermitRootLogin no and restart ssh.")
                if opt("PasswordAuthentication") in (None, "yes") and 22 in [c.laddr.port for c in psutil.net_connections(kind="inet") if c.status == psutil.CONN_LISTEN]:
                    add("medium", "SSH accepts passwords", "PasswordAuthentication is on (the default).",
                        "Use SSH keys, then set PasswordAuthentication no.")
                if opt("PermitEmptyPasswords") == "yes":
                    add("critical", "SSH allows empty passwords", "PermitEmptyPasswords yes", "Set it to no.")
            out, cmd = _run(["find", "/etc", "/usr/local/bin", "/usr/local/sbin", "-xdev", "-type", "f", "-perm",
                             "-0002"], timeout=30)
            ran.append(cmd)
            ww = [x for x in out.splitlines() if x.strip()]
            if ww:
                add("high", f"{len(ww)} system files can be changed by any user", ", ".join(ww[:8]),
                    "sudo chmod o-w <file>")
            out, cmd = _run(["systemctl", "is-active", "unattended-upgrades"])
            ran.append(cmd)
            if out.strip() not in ("active",):
                add("low", "Automatic security updates are off", f"unattended-upgrades: {out.strip() or 'not installed'}",
                    "sudo apt install unattended-upgrades")
        elif system == "Windows":
            out, cmd = _run(["netsh", "advfirewall", "show", "allprofiles", "state"])
            ran.append(cmd)
            if re.search(r"State\s+OFF", out, re.I):
                add("high", "Windows Firewall is off for a network profile", out.strip()[:200],
                    "Turn the firewall on in Windows Security.")

        findings.sort(key=lambda f: SEV[f["severity"]])
        return self._report(findings, ran, disk, mem, temp, system)

    def _report(self, findings, ran, disk, mem, temp, system) -> dict:
        now = time.strftime("%Y-%m-%d %H:%M")
        prev_file = max(self.reports.glob("audit-*.json"), default=None, key=lambda p: p.stat().st_mtime)
        prev = json.loads(prev_file.read_text(encoding="utf-8")) if prev_file else None
        titles = {f["title"] for f in findings}
        new = [t for t in titles if prev and t not in {f["title"] for f in prev}]
        fixed = [f["title"] for f in prev or [] if f["title"] not in titles]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        counts = {s: sum(f["severity"] == s for f in findings) for s in SEV}
        md = [f"# Device check, {now} ({system})", "",
              f"Disk {disk.percent:.0f}% used, memory {mem.percent:.0f}% used"
              + (f", CPU {temp:.0f} °C" if temp is not None else "") + ".", "",
              "**Findings:** " + ", ".join(f"{n} {s}" for s, n in counts.items() if n) if findings else "**No problems found.**",
              ""]
        if prev is not None:
            md += [f"**Since the last check:** {len(new)} new, {len(fixed)} fixed.", ""]
        for i, f in enumerate(findings, 1):
            md += [f"## {i}. [{f['severity'].upper()}] {f['title']}", f["detail"], ""]
            if f["fix"]:
                md += [f"Fix: `{f['fix']}`", ""]
        md += ["## What was checked (read only, nothing was changed)"] + [f"- `{r}`" for r in ran]
        path = self.reports / f"audit-{stamp}.md"
        path.write_text("\n".join(md) + "\n", encoding="utf-8")
        (self.reports / f"audit-{stamp}.json").write_text(json.dumps(findings), encoding="utf-8")
        top = findings[:3]
        summary = (f"{len(findings)} findings: " + ", ".join(f"{n} {s}" for s, n in counts.items() if n)
                   if findings else "No problems found.")
        return {"summary": summary, "top": [{"severity": f["severity"], "title": f["title"], "fix": f["fix"]} for f in top],
                "new_since_last": new, "fixed_since_last": fixed, "report": str(path.name),
                "note": "Read-only check: nothing was changed. Fixes are suggestions for the owner to run."}
