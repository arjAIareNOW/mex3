#!/usr/bin/env bash
# Raspberry Pi 5 (Raspberry Pi OS Bookworm, desktop): install FDA, start it on boot, show the display full screen.
#   cd ~/fda && bash deploy/pi/install.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
APP="$(pwd)"
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libportaudio2 chromium-browser unclutter nodejs npm git
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# app checker: a real DOM (jsdom) for testing the apps and games FDA builds
(cd fda/mcp_servers && npm install --no-audit --no-fund)
[ -f .env ] || { printf "# FDA keys (never share this file). Leave a line empty if you do not use that service.\nQWEN_API_KEY=\nGROQ_API_KEY=\nSAMBANOVA_API_KEY=\nGEMINI_API_KEY=\n" > .env; chmod 600 .env; echo ">> edit $APP/.env and paste your API keys"; }
[ -f fda.toml ] || cp fda.example.toml fda.toml
# no browser from FDA itself on the Pi: the kiosk below shows the display
grep -q '^open_browser' fda.toml || echo 'open_browser = false' >> fda.toml
# offline models (listening, voice, local LLM) for the engines in fda.toml, built for this Pi
python -m fda download

# FDA as a user service: starts at boot, restarts if it stops
mkdir -p ~/.config/systemd/user
sed "s#@APP@#$APP#g" deploy/pi/fda.service > ~/.config/systemd/user/fda.service
systemctl --user daemon-reload
systemctl --user enable fda.service
sudo loginctl enable-linger "$USER"

# Display: Chromium kiosk on the 7" screen after the desktop starts
mkdir -p ~/.config/autostart
cp deploy/pi/fda-kiosk.desktop ~/.config/autostart/
echo "Done. Reboot, or start now: systemctl --user start fda && journalctl --user -u fda -f"
