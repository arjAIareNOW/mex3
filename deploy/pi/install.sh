#!/usr/bin/env bash
# Raspberry Pi 5 (Raspberry Pi OS Bookworm, desktop): install FDA, start it on boot, show the display full screen.
#   cd ~/fda && bash deploy/pi/install.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
APP="$(pwd)"
sudo apt-get update
# Chromium is "chromium" on newer Pi OS (Trixie), "chromium-browser" on older Bookworm images
CHROMIUM=chromium; apt-cache show chromium >/dev/null 2>&1 || CHROMIUM=chromium-browser
sudo apt-get install -y python3-venv python3-dev libportaudio2 "$CHROMIUM" curl nodejs npm git
sudo apt-get install -y unclutter || true       # hides the mouse pointer (optional)
# sound through PipeWire (its default mic and speaker): pipewire-alsa gives ALSA programs a "default" that is PipeWire
sudo apt-get install -y pipewire-alsa
# an older project's ALSA config can point "default" somewhere else (a Loopback card): keep it, but out of the way
if [ -f /etc/asound.conf ]; then sudo mv /etc/asound.conf /etc/asound.conf.before-elkit; echo ">> moved /etc/asound.conf aside"; fi
if [ -f ~/.asoundrc ]; then mv ~/.asoundrc ~/.asoundrc.before-elkit; echo ">> moved ~/.asoundrc aside"; fi
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# app checker: a real DOM (jsdom) for testing the apps and games FDA builds
(cd fda/mcp_servers && npm install --no-audit --no-fund)
[ -f .env ] || { printf "# FDA keys (never share this file). Leave a line empty if you do not use that service.\nGROQ_API_KEY=\nQWEN_API_KEY=\n" > .env; chmod 600 .env; echo ">> edit $APP/.env and paste your API keys"; }
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
chmod +x deploy/pi/kiosk.sh
LABWC=~/.config/labwc/autostart
if [ -f "$LABWC" ]; then
    # this Pi has its own labwc autostart (it replaces the desktop's, so ~/.config/autostart may never run):
    # keep a copy, turn off any other kiosk browser in it (an older project's), and start Elkit's display there
    [ -f "$LABWC.before-elkit" ] || cp "$LABWC" "$LABWC.before-elkit"
    sed -i -E '/elkit-kiosk/! s@^([^#].*(chromium|kiosk|firefox).*)$@# (off for Elkit) \1@' "$LABWC"
    grep -q elkit-kiosk "$LABWC" || echo "$APP/deploy/pi/kiosk.sh &   # elkit-kiosk" >> "$LABWC"
    echo ">> kiosk added to $LABWC (old version kept as $LABWC.before-elkit)"
else
    mkdir -p ~/.config/autostart
    sed "s#@APP@#$APP#g" deploy/pi/fda-kiosk.desktop > ~/.config/autostart/fda-kiosk.desktop
fi
# anything else already on Elkit's ports (an older project's server) would keep Elkit from starting
if command -v ss >/dev/null && ss -ltnp 2>/dev/null | grep -qE ':(8765|8101|8102|8103)[[:space:]]'; then
    echo ">> WARNING: something already listens on Elkit's ports (8765, 8101-8103):"
    sudo ss -ltnp | grep -E ':(8765|8101|8102|8103)[[:space:]]' || true
    echo "   stop it first (e.g. sudo systemctl disable --now <its service>)"
fi
echo "Done. Reboot, or start now: systemctl --user start fda && journalctl --user -u fda -f"
