#!/usr/bin/env bash
# Elkit's display, full screen: wait for the agent's page, then open it in Chromium (whichever name this Pi OS uses).
command -v unclutter >/dev/null && { unclutter -idle 1 & } || true
until curl -s localhost:8765 >/dev/null; do sleep 1; done
CHROME="$(command -v chromium || command -v chromium-browser)"
exec "$CHROME" --kiosk --autoplay-policy=no-user-gesture-required --noerrdialogs --disable-infobars \
    --check-for-update-interval=31536000 http://localhost:8765
