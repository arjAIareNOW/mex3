#!/usr/bin/env bash
# School VPN on the Pi, for the DGX3 voice: connects at boot and reconnects by itself.
#   bash deploy/pi/vpn.sh ~/SHARCHPC.ovpn
# Asks for the VPN username and password once and keeps them in a file only root can read.
# Elkit does not need the VPN to run: without it the voice is Piper, and it switches to DGX3 when the VPN is up.
set -euo pipefail
OVPN="${1:?usage: bash deploy/pi/vpn.sh path/to/SHARCHPC.ovpn}"
[ -f "$OVPN" ] || { echo "no such file: $OVPN"; exit 1; }
NAME=school
DIR=/etc/openvpn/client
DGX3=http://10.153.60.233:8001

sudo apt-get install -y openvpn curl
sudo install -d -m 700 "$DIR"
# the profile as it is, except the login comes from the root-only file (no prompt at boot)
sed -E "s#^auth-user-pass.*#auth-user-pass $DIR/$NAME.auth#; s/\r$//" "$OVPN" | sudo tee "$DIR/$NAME.conf" >/dev/null
sudo chmod 600 "$DIR/$NAME.conf"
if ! sudo test -s "$DIR/$NAME.auth"; then
    read -rp "School VPN username: " U
    read -rsp "School VPN password: " P; echo
    printf '%s\n%s\n' "$U" "$P" | sudo tee "$DIR/$NAME.auth" >/dev/null
    unset U P
    sudo chmod 600 "$DIR/$NAME.auth"
fi
sudo systemctl enable "openvpn-client@$NAME"
sudo systemctl restart "openvpn-client@$NAME"

echo "connecting ..."
for _ in $(seq 1 20); do
    if curl -s -m 3 -o /dev/null "$DGX3"; then
        echo "VPN up: the DGX3 answers. It connects by itself at every boot."
        exit 0
    fi
    sleep 2
done
echo "The DGX3 does not answer yet. What OpenVPN says:"
sudo journalctl -u "openvpn-client@$NAME" -n 25 --no-pager
echo "Wrong login? Delete it and run this again:  sudo rm $DIR/$NAME.auth"
exit 1
