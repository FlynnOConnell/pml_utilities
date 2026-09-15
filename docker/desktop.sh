#!/bin/sh
# X server + XFCE, exposed to the browser through noVNC on port 6080
set -e
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1
Xtigervnc :1 -geometry "${GEOMETRY:-1600x900}" -depth 24 -rfbport 5901 \
    -SecurityTypes None -localhost yes >/tmp/xvnc.log 2>&1 &
sleep 1
dbus-launch --exit-with-session startxfce4 >/tmp/xfce.log 2>&1 &
exec websockify --web /usr/share/novnc 6080 localhost:5901
