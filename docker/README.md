# ubuntu test containers

Three images from one Dockerfile. All bind-mount the repo at `/app`, so edits
on Windows are live inside the container. The venv lives at `/opt/venv`,
outside the mount, so it survives the bind.

- `headless`: GL/EGL present, no X libraries. A Qt window attempt fails with
  the xcb plugin error (`libxcb-cursor0 is needed to load the Qt xcb platform plugin`).
- `x11`: adds the xcb libs, `xvfb` and software Vulkan, for checking the GUI
  path works when a display exists.
- `desktop`: `x11` plus an XFCE desktop in the browser at
  <http://localhost:6080/vnc.html?autoconnect=1&resize=remote>.

```
cd docker
docker compose build headless          # first build is slow, later ones hit the uv cache
docker compose run --rm headless       # bash inside the container
docker compose run --rm headless pytest tests/test_mesc.py -q
docker compose run --rm headless mbo --help

# reproduce the error
docker compose run --rm headless python -c "from PyQt6.QtWidgets import QApplication; QApplication([])"
# what a headless fix should produce
docker compose run --rm -e QT_QPA_PLATFORM=offscreen headless python -c "from PyQt6.QtWidgets import QApplication; QApplication([]); print('ok')"

# GUI path under a virtual X server
docker compose build x11
docker compose run --rm x11 bash -c 'xvfb-run -a python -c "from PyQt6.QtWidgets import QApplication; QApplication([]); print(\"ok\")"'

# interactive desktop in the browser, then open a terminal there and run `mbo`
docker compose build desktop
docker compose up desktop
```

In Git Bash set `MSYS_NO_PATHCONV=1` before passing Linux paths to `docker run`.
