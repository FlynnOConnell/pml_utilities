# ubuntu image for testing pml_utilities headless. it has the GL/EGL libs a
# cluster node ships with but no X libraries, so a Qt window attempt fails
# exactly as it does on a bare compute node. build with --build-arg WITH_X=1
# to add the xcb libs, xvfb and software vulkan. the desktop stage adds an
# XFCE desktop served to the browser over noVNC.
FROM ubuntu:24.04 AS base
ARG WITH_X=0

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed \
    GIT_LFS_SKIP_SMUDGE=1 \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        ca-certificates curl git ffmpeg libgl1 libegl1 libglib2.0-0 \
        libfontconfig1 libdbus-1-3 \
    && if [ "$WITH_X" = "1" ]; then apt-get install -y --no-install-recommends \
        xvfb xauth libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 \
        libxcb-xkb1 libxkbcommon-x11-0 libvulkan1 mesa-vulkan-drivers; fi \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 \
    && uv sync --frozen --no-default-groups --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups \
    && uv pip install pytest

CMD ["bash"]

FROM base AS desktop
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        tigervnc-standalone-server novnc websockify dbus-x11 \
        xfce4-session xfce4-panel xfwm4 xfdesktop4 xfce4-terminal \
        xfce4-settings thunar adwaita-icon-theme fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
COPY docker/desktop.sh /usr/local/bin/desktop.sh
ENV DISPLAY=:1
EXPOSE 6080
CMD ["/usr/local/bin/desktop.sh"]
