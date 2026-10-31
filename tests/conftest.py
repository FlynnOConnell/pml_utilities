"""
Shared pytest fixtures for mbo_utilities tests.

Test data location: ~/.mbo/tests/lbm/mbo_utilities/

Usage:
    pytest tests/ -v                    # Run all tests
    pytest tests/test_roundtrip.py -v   # Run specific test file
    KEEP_TEST_OUTPUT=1 pytest tests/    # Keep output files for inspection
"""

import os
import shutil
from pathlib import Path

# Force the offscreen rendercanvas backend for the whole test process.
# rendercanvas.auto picks its backend the moment it is imported (any
# fastplotlib import at collection time pulls it in), so the env var must
# be set here — before any test module imports fastplotlib — for the
# offscreen viewer tests (tests/test_ndviewer.py) to work. No test opens
# a real window; `setdefault` still lets a caller override explicitly.
os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")

import mbo_utilities as mbo
import numpy as np
import pytest

TEST_DATA_ROOT = Path.home() / ".mbo" / "tests" / "lbm" / "mbo_utilities"
TEST_INPUT_TIFF = TEST_DATA_ROOT / "test_input.tif"
OUTPUT_DIR = TEST_DATA_ROOT / "test_outputs"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow tests (real suite2p runs)")


@pytest.fixture(autouse=True)
def options(monkeypatch):
    """The persisted GUI options (``preferences._get_options``) as an
    in-memory dict: tests start from the defaults, never read the user's
    ``~/.mbo`` and never write it. Set a key to change one for a test.
    """
    from mbo_utilities import preferences

    store = {}
    monkeypatch.setattr(preferences, "_get_options", lambda: dict(store))
    monkeypatch.setattr(preferences, "_set_option", store.__setitem__)
    return store


@pytest.fixture(scope="session")
def test_data_root():
    """Root directory for test data."""
    if not TEST_DATA_ROOT.exists():
        pytest.skip(f"Test data directory not found: {TEST_DATA_ROOT}")
    return TEST_DATA_ROOT


@pytest.fixture(scope="session")
def source_tiff_path(test_data_root):
    """Path to the source test TIFF file."""
    if not TEST_INPUT_TIFF.exists():
        pytest.skip(f"Test input not found: {TEST_INPUT_TIFF}")
    return TEST_INPUT_TIFF


@pytest.fixture(scope="session")
def source_array(source_tiff_path):
    """
    Load the source test array once per session.

    Returns the MboRawArray with phase correction disabled for consistent baseline.
    """
    arr = mbo.imread(source_tiff_path, fix_phase=False)
    return arr


@pytest.fixture
def output_dir(request, test_data_root):
    """
    Unique output directory for each test.

    Creates: ~/.mbo/tests/lbm/mbo_utilities/test_outputs/<test_name>/
    Cleaned up after test unless KEEP_TEST_OUTPUT=1
    """
    import os

    test_name = request.node.name
    out_dir = OUTPUT_DIR / test_name

    # Clean previous run
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    yield out_dir

    # Cleanup unless explicitly kept
    if os.environ.get("KEEP_TEST_OUTPUT", "0") != "1":
        shutil.rmtree(out_dir, ignore_errors=True)


@pytest.fixture
def synthetic_3d_data():
    """
    Synthetic 3D test data (T, Y, X) - single z-plane.

    Deterministic, reproducible, fast to generate.
    """
    rng = np.random.RandomState(42)
    shape = (20, 128, 128)

    # Base noise
    data = rng.normal(500, 100, shape).astype(np.int16)

    # Add some structure (simulated cells)
    for i in range(5):
        cy, cx = rng.randint(20, 108, size=2)
        yy, xx = np.ogrid[:128, :128]
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < 100

        # Temporal signal
        for t in range(20):
            signal = 200 + 100 * np.sin(2 * np.pi * t / 10 + i)
            data[t][mask] += int(signal)

    return data.clip(0, 4095).astype(np.int16)


@pytest.fixture
def synthetic_4d_data():
    """
    Synthetic 4D test data (T, Z, Y, X) - multi z-plane.

    Deterministic, reproducible, fast to generate.
    """
    rng = np.random.RandomState(42)
    shape = (10, 3, 64, 64)  # 10 frames, 3 planes, 64x64

    # Base noise
    data = rng.normal(500, 100, shape).astype(np.int16)

    # Add structure per plane
    for z in range(3):
        for i in range(3):
            cy, cx = rng.randint(10, 54, size=2)
            yy, xx = np.ogrid[:64, :64]
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < 50

            for t in range(10):
                # Z-dependent signal
                z_factor = 1.0 - 0.2 * abs(z - 1)  # Peak at middle plane
                signal = (200 + 100 * np.sin(2 * np.pi * t / 5 + i)) * z_factor
                data[t, z][mask] += int(signal)

    return data.clip(0, 4095).astype(np.int16)
