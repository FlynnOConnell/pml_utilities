"""Apps a package adds through the ``mbo_utilities.apps`` entry-point group."""

from __future__ import annotations

import pytest

apps_module = pytest.importorskip("mbo_utilities.gui.app.apps")
from mbo_utilities.gui.app._app import App  # noqa: E402


class DemoApp(App):
    id = "demo"
    title = "Demo"


class FakeEntryPoint:
    """An installed entry point: its name, and what loading it gives."""

    def __init__(self, name, target):
        self.name = name
        self.target = target

    def load(self):
        if isinstance(self.target, Exception):
            raise self.target
        return self.target


def two_apps():
    return [DemoApp(), DemoApp()]


def test_an_entry_point_names_a_class_or_a_factory(monkeypatch):
    points = [FakeEntryPoint("one", DemoApp), FakeEntryPoint("two", two_apps)]
    monkeypatch.setattr(apps_module, "entry_points", lambda group: points)
    made = apps_module.plugin_apps()
    assert [type(app) for app in made] == [DemoApp, DemoApp, DemoApp]


def test_a_broken_entry_point_is_skipped(monkeypatch):
    points = [
        FakeEntryPoint("bad", ImportError("no such module")),
        FakeEntryPoint("ok", DemoApp),
    ]
    monkeypatch.setattr(apps_module, "entry_points", lambda group: points)
    assert [app.id for app in apps_module.plugin_apps()] == ["demo"]


def test_the_group_is_the_documented_one():
    assert apps_module.ENTRY_POINT_GROUP == "mbo_utilities.apps"
