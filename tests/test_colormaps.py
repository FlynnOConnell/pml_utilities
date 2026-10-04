"""The name a selector lists a graphic's colormap under."""

from cmap import Colormap
from mbo_utilities.gui._colormaps import DEFAULT_COLORMAPS, listed_name


def test_a_graphics_colormap_is_listed_under_a_name_that_opens_it():
    # fastplotlib hands back the object; str() of it is its repr, not a name
    gnuplot2 = Colormap("gnuplot2")
    assert str(gnuplot2).startswith("Colormap(")
    assert listed_name(gnuplot2) == "gnuplot2"
    for name in DEFAULT_COLORMAPS:
        assert listed_name(Colormap(name)) == name
    # one we do not list keeps the catalog's own name, which opens the same map
    other = Colormap("nipy_spectral")
    assert listed_name(other) not in DEFAULT_COLORMAPS
    assert Colormap(listed_name(other)).name == other.name
