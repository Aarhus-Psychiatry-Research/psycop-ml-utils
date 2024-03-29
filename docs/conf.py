# pylint: disable-all

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

from psycop_ml_utils import __version__

# -- Project information -----------------------------------------------------

project = "psycop-ml-utils"
author = "Martin Bernstorff, Lasse Hansen, and Kenneth Enevoldsen"

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinx.ext.githubpages",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# list of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"  # "press", "sphinx_rtd_theme", "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_show_sourcelink = True

html_context = {
    "display_github": True,  # Add 'Edit on Github' link instead of 'View page source'
    "github_user": "Aarhus-Psychiatry-Research",
    "github_repo": project,
    "github_version": "main",
    "conf_py_path": "/docs/",
}


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "source_repository": "https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_logo": "icon_with_title.png",
    "dark_logo": "icon_with_title.png",
    "light_css_variables": {
        "color-brand-primary": "#204279",
        "color-brand-content": "#204279",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4872b8",
        "color-brand-content": "#4872b8",
    },
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
}
