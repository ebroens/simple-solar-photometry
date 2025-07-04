# docs/conf.py

import os
import sys

# Force matplotlib to use a non-interactive backend for headless environments like GitHub Actions
import matplotlib
matplotlib.use('Agg')

# Optional: Sphinx plot directive settings (helps with clarity and consistency in HTML output)
plot_html_show_formats = False
plot_html_show_source_link = True
plot_rcparams = {'savefig.dpi': 150}


# Adjust this path to point to your project's root.
# Assuming 'docs' folder is directly inside the project root.
sys.path.insert(0, os.path.abspath('../')) 

# -- Project information -----------------------------------------------------
project = 'Simple Solar Photometry'
copyright = '2025, Eric Broens'
author = 'Eric Broens'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
#extensions = [
#    'sphinx.ext.autodoc',     # For automatic API documentation from docstrings
#    'sphinx.ext.napoleon',    # For parsing NumPy and Google style docstrings
#    'sphinx.ext.viewcode',    # To link to the source code of objects
#    'sphinx.ext.autosummary', # For auto-generated summaries of modules/classes
#    'sphinx.ext.todo',        # For todo notes in documentation
#    'sphinx.ext.mathjax',     # For mathematical equations (e.g., for plot labels)
#    'sphinx.ext.githubpages', # Enables support for GitHub Pages (optional, but good practice)
#    'matplotlib.sphinxext.plot_directive', For embedding matplotlib plots directly in rst files
#]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'matplotlib.sphinxext.plot_directive', 
]


# Configure Napoleon for docstring styles (assuming Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo' # A modern, responsive theme. You can also use 'sphinx_rtd_theme'
# html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# You can add a logo
html_logo = "_static/logo_2.png"

# -- Options for Autodoc -----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource', # or 'alphabetical'
}

# Add the modules to autosummary. You might need to regenerate the autosummary files
# if you add new modules or change their structure.
autosummary_generate = True
