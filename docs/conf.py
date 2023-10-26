#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config
# The full version, including alpha/beta/rc tags
from arccnet import __version__

release = __version__

# -- Project information -----------------------------------------------------
project = "ARCCnet"
copyright = "2022, ARCAFF Team"
author = "ARCAFF Team"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinxcontrib.bibtex",
]

myst_enable_extensions = [
    "amsmath",
    # "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "reports"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:

source_suffix = {".rst": "restructuredtext"}

# The master toctree document.
master_doc = "index"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        (None, "http://www.astropy.org/astropy-data/intersphinx/python3.inv"),
    ),
    "numpy": (
        "https://numpy.org/doc/stable/",
        (None, "http://www.astropy.org/astropy-data/intersphinx/numpy.inv"),
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/reference/",
        (None, "http://www.astropy.org/astropy-data/intersphinx/scipy.inv"),
    ),
    "aiapy": ("https://aiapy.readthedocs.io/en/stable/", None),
    "asdf": ("https://asdf.readthedocs.io/en/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "astroquery": ("https://astroquery.readthedocs.io/en/latest/", None),
    "drms": ("https://docs.sunpy.org/projects/drms/en/stable/", None),
    "hvpy": ("https://hvpy.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "mpl_animators": ("https://docs.sunpy.org/projects/mpl-animators/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "parfive": ("https://parfive.readthedocs.io/en/stable/", None),
    "reproject": ("https://reproject.readthedocs.io/en/stable/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/latest/", None),
    "sunkit_image": ("https://docs.sunpy.org/projects/sunkit-image/en/stable/", None),
    "sunkit_instruments": ("https://docs.sunpy.org/projects/sunkit-instruments/en/stable/", None),
    "zeep": ("https://docs.python-zeep.org/en/stable/", None),
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "navigation_with_keys": False,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_static_path = ["_static"]
html_css_files = ["custom.css"]

numfig = True
recursive_update = False
suppress_warnings = ["myst.domains"]
bibtex_bibfiles = ["reports/refs.bib"]

# def setup(app):
#     """Add functions to the Sphinx setup."""
#     import subprocess
#     from typing import cast
#
#     from docutils import nodes
#     from docutils.parsers.rst import directives
#     from myst_parser.config.main import MdParserConfig
#     from sphinx.application import Sphinx
#     from sphinx.util.docutils import SphinxDirective
#
#     from myst_nb.core.config import NbParserConfig, Section
#
#     app = cast(Sphinx, app)
#
#     class _ConfigBase(SphinxDirective):
#         """Directive to automate printing of the configuration."""
#
#         @staticmethod
#         def table_header():
#             return [
#                 "```````{list-table}",
#                 ":header-rows: 1",
#                 "",
#                 "* - Name",
#                 "  - Type",
#                 "  - Default",
#                 "  - Description",
#             ]
#
#         @staticmethod
#         def field_default(value):
#             default = " ".join(f"{value!r}".splitlines())
#             if len(default) > 20:
#                 default = default[:20] + "..."
#             return default
#
#         @staticmethod
#         def field_type(field):
#             ctype = " ".join(str(field.type).splitlines())
#             ctype = ctype.replace("typing.", "")
#             ctype = ctype.replace("typing_extensions.", "")
#             for tname in ("str", "int", "float", "bool"):
#                 ctype = ctype.replace(f"<class '{tname}'>", tname)
#             return ctype
#
#     class MystNbConfigDirective(_ConfigBase):
#         required_arguments = 1
#         option_spec = {
#             "sphinx": directives.flag,
#             "section": lambda x: directives.choice(
#                 x, ["config", "read", "execute", "render"]
#             ),
#         }
#
#         def run(self):
#             """Run the directive."""
#             level_name = directives.choice(
#                 self.arguments[0], ["global_lvl", "file_lvl", "cell_lvl"]
#             )
#             level = Section[level_name]
#
#             config = NbParserConfig()
#             text = self.table_header()
#             count = 0
#             for name, value, field in config.as_triple():
#                 # filter by sphinx options
#                 if "sphinx" in self.options and field.metadata.get("sphinx_exclude"):
#                     continue
#                 # filter by level
#                 sections = field.metadata.get("sections") or []
#                 if level not in sections:
#                     continue
#                 # filter by section
#                 if "section" in self.options:
#                     section = Section[self.options["section"]]
#                     if section not in sections:
#                         continue
#
#                 if level == Section.global_lvl:
#                     name = f"nb_{name}"
#                 elif level == Section.cell_lvl:
#                     name = field.metadata.get("cell_key", name)
#
#                 description = " ".join(field.metadata.get("help", "").splitlines())
#                 default = self.field_default(value)
#                 ctype = self.field_type(field)
#                 text.extend(
#                     [
#                         f"* - `{name}`",
#                         f"  - `{ctype}`",
#                         f"  - `{default}`",
#                         f"  - {description}",
#                     ]
#                 )
#
#                 count += 1
#
#             if not count:
#                 return []
#
#             text.append("```````")
#             node = nodes.Element()
#             self.state.nested_parse(text, 0, node)
#             return node.children
#
#     class MystConfigDirective(_ConfigBase):
#         option_spec = {
#             "sphinx": directives.flag,
#         }
#
#         def run(self):
#             """Run the directive."""
#             config = MdParserConfig()
#             text = self.table_header()
#             count = 0
#             for name, value, field in config.as_triple():
#                 # filter by sphinx options
#                 if "sphinx" in self.options and field.metadata.get("sphinx_exclude"):
#                     continue
#
#                 name = f"myst_{name}"
#                 description = " ".join(field.metadata.get("help", "").splitlines())
#                 default = self.field_default(value)
#                 ctype = self.field_type(field)
#                 text.extend(
#                     [
#                         f"* - `{name}`",
#                         f"  - `{ctype}`",
#                         f"  - `{default}`",
#                         f"  - {description}",
#                     ]
#                 )
#
#                 count += 1
#
#             if not count:
#                 return []
#
#             text.append("```````")
#             node = nodes.Element()
#             self.state.nested_parse(text, 0, node)
#             return node.children
#
#     class DocutilsCliHelpDirective(SphinxDirective):
#         """Directive to print the docutils CLI help."""
#
#         has_content = False
#         required_arguments = 0
#         optional_arguments = 0
#         final_argument_whitespace = False
#         option_spec = {}
#
#         def run(self):
#             """Run the directive."""
#             import io
#
#             from docutils import nodes
#             from docutils.frontend import OptionParser
#
#             from myst_nb.docutils_ import Parser as DocutilsParser
#
#             stream = io.StringIO()
#             OptionParser(
#                 components=(DocutilsParser,),
#                 usage="mystnb-docutils-<writer> [options] [<source> [<destination>]]",
#             ).print_help(stream)
#             return [nodes.literal_block("", stream.getvalue())]
#
#     app.add_directive("myst-config", MystConfigDirective)
#     app.add_directive("mystnb-config", MystNbConfigDirective)
#     app.add_directive("docutils-cli-help", DocutilsCliHelpDirective)
