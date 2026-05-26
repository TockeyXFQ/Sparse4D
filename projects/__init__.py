# Marker file: turn ``projects`` from an implicit namespace package into a
# regular package, so that ``import projects.mmdet3d_plugin`` resolves to
# *this* directory even when other site-packages (e.g. some Meta libraries)
# also provide a ``projects/__init__.py``.
#
# Without this file, Python's namespace-package vs regular-package precedence
# causes the site-packages copy to win, which silently breaks any sub-import
# (``projects.mmdet3d_plugin`` would raise ``ModuleNotFoundError``).
