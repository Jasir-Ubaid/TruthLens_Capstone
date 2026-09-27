"""
TruthLens Backend Package
Registers module aliases for hyphenated directory names (cv-pipeline -> cv_pipeline, nlp-pipeline -> nlp_pipeline).
"""

import os
import sys
import importlib.util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


def _register_package_alias(alias_name: str, folder_name: str):
    folder_path = os.path.join(PROJECT_ROOT, folder_name)
    init_path = os.path.join(folder_path, "__init__.py")
    if os.path.exists(init_path) and alias_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            alias_name,
            init_path,
            submodule_search_locations=[folder_path]
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[alias_name] = mod
            spec.loader.exec_module(mod)


_register_package_alias("cv_pipeline", "cv-pipeline")
_register_package_alias("nlp_pipeline", "nlp-pipeline")
