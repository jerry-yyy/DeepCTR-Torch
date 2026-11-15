# -*- coding: utf-8 -*-
import ast
import io
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _required_packages():
    with io.open(os.path.join(REPO_ROOT, "setup.py"), "r", encoding="utf-8") as setup_file:
        tree = ast.parse(setup_file.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "REQUIRED_PACKAGES":
            return [requirement.split(";")[0].strip() for requirement in ast.literal_eval(node.value)]
    raise AssertionError("REQUIRED_PACKAGES not found in setup.py")


def test_requests_is_a_declared_dependency():
    # deepctr_torch/__init__.py imports deepctr_torch/utils.py, which imports `requests` at module
    # level, so a plain `pip install deepctr-torch` has to install it.
    assert "requests" in _required_packages()
