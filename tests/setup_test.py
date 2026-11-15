# -*- coding: utf-8 -*-
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_setup_py_does_not_depend_on_locale_encoding():
    # README.md is UTF-8. With a non UTF-8 locale encoding (ASCII here, cp936/cp1252 on Windows)
    # setup.py must still be able to read it, otherwise `pip install` fails.
    env = dict(os.environ, PYTHONUTF8="0", PYTHONCOERCECLOCALE="0", LC_ALL="C", LANG="C")
    process = subprocess.Popen([sys.executable, "setup.py", "--version"], cwd=REPO_ROOT, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    assert process.returncode == 0, stderr.decode("utf-8", "replace")
    assert stdout.decode("utf-8", "replace").strip()
