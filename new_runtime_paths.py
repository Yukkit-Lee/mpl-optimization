#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable path resolution shared by the ``new_`` secure-feature pipeline.

Every location can be overridden with a ``NEW_*`` environment variable. The
fallback layout uses paths relative to the checked-out project, so an AutoDL
clone never inherits workstation-specific paths. This module remains Python
3.6 compatible because legacy TF-Encrypted scripts import it.
"""

from __future__ import print_function

import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent)
).resolve()
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_ROOT = Path(
    os.environ.get("NEW_DATA_ROOT", WORKSPACE_ROOT / "new_data")
).resolve()
MTT_ROOT = Path(
    os.environ.get("NEW_MTT_ROOT", WORKSPACE_ROOT / "mtt-distillation")
).resolve()
OFFICIAL_MTT_ROOT = Path(
    os.environ.get(
        "NEW_OFFICIAL_MTT_ROOT", DATA_ROOT / "mtt_data" / "georgecazenavette"
    )
).resolve()
OUTPUT_ROOT = Path(os.environ.get("NEW_OUTPUT_ROOT", PROJECT_ROOT)).resolve()
SECRET_ROOT = Path(
    os.environ.get("NEW_SECRET_ROOT", DATA_ROOT / "new_secure_feature_secrets")
).resolve()

ENV_ROOT = Path(os.environ.get("NEW_ENV_ROOT", WORKSPACE_ROOT / "new_envs")).resolve()
MODERN_PYTHON = os.environ.get(
    "NEW_MODERN_PYTHON", str(ENV_ROOT / "distillation" / "bin" / "python")
)
TFE_PYTHON = os.environ.get(
    "NEW_TFE_PYTHON", str(ENV_ROOT / "tfe" / "bin" / "python")
)
TMUX = os.environ.get("NEW_TMUX", shutil.which("tmux") or "tmux")


def current_or_modern_python():
    """Use the active interpreter unless an explicit modern Python is set."""

    return os.environ.get("NEW_MODERN_PYTHON", sys.executable)

