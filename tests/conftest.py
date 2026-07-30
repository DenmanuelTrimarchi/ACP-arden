"""Test configuration.

Puts the project root on ``sys.path`` so ``ACP_arden`` imports as a plain
module. Nothing outside this repository is added to the path, and no test in
this suite loads a real model binary or reads a real dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
