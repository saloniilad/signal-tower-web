# /// script
# dependencies = [
#     "numpy",
#     "pygame-ce",
# ]
# ///
"""
Pygbag entry point for Signal Tower.

Desktop:   python main.py          (or: python signal_tower.py)
Browser:   python -m pygbag .      (run inside this folder)
"""
import asyncio

import pygame   # Pygbag scans THIS file for imports, so pygame must be imported here
import numpy    # make sure Pygbag bundles numpy

import signal_tower

asyncio.run(signal_tower.main())