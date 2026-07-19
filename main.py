#!/usr/bin/env python3
"""Entrypoint chính cho Aegis Trading System (v11.8)."""
import sys
from aegis.pipelines.live_pipeline import main as live_main

if __name__ == '__main__':
    sys.exit(live_main())
