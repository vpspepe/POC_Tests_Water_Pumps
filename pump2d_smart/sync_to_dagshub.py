#!/usr/bin/env python3
"""Convenience entrypoint to sync untracked local MLflow runs to DAGsHub."""

from src.training.sync_cloud import main

if __name__ == "__main__":
    main()
