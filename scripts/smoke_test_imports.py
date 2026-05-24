#!/usr/bin/env python3
"""Quick import check (no GPU training). Run: python scripts/smoke_test_imports.py"""

from __future__ import annotations

import sys


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK  {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        raise


def main() -> None:
    print("Core training stack...")
    check("torch", lambda: __import__("torch"))
    check("diffusers", lambda: __import__("diffusers"))
    check("transformers", lambda: __import__("transformers"))
    check("peft", lambda: __import__("peft"))
    check("open_clip", lambda: __import__("open_clip"))
    check("wandb", lambda: __import__("wandb"))
    check("PIL", lambda: __import__("PIL"))
    check("matplotlib", lambda: __import__("matplotlib"))

    print("Reward (optional at train time — required for dpok_imagereward.py)...")
    try:
        import ImageReward  # noqa: F401
        print("  OK  ImageReward")
    except ImportError:
        print("  WARN ImageReward not installed — pip install image-reward")

    print("Project modules...")
    check("dpok_scene_reward", lambda: __import__("dpok_scene_reward"))
    try:
        __import__("dpok_eval_metrics")
        print("  OK  dpok_eval_metrics")
    except ImportError as e:
        print(f"  WARN dpok_eval_metrics (install requirements-eval.txt for VQA): {e}")

    print("\nAll required imports passed.")
    print(f"CUDA available: {__import__('torch').cuda.is_available()}")


if __name__ == "__main__":
    main()
