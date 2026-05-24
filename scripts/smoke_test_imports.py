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

    print("Project modules (syntax check via py_compile)...")
    import py_compile
    for mod in ("src/dpok_imagereward.py", "src/eval_report.py", "src/dpok_eval_metrics.py"):
        try:
            py_compile.compile(mod, doraise=True)
            print(f"  OK  {mod}")
        except py_compile.PyCompileError as e:
            print(f"  FAIL {mod}: {e}")
            raise

    print("\nAll required imports passed.")
    print(f"CUDA available: {__import__('torch').cuda.is_available()}")


if __name__ == "__main__":
    main()
