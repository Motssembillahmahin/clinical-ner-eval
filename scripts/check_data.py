"""Data smoke-test.

Loads NCBI Disease, BC5CDR, and MACCROBAT individually, then builds the combined
dataset. Prints split sizes, a sample note→entity pair, and label distributions.
Exit code 0 = all green; 1 = at least one failure.
"""

import collections
import importlib
import sys
import traceback

_ADAPTERS = [
    ("NCBI Disease", "src.datasets.ncbi", "NCBIDiseaseAdapter"),
    ("BC5CDR", "src.datasets.bc5cdr", "BC5CDRAdapter"),
    ("MACCROBAT", "src.datasets.maccrobat", "MaccrobatAdapter"),
]


def _check_adapter(label, module_path, cls_name):
    print(f"\n{'='*60}")
    print(f"[{label}]")
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        adapter = cls(label_mode="harmonized")
        ds = adapter.load()

        for split in ("train", "validation", "test"):
            if split in ds:
                cols = list(ds[split].features.keys())
                print(f"  {split}: {len(ds[split])} examples | columns: {cols}")

        # Verify required columns exist
        train = ds["train"]
        for col in ("tokens", "bio"):
            if col not in train.features:
                print(f"  [FAIL] missing column '{col}' — got {list(train.features)}")
                return False

        # Sample note → entity pair
        sample = train[0]
        pairs = [(t, b) for t, b in zip(sample["tokens"][:20], sample["bio"][:20])
                 if b != "O"]
        print(f"  Sample entities (first 20 tokens): {pairs or '(none)'}")

        # Label distribution
        all_tags = [t for ex in train for t in ex["bio"]]
        counter = collections.Counter(all_tags)
        non_o = {k: v for k, v in sorted(counter.items()) if k != "O"}
        print(f"  Entity tags (train): {non_o}")
        print(f"  [OK] {label}")
        return True

    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        traceback.print_exc()
        return False


def _check_combined():
    print(f"\n{'='*60}")
    print("[Combined dataset — NCBI + BC5CDR + MACCROBAT]")
    try:
        from src.datasets import build_dataset
        ds, labels = build_dataset()
        for split in ds:
            print(f"  {split}: {len(ds[split])} examples")
        print(f"  Labels: {labels}")
        sample = ds["train"][0]
        print(f"  Sample tokens[:5]: {sample['tokens'][:5]}")
        print(f"  Sample int labels[:5]: {sample['labels'][:5]}")
        print(f"  [OK] Combined build")
        return True
    except Exception as e:
        print(f"  [FAIL] Combined build: {e}")
        traceback.print_exc()
        return False


def main():
    results = [_check_adapter(*args) for args in _ADAPTERS]
    results.append(_check_combined())

    print(f"\n{'='*60}")
    n_pass = sum(results)
    print(f"Results: {n_pass}/{len(results)} passed")
    if all(results):
        print("[ALL PASS] Ready to train.")
        return 0
    print("[SOME FAILURES] Fix issues above before training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
