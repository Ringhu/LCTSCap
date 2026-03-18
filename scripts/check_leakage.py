"""Check for subject-level data leakage between splits."""
import argparse
from pathlib import Path

from lctscap.data.splits import verify_no_leakage
from lctscap.utils.io import read_jsonl


def main():
    parser = argparse.ArgumentParser(description="Verify no subject leakage between splits")
    parser.add_argument(
        "--splits_dir",
        type=str,
        required=True,
        help="Directory containing split JSONL files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="capture24",
        choices=["capture24", "harth"],
    )
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)

    # Load subjects from each split
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_file = splits_dir / f"{args.dataset}_{split_name}.jsonl"
        if not split_file.exists():
            split_file = splits_dir / f"{split_name}.jsonl"
        if split_file.exists():
            samples = read_jsonl(split_file)
            subjects = set()
            for s in samples:
                subj = s.get("subject_id", s.get("participant_id"))
                if subj is not None:
                    subjects.add(subj)
            splits[split_name] = list(subjects)
            print(f"{split_name}: {len(subjects)} subjects")

    if len(splits) < 2:
        print("ERROR: Found fewer than 2 splits. Cannot check leakage.")
        return

    is_clean = verify_no_leakage(splits)
    if is_clean:
        print("\nNo subject leakage detected. All splits are clean.")
    else:
        print("\nWARNING: Subject leakage detected!")
        split_names = list(splits.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                a, b = split_names[i], split_names[j]
                overlap = set(splits[a]) & set(splits[b])
                if overlap:
                    print(f"  {a} & {b}: {len(overlap)} overlapping subjects: {overlap}")


if __name__ == "__main__":
    main()
