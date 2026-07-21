"""
Seeds the local synthetic dataset used for offline training/testing.

    python scripts/seed_data.py --n-per-class 200
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.data_prep import write_sample_file, SAMPLE_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=200)
    args = parser.parse_args()

    write_sample_file(SAMPLE_PATH, n_per_class=args.n_per_class)
    print(f"Seeded {args.n_per_class * 2} labeled prompts to {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
