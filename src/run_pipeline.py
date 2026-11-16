
import argparse
import subprocess
import sys
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"


def run_script(script_name, *arguments):
    """Execute a Python script and stop if it fails."""

    script_path = SRC_DIR / script_name

    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print(f"{'=' * 60}\n")

    subprocess.run(
        [sys.executable, str(script_path), *arguments],
        cwd=BASE_DIR,
        check=True
    )


def main():

    parser = argparse.ArgumentParser(
        description="RNA-seq Machine Learning Classification Pipeline"
    )

    parser.add_argument(
        "--counts",
        default="data/example_gene_counts.tsv",
        help="Path to raw gene count matrix"
    )

    parser.add_argument(
        "--metadata",
        default="data/example_metadata.csv",
        help="Path to sample metadata"
    )

    args = parser.parse_args()

    counts = str(Path(args.counts).resolve())
    metadata = str(Path(args.metadata).resolve())

    # Validate inputs
    if not Path(counts).is_file():
        parser.error(f"Count matrix not found: {counts}")

    if not Path(metadata).is_file():
        parser.error(f"Metadata file not found: {metadata}")

    try:
        # Step 1: Evaluate seven ML models
        run_script(
            "train_models.py",
            "--counts", counts,
            "--metadata", metadata
        )

        # Step 2: Select and refit the best algorithm
        run_script(
            "refit_best_model.py",
            "--counts", counts,
            "--metadata", metadata
        )

        # Step 3: Generate evaluation figures
        run_script("plot_results.py")

    except subprocess.CalledProcessError as error:
        print(
            f"\nPipeline failed with exit code "
            f"{error.returncode}."
        )
        sys.exit(error.returncode)

    print("\nRNA-seq ML pipeline completed successfully!")
    print("Check results/, figures/, and models/.")


if __name__ == "__main__":
    main()
