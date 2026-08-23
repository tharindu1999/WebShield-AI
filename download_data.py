"""Download UCI dataset 967 and store its CSV in data/."""

from pathlib import Path
import shutil

from ucimlrepo import fetch_ucirepo


DATASET_ID = 967
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"


def download_dataset(output_path: Path = OUTPUT_PATH, force: bool = False) -> Path:
    """Fetch the official UCI dataset and write one reproducible CSV artifact."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        print(f"Dataset already exists: {output_path}")
        return output_path

    print(f"Downloading official UCI dataset ID {DATASET_ID}...")
    dataset = fetch_ucirepo(id=DATASET_ID)
    frame = dataset.data.original
    if frame is None:
        frame = dataset.data.features.copy()
        frame[dataset.data.targets.columns] = dataset.data.targets

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".csv.part")
    frame.to_csv(temporary_path, index=False)
    shutil.move(str(temporary_path), str(output_path))
    print(f"Saved {len(frame):,} records and {len(frame.columns)} columns to {output_path}")
    return output_path


if __name__ == "__main__":
    download_dataset()

