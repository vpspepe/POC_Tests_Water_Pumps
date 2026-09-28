"""Module for downloading metadata and diverse pump geometries from Hugging Face."""

import argparse
import glob
import json
import os
from collections.abc import Sequence

import kennard_stone as ks
import polars as pl
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# Default configurations and repository constants
DEFAULT_REPO_ID = "luminary-shift/Pump"
DEFAULT_METADATA_DIR = "./pump_metadata"
DEFAULT_CACHE_FILE = "shift_pump_files.txt"

# Column groups for typing metadata records
FLOAT_COLS = [
    "flow_rate",
    "head",
    "outer_diameter_factor",
    "outlet_width_factor",
    "shroud_diameter_factor",
    "volute_q_factor",
    "volute_r_factor",
    "teAngleDelta",
    "leAngleDelta",
    "compactness",
    "leWidthVar",
    "leRake",
    "teRake",
    "flow_rate_op_condition",
    "leShapevar",
    "shrAngleVar",
    "stream_velocity",
    "area_ref",
    "length_ref",
    "length_ref_pitch",
    "length_ref_roll",
    "length_ref_yaw",
    "moment_center_x",
    "moment_center_y",
    "moment_center_z",
    "R",
    "gamma",
    "cp",
    "air_density",
]

INT_COLS = [
    "rotational_speed",
    "fix_blade_number",
    "leAngleVar",
    "leShapeVar",
    "tipGap",
    "leEllipseFactor",
    "mach",
    "pressure",
    "temperature",
]

BOOL_COLS = ["teTangential"]

# Features used for geometric shape diversity analysis
ACTIVE_GEOMETRIC_COLS = [
    "outer_diameter_factor",
    "outlet_width_factor",
    "shroud_diameter_factor",
    "teAngleDelta",
    "leAngleDelta",
    "compactness",
    "leWidthVar",
    "leRake",
    "teRake",
    "leShapevar",
]


def get_repo_files(
    api: HfApi,
    repo_id: str = DEFAULT_REPO_ID,
    cache_file: str = DEFAULT_CACHE_FILE,
) -> list[str]:
    """Retrieve repository file list, utilizing local disk cache if available.

    Args:
        api: Initialized Hugging Face API client.
        repo_id: Repository ID on Hugging Face.
        cache_file: Path to local cache text file storing the file list.

    Returns:
        List of all file paths in the dataset repository.
    """
    if not os.path.exists(cache_file):
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        with open(cache_file, "w") as f:
            f.writelines(f"{file}\n" for file in all_files)
    else:
        with open(cache_file, "r") as f:
            all_files = f.read().splitlines()
    return all_files


def download_metadata_params(
    repo_id: str,
    token: str | None,
    files: Sequence[str],
    metadata_dir: str,
) -> None:
    """Download individual params.json metadata files from Hugging Face repository.

    Args:
        repo_id: Repository ID on Hugging Face.
        token: Hugging Face authentication token.
        files: List of file paths to download (e.g. params.json paths).
        metadata_dir: Target local directory where files will be stored.
    """
    for f in tqdm(files, desc="Downloading params.json files", unit="file"):
        hf_hub_download(
            repo_id=repo_id,
            token=token,
            filename=f,
            local_dir=metadata_dir,
            repo_type="dataset",
        )


def load_and_clean_metadata(metadata_dir: str) -> pl.DataFrame:
    """Load sample params.json metadata files and cast columns to appropriate Polars types.

    Args:
        metadata_dir: Path to the local metadata directory containing sample_* folders.

    Returns:
        Typed Polars DataFrame containing all sample parameters and sample_id.
    """
    json_paths = sorted(
        glob.glob(os.path.join(metadata_dir, "sample_*", "params.json"))
    )
    records = []
    for path in tqdm(json_paths, desc="Loading sample metadata", unit="sample"):
        with open(path, "r") as f:
            data = json.load(f)
            data["sample_id"] = os.path.basename(os.path.dirname(path))
            records.append(data)

    df = pl.DataFrame(records)

    cast_exprs = []
    for col in FLOAT_COLS:
        if col in df.columns:
            cast_exprs.append(pl.col(col).cast(pl.Float64))

    for col in INT_COLS:
        if col in df.columns:
            cast_exprs.append(pl.col(col).cast(pl.Int64))

    for col in BOOL_COLS:
        if col in df.columns:
            if df[col].dtype == pl.String:
                cast_exprs.append(
                    (pl.col(col).str.to_lowercase() == "true").cast(pl.Boolean)
                )
            else:
                cast_exprs.append(pl.col(col).cast(pl.Boolean))

    return df.with_columns(cast_exprs)


def select_diverse_samples_ks(
    df: pl.DataFrame,
    n_samples: int = 200,
    feature_cols: Sequence[str] = ACTIVE_GEOMETRIC_COLS,
) -> tuple[pl.DataFrame, list[str]]:
    """Select a diverse subset of shapes using the Kennard-Stone algorithm.

    Args:
        df: Polars DataFrame containing parameter metadata.
        n_samples: Number of diverse samples to select.
        feature_cols: Geometric feature columns used for similarity and distance calculation.

    Returns:
        Tuple containing:
            - df_selected: Polars DataFrame with the selected samples.
            - selected_folders: List of sample IDs corresponding to the selected shapes.
    """
    existing_geo_cols = [col for col in feature_cols if col in df.columns]

    df_features = (
        df.select(existing_geo_cols)
        .with_columns([pl.col(col).cast(pl.Float64) for col in existing_geo_cols])
        .fill_null(0.0)
    )

    X = df_features.to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _, _, df_selected, _ = ks.train_test_split(X_scaled, df, train_size=n_samples)

    selected_folders = df_selected["sample_id"].to_list()
    return df_selected, selected_folders


def get_missing_stl_files(
    all_stl_files: Sequence[str],
    metadata_dir: str,
) -> list[str]:
    """Find STL files present in the repository that have not yet been downloaded locally.

    Args:
        all_stl_files: Full list of STL files in the remote dataset.
        metadata_dir: Local directory containing downloaded dataset files.

    Returns:
        List of missing STL file paths relative to dataset root.
    """
    existing_stls = glob.glob(os.path.join(metadata_dir, "**", "*.stl"), recursive=True)
    existing_rel_paths = {os.path.relpath(path, metadata_dir) for path in existing_stls}
    return sorted(set(all_stl_files) - existing_rel_paths)


def download_stl_meshes(
    repo_id: str,
    token: str | None,
    metadata_dir: str,
    allow_patterns: Sequence[str],
    max_workers: int = 4,
    tqdm_class: type[tqdm] | None = tqdm,
) -> None:
    """Download STL files from Hugging Face repository using snapshot_download.

    Args:
        repo_id: Repository ID on Hugging Face.
        token: Hugging Face authentication token.
        metadata_dir: Target local directory where files will be stored.
        allow_patterns: List or glob patterns of files to download.
        max_workers: Number of concurrent workers for parallel downloads.
        tqdm_class: Progress bar class to use for reporting download progress.
    """
    snapshot_download(
        repo_id=repo_id,
        token=token,
        repo_type="dataset",
        local_dir=metadata_dir,
        allow_patterns=list(allow_patterns),
        max_workers=max_workers,
        tqdm_class=tqdm_class,
    )


def main(kennard_stone_select: bool = True, ks_samples: int = 200) -> None:
    """Download metadata, perform optional diverse sampling, and fetch pump 3D STL meshes."""
    load_dotenv()
    hf_token = os.getenv("HF_READ_TOKEN")
    repo_id = DEFAULT_REPO_ID
    metadata_dir = DEFAULT_METADATA_DIR

    api = HfApi(token=hf_token)

    # 1. Fetch file list
    all_files = get_repo_files(api, repo_id=repo_id)
    params_json_files = [f for f in all_files if f.endswith("params.json")]
    all_stl_files = [f for f in all_files if f.endswith("merged_surfaces.stl")]

    # 2. Download parameter metadata
    download_metadata_params(repo_id, hf_token, params_json_files, metadata_dir)

    # 3. Determine STL files to download
    if kennard_stone_select:
        # Load and clean parameters
        df = load_and_clean_metadata(metadata_dir)

        # Diverse subset selection via Kennard-Stone algorithm
        df_selected, selected_folders = select_diverse_samples_ks(
            df, n_samples=ks_samples
        )

        print(f"Selected {len(selected_folders)} diverse shapes:")
        print(selected_folders)

        existing_geo_cols = [c for c in ACTIVE_GEOMETRIC_COLS if c in df.columns]
        df_filtered = df_selected.filter(pl.col("sample_id").is_in(selected_folders))
        print(df_filtered.select(existing_geo_cols).describe())

        # Filter STL files matching selected diverse samples
        selected_set = set(selected_folders)
        pump_stl_selected_files = [
            f
            for f in all_files
            if f.endswith(".stl") and f.split("/")[0] in selected_set
        ]
    else:
        pump_stl_selected_files = get_missing_stl_files(all_stl_files, metadata_dir)

    # 4. Download meshes
    snapshot_download(
        repo_id=repo_id,
        token=hf_token,
        repo_type="dataset",
        local_dir=metadata_dir,
        allow_patterns=pump_stl_selected_files,
        max_workers=4,
        tqdm_class=tqdm,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download pump metadata and CAD meshes with optional Kennard-Stone diversity sampling."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all missing meshes instead of selecting diverse samples.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of diverse samples to select with Kennard-Stone (default: 200).",
    )
    args = parser.parse_args()

    main(kennard_stone_select=not args.all, ks_samples=args.samples)
