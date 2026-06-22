"""
Libera L2 Unfiltered Radiances algorithm.

Implements the 8-step Libera SDC processing workflow:
1. Read input manifest
2. Load all NetCDF inputs (L1B RAD-4CH + FMATCH-CAM)
3. Classify scene/cloud from CAM; apply polynomial regression to produce unfiltered radiances
4-5. Write NetCDF data product
6-8. Create and write output manifest
"""

import argparse
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr
from cloudpathlib import AnyPath, S3Path

from libera_utils import Manifest
from libera_utils import smart_open
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_COEF_DIR = _REPO_ROOT / "coefficients"


def main():
    now = datetime.now(UTC)
    args = parse_cli_args()
    configure_task_logging(f"example_algorithm_{now}")
    logger.debug(f"CLI args: {args}")
    if not args.manifest:
        raise ValueError("Manifest file path must be provided as a command line argument")
    manifest_path = AnyPath(args.manifest)
    output_manifest_path = algorithm(manifest_path)
    logger.info(f"Processing complete. Output manifest: {output_manifest_path}")


def parse_cli_args():
    parser = argparse.ArgumentParser(
        prog="libera-l2-unfiltered-radiances",
        description="Libera science data processing for unfiltering radiances"
    )
    parser.add_argument("manifest", type=str, help="Absolute path to the input manifest file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    logger.info("Step 1: Reading the input manifest file")
    input_manifest = Manifest.from_file(manifest_path)
    logger.info(f"Loaded manifest with {len(input_manifest.files)} files")

    logger.info("Step 2: Reading all input data from manifest files")
    all_input_data = read_all_input_data(input_manifest)

    logger.info("Step 3: Calculating science data variables")
    processed_data = calculate_science_data(all_input_data)

    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    logger.info("Steps 4-5: Creating and writing data product")
    output_data_file_path = create_and_write_data_product(
        processed_data=processed_data,
        output_path=dropbox_path
    )

    logger.info("Step 6: Creating output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)

    logger.info("Step 7: Adding data files to output manifest")
    output_manifest.add_files(output_data_file_path.path)

    logger.info("Step 8: Writing the output manifest")
    output_manifest_filepath = output_manifest.write(dropbox_path)
    logger.info(f"Output manifest written to: {output_manifest_filepath}")

    return output_manifest_filepath


def read_all_input_data(input_manifest: Manifest) -> dict[str, xr.Dataset]:
    all_data = {}
    for i, file_info in enumerate(input_manifest.files):
        logger.info(f"Reading file {i + 1}/{len(input_manifest.files)}: {file_info.filename}")
        try:
            with smart_open(file_info.filename) as file_handle:
                dataset = xr.open_dataset(file_handle)
                dataset.load()
                all_data[file_info.filename] = dataset
                logger.info(f"Loaded dataset with variables: {list(dataset.variables)}")
        except Exception as e:
            logger.error(f"Failed to open file {file_info.filename}: {e}")
            raise
    logger.info(f"Successfully loaded {len(all_data)} datasets")
    return all_data


def _get_l1b_dataset(all_input_data: dict[str, xr.Dataset]) -> xr.Dataset:
    for ds in all_input_data.values():
        if ds.attrs.get("ProductID") == "RAD-4CH":
            return ds
    return next(iter(all_input_data.values()))


def _get_cam_dataset(all_input_data: dict[str, xr.Dataset]) -> xr.Dataset:
    for ds in all_input_data.values():
        if ds.attrs.get("ProductID") == "FMATCH-CAM":
            return ds
    raise KeyError("No FMATCH-CAM dataset found in manifest")


def _find_coefficient_file() -> Path:
    env_path = os.getenv("COEFFICIENTS_FILE")
    if env_path:
        return Path(env_path)
    coef_files = sorted(_DEFAULT_COEF_DIR.glob("unfiltering_coefficients_*.nc"))
    if not coef_files:
        raise FileNotFoundError(f"No coefficient file found in {_DEFAULT_COEF_DIR}")
    return coef_files[-1]


def calculate_science_data(all_input_data: dict[str, xr.Dataset]) -> dict:
    """
    Step 3: Classify scene/cloud from CAM file, then apply unfiltering regression.

    Reads filtered radiances and angles from L1B, derives per-sample scene/cloud
    from the FMATCH-CAM file, looks up regression coefficients per angular bin,
    and applies the polynomial model to produce four unfiltered radiance channels.
    """
    from unfiltered_radiances.unfiltering import (
        classify_scene_cloud,
        apply_unfiltering,
        load_coefficients,
    )
    from prod.std.standard_method import SCENE_TYPES

    logger.info("Step 3: Applying unfiltering regression")

    l1b_ds = _get_l1b_dataset(all_input_data)
    cam_ds = _get_cam_dataset(all_input_data)

    sw_f  = l1b_ds["Filtered_Radiance_SW"].values
    ssw_f = l1b_ds["Filtered_Radiance_SSW"].values
    lw_f  = l1b_ds["Filtered_Radiance_LW"].values
    tot_f = l1b_ds["Filtered_Radiance_Tot"].values
    sza   = l1b_ds["Solar_Zenith_Surface"].values
    vza   = l1b_ds["Viewing_Zenith_Surface"].values
    raz   = l1b_ds["Relative_Azimuth_Surface"].values
    lat   = l1b_ds["Latitude"].values
    lon   = l1b_ds["Longitude"].values
    qf    = l1b_ds["Quality_Flag"].values
    times = l1b_ds["radiometer_time"].values

    logger.info(f"L1B loaded: {len(times)} samples")

    # Align CAM to L1B times (nearest-neighbor handles synthetic example mismatches;
    # in production both files share the same RADIOMETER_TIME values)
    l1b_times = l1b_ds["radiometer_time"].values
    cam_aligned = cam_ds.sel(RADIOMETER_TIME=l1b_times, method="nearest")

    scene_idx, cloud = classify_scene_cloud(cam_aligned)

    scene_counts = {SCENE_TYPES[i]: int((scene_idx == i).sum()) for i in range(len(SCENE_TYPES))}
    logger.info(f"Scene classification: {scene_counts}")
    logger.info(f"Cloud=1 samples: {int(cloud.sum())} / {len(cloud)}")

    coef_path = _find_coefficient_file()
    logger.info(f"Using coefficient file: {coef_path.name}")
    coef_ds = load_coefficients(coef_path)
    try:
        sw_u, ssw_u, lw_u, tot_u = apply_unfiltering(
            sw_f, ssw_f, lw_f, tot_f, sza, vza, raz, coef_ds,
            scene_idx=scene_idx,
            cloud=cloud,
        )
    finally:
        coef_ds.close()

    filled = int(np.isfinite(sw_u).sum())
    logger.info(f"Unfiltering complete: {filled}/{len(sw_u)} samples filled, {np.isnan(sw_u).sum()} NaN")

    return {
        "radiometer_time":                     times,
        "shortwave_unfiltered_radiance":       sw_u,
        "split_shortwave_unfiltered_radiance": ssw_u,
        "longwave_unfiltered_radiance":        lw_u,
        "total_unfiltered_radiance":           tot_u,
        "latitude":                            lat,
        "longitude":                           lon,
        "quality_flags":                       qf.astype(np.int32),
    }


def create_and_write_data_product(
        processed_data: dict,
        output_path: str | Path | S3Path
) -> AnyPath:
    logger.info("Steps 4-5: Creating and writing data product")

    script_dir = Path(__file__).parent
    product_config_file = script_dir / "l2-unfiltered-radiance-product-definition.yml"

    if not product_config_file.exists():
        raise FileNotFoundError(f"Product definition file not found: {product_config_file}")

    logger.info(f"Saving to {output_path}")
    output_file_path = write_libera_data_product(
        data_product_definition=product_config_file,
        data=processed_data,
        output_path=output_path,
        time_variable="radiometer_time",
        strict=True
    )
    logger.info(f"Data product written to: {output_file_path}")
    return output_file_path


if __name__ == "__main__":
    main()
