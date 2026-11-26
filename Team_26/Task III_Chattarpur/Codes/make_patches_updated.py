import argparse
import csv
import os
from contextlib import ExitStack

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


PATCH = 64
OVERLAP = 10
STRIDE = PATCH - OVERLAP

IMAGE_KEYS = ["before_VH", "before_VV", "after_VH", "after_VV"]
ALL_KEYS = IMAGE_KEYS + ["mask"]
MIN_VALID_DEFAULT = 0.80
MIN_FLOOD_DEFAULT = 0.20


def same_grid(a, b):
    """Return whether two rasters have exactly the same pixel grid."""
    return (
        a.width == b.width
        and a.height == b.height
        and a.crs == b.crs
        and a.transform.almost_equals(b.transform)
    )


def describe_grid(src):
    return (
        f"size={src.width}x{src.height}, CRS={src.crs}, "
        f"transform={src.transform}"
    )


def window_starts(length, patch=PATCH, stride=STRIDE):
    """Generate patch offsets, including the final image edge."""
    if length < patch:
        return []

    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] + patch < length:
        starts.append(length - patch)
    return starts


def open_aligned(path, ref, stack, key):
    """
    Open a raster and return it on the reference grid.

    SAR rasters are resampled with bilinear interpolation when their grid
    differs from the mask grid. The mask itself is the reference and is
    never resampled.
    """
    src = stack.enter_context(rasterio.open(path))

    if same_grid(src, ref):
        print(f"[grid] {key}: already matches the mask grid.")
        return src

    print(
        f"[grid] {key}: DOES NOT match the mask grid; "
        "using a virtual warp to the mask grid (bilinear resampling)."
    )
    return stack.enter_context(
        WarpedVRT(
            src,
            crs=ref.crs,
            transform=ref.transform,
            width=ref.width,
            height=ref.height,
            resampling=Resampling.bilinear,
        )
    )


def write_patch(path, arr, transform, crs, nodata):
    """Write a single-band GeoTIFF patch."""
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": arr.dtype,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }

    if nodata is not None and np.isfinite(nodata):
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def read_sar_patch(src, win, zero_is_nodata):
    """Read one SAR patch and return float32 data plus its valid-pixel mask."""
    raw = src.read(1, window=win, masked=True)
    arr = raw.astype(np.float32).filled(np.nan)

    valid = ~np.ma.getmaskarray(raw) & np.isfinite(arr)
    if zero_is_nodata:
        valid &= arr != 0

    return arr, valid


def read_mask_patch(src, win):
    """Read the label patch and independently determine valid mask pixels."""
    raw = src.read(1, window=win, masked=True)
    values = np.asarray(raw.data)

    valid = ~np.ma.getmaskarray(raw)
    if np.issubdtype(values.dtype, np.number):
        valid &= np.isfinite(values)

    # Output mask is binary: 1=flood, 0=non-flood or invalid.
    flood = np.where((values > 0) & valid, 255, 0).astype(np.uint8)
    return flood, valid


def make_patches(
    paths,
    out_root,
    district,
    min_valid=MIN_VALID_DEFAULT,
    min_flood=MIN_FLOOD_DEFAULT,
    zero_is_nodata=False,
    preview=True,
):
    """
    Create five co-registered patch sets: before/after VH, before/after VV,
    and the flood mask.

    A candidate area is retained only when:
      1. Each of the four SAR patches has at least min_valid valid pixels.
      2. The mask patch has at least min_valid valid pixels.
      3. At least min_flood of the valid mask pixels are labelled as flood.

    All five outputs for a retained area use the same patch ID and
    geospatial transform.
    """
    os.makedirs(out_root, exist_ok=True)
    for key in ALL_KEYS:
        os.makedirs(os.path.join(out_root, key), exist_ok=True)

    index_path = os.path.join(out_root, "patch_index.csv")

    with ExitStack() as stack:
        # Mask defines the output grid and patch geometry.
        mask_src = stack.enter_context(rasterio.open(paths["mask"]))
        height, width = mask_src.height, mask_src.width

        if height < PATCH or width < PATCH:
            raise ValueError(
                f"Mask raster is {width}x{height}; it is smaller than "
                f"one {PATCH}x{PATCH} patch."
            )

        print("[grid] Reference grid (mask):")
        print("       " + describe_grid(mask_src))

        # Explicitly report whether each before/after polarization pair
        # already shares a grid, before any alignment is performed.
        raw_sources = {}
        for key in IMAGE_KEYS:
            raw_sources[key] = stack.enter_context(
                rasterio.open(paths[key])
            )

        for before_key, after_key, polarization in (
            ("before_VH", "after_VH", "VH"),
            ("before_VV", "after_VV", "VV"),
        ):
            before_src = raw_sources[before_key]
            after_src = raw_sources[after_key]
            if same_grid(before_src, after_src):
                print(
                    f"[grid] Before/after {polarization}: SAME grid."
                )
            else:
                print(
                    f"[grid] Before/after {polarization}: DIFFERENT grids."
                )
                print(f"       before: {describe_grid(before_src)}")
                print(f"       after : {describe_grid(after_src)}")

        # Use mask grid as the common grid so every saved patch is aligned.
        # Reuse already-open datasets when they match; otherwise create VRTs.
        aligned = {}
        for key in IMAGE_KEYS:
            src = raw_sources[key]
            if same_grid(src, mask_src):
                print(f"[grid] {key}: matches the mask grid.")
                aligned[key] = src
            else:
                print(
                    f"[grid] {key}: differs from the mask grid; "
                    "virtually aligning to mask grid with bilinear resampling."
                )
                aligned[key] = stack.enter_context(
                    WarpedVRT(
                        src,
                        crs=mask_src.crs,
                        transform=mask_src.transform,
                        width=width,
                        height=height,
                        resampling=Resampling.bilinear,
                        nodata=np.nan,
                    )
                )

        # Output nodata is retained from the source where available.
        nodata = {}
        for key, src in aligned.items():
            nodata[key] = (
                src.nodata if src.nodata is not None else np.nan
            )

        rows = window_starts(height)
        cols = window_starts(width)
        candidate_count = len(rows) * len(cols)

        print(
            f"[patches] Reference raster: {width} x {height} | "
            f"candidate patches: {len(rows)} x {len(cols)} "
            f"= {candidate_count}"
        )
        print(
            f"[filters] Minimum valid fraction per each of 5 inputs: "
            f"{min_valid:.0%}; minimum flood fraction of valid mask pixels: "
            f"{min_flood:.0%}"
        )

        n_saved = 0
        rejected_valid = 0
        rejected_flood = 0
        preview_data = None
        preview_best = -1.0

        with open(index_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "id",
                "row",
                "col",
                "before_VH_valid_frac",
                "before_VV_valid_frac",
                "after_VH_valid_frac",
                "after_VV_valid_frac",
                "mask_valid_frac",
                "flood_frac_of_valid_mask",
            ])

            for ri, row in enumerate(rows):
                for ci, col in enumerate(cols):
                    win = Window(col, row, PATCH, PATCH)

                    patch_data = {}
                    valid_fracs = {}
                    valid_masks = {}

                    # Read each SAR patch and assess its own valid coverage.
                    for key in IMAGE_KEYS:
                        arr, valid = read_sar_patch(
                            aligned[key], win, zero_is_nodata
                        )
                        patch_data[key] = arr
                        valid_masks[key] = valid
                        valid_fracs[key] = float(valid.mean())

                    # Read and assess mask validity independently.
                    mask_patch, mask_valid = read_mask_patch(mask_src, win)
                    valid_fracs["mask"] = float(mask_valid.mean())

                    # Reject the entire location if ANY of the five inputs
                    # has less than the required valid-pixel coverage.
                    if any(
                        fraction < min_valid
                        for fraction in valid_fracs.values()
                    ):
                        rejected_valid += 1
                        continue

                    valid_mask_pixels = int(mask_valid.sum())
                    flood_pixels = int(np.count_nonzero(mask_patch))
                    flood_frac = (
                        flood_pixels / valid_mask_pixels
                        if valid_mask_pixels
                        else 0.0
                    )

                    # Reject all five patches together if the valid part of
                    # the label patch contains less than min_flood flood.
                    if flood_frac < min_flood:
                        rejected_flood += 1
                        continue

                    # Preserve invalid mask pixels as 0 in the binary output.
                    # No erosion/boundary cleaning is applied.
                    tag = f"{ri:03d}_{ci:03d}"
                    patch_id = f"{district}_{tag}"
                    transform = mask_src.window_transform(win)

                    # Write all four SAR patches with identical patch ID.
                    for key in IMAGE_KEYS:
                        write_patch(
                            os.path.join(
                                out_root,
                                key,
                                f"{patch_id}_{key}.tif",
                            ),
                            patch_data[key],
                            transform,
                            mask_src.crs,
                            nodata[key],
                        )

                    # Write the matching binary mask patch.
                    write_patch(
                        os.path.join(
                            out_root,
                            "mask",
                            f"{patch_id}_mask.tif",
                        ),
                        mask_patch,
                        transform,
                        mask_src.crs,
                        None,
                    )

                    writer.writerow([
                        patch_id,
                        ri,
                        ci,
                        round(valid_fracs["before_VH"], 4),
                        round(valid_fracs["before_VV"], 4),
                        round(valid_fracs["after_VH"], 4),
                        round(valid_fracs["after_VV"], 4),
                        round(valid_fracs["mask"], 4),
                        round(flood_frac, 4),
                    ])

                    n_saved += 1

                    # Keep the most complete common SAR coverage for preview.
                    common_valid = np.logical_and.reduce(
                        [valid_masks[key] for key in IMAGE_KEYS]
                    )
                    common_fraction = float(common_valid.mean())
                    if common_fraction > preview_best:
                        preview_best = common_fraction
                        preview_data = common_valid.copy()

                print(
                    f"[progress] Row {ri + 1}/{len(rows)} completed | "
                    f"saved={n_saved}, rejected_valid={rejected_valid}, "
                    f"rejected_flood={rejected_flood}"
                )

    print(f"\nDone. Saved {n_saved} aligned five-patch groups to: {out_root}")
    print(f"Rejected for <{min_valid:.0%} valid pixels in any input: {rejected_valid}")
    print(f"Rejected for <{min_flood:.0%} flood in valid mask pixels: {rejected_flood}")
    print(f"Patch index: {index_path}")

    if preview and preview_data is not None:
        plt.figure(figsize=(7, 6))
        plt.imshow(preview_data, cmap="gray")
        plt.title("Common Valid SAR Coverage - Best Saved Patch")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create co-registered 256x256 patches from before/after VH, "
            "before/after VV, and a flood mask."
        )
    )

    parser.add_argument("--before_vh", required=True)
    parser.add_argument("--before_vv", required=True)
    parser.add_argument("--after_vh", required=True)
    parser.add_argument("--after_vv", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--district", required=True)

    parser.add_argument(
        "--min_valid",
        type=float,
        default=MIN_VALID_DEFAULT,
        help="Minimum valid-pixel fraction required in EACH of the five patches (default: 0.10).",
    )
    parser.add_argument(
        "--min_flood",
        type=float,
        default=MIN_FLOOD_DEFAULT,
        help="Minimum flood fraction among valid mask pixels (default: 0.10).",
    )
    parser.add_argument(
        "--zero_is_nodata",
        action="store_true",
        help="Treat zero as nodata in all four SAR images.",
    )
    parser.add_argument(
        "--no_preview",
        action="store_true",
        help="Disable the valid-coverage preview.",
    )

    args = parser.parse_args()

    if not 0 <= args.min_valid <= 1:
        parser.error("--min_valid must be between 0 and 1")
    if not 0 <= args.min_flood <= 1:
        parser.error("--min_flood must be between 0 and 1")

    make_patches(
        paths={
            "before_VH": args.before_vh,
            "before_VV": args.before_vv,
            "after_VH": args.after_vh,
            "after_VV": args.after_vv,
            "mask": args.mask,
        },
        out_root=args.out,
        district=args.district,
        min_valid=args.min_valid,
        min_flood=args.min_flood,
        zero_is_nodata=args.zero_is_nodata,
        preview=not args.no_preview,
    )
