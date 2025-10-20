import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


@dataclass
class DiagnosticSlice:
    """Data captured for a single slice during simulation to aid plotting."""

    index: int
    expected_photons: np.ndarray
    shot_noise: np.ndarray
    electrons: np.ndarray
    electrons_out: np.ndarray
    adu: np.ndarray
    original_slice: np.ndarray
    after_shot_noise: np.ndarray
    after_electrons: np.ndarray
    after_dark_noise: np.ndarray


def simulate_dark_noise(
    data: np.ndarray,
    num_photons: float = 500,
    quantum_efficiency: float = 0.6,
    dark_noise_std: float = 2.29,
    sensitivity: float = 5.88,
    bitdepth: int = 12,
    seed: int | None = 42,
    diagnostics_slice: int | None = None,
    mask_threshold: float = 512,
) -> tuple[np.ndarray, DiagnosticSlice | None]:
    """Apply dark noise using the original simulator logic and return noisy datacube."""
    rng = np.random.default_rng(seed)

    input_dtype = data.dtype
    original = data.astype(np.float32, copy=False)

    shot_noise = rng.poisson(num_photons, size=original.shape)
    after_shot_noise = original.copy()
    after_shot_noise[shot_noise > mask_threshold] = 0

    electrons = np.round(quantum_efficiency * shot_noise)
    after_electrons = after_shot_noise.copy()
    after_electrons[electrons > mask_threshold] = 0

    electrons_out = np.round(
        electrons + rng.normal(loc=0.0, scale=dark_noise_std, size=original.shape)
    )
    after_dark_noise = after_electrons.copy()
    after_dark_noise[electrons_out > mask_threshold] = 0

    adu = np.round(electrons_out * sensitivity)
    adu = np.clip(adu, 0, 2**bitdepth - 1)

    # Clamp final noisy datacube to the original dtype range.
    if np.issubdtype(input_dtype, np.integer):
        info = np.iinfo(input_dtype)
    else:
        info = np.finfo(input_dtype)
    noisy = np.clip(after_dark_noise, info.min, info.max).astype(input_dtype)

    diagnostics = None
    if diagnostics_slice is not None:
        slice_index = int(np.clip(diagnostics_slice, 0, data.shape[2] - 1))
        diagnostics = DiagnosticSlice(
            index=slice_index,
            expected_photons=np.full(
                data.shape[:2], num_photons, dtype=np.float32
            ),
            shot_noise=shot_noise[:, :, slice_index].astype(np.float32, copy=True),
            electrons=electrons[:, :, slice_index].astype(np.float32, copy=True),
            electrons_out=electrons_out[:, :, slice_index].astype(
                np.float32, copy=True
            ),
            adu=adu[:, :, slice_index].astype(np.float32, copy=True),
            original_slice=original[:, :, slice_index].copy(),
            after_shot_noise=after_shot_noise[:, :, slice_index].copy(),
            after_electrons=after_electrons[:, :, slice_index].copy(),
            after_dark_noise=after_dark_noise[:, :, slice_index].copy(),
        )

    return noisy, diagnostics


def plot_slice(original: np.ndarray, noisy: np.ndarray, slice_index: int) -> None:
    """Plot a before/after comparison for a single spectral slice."""
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))

    ax0.set_title("Original")
    img0 = ax0.imshow(original[:, :, slice_index], cmap="gray")
    ax0.set_xticks([])
    ax0.set_yticks([])
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img0, cax=cax0)

    ax1.set_title("With Dark Noise")
    img1 = ax1.imshow(noisy[:, :, slice_index], cmap="gray")
    ax1.set_xticks([])
    ax1.set_yticks([])
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img1, cax=cax1)

    fig.tight_layout()
    plt.show()


def plot_diagnostics(diag: DiagnosticSlice) -> None:
    """Recreate the detailed diagnostic plots from the original simulator."""

    def percentile_limits(arr: np.ndarray, low: float = 5, high: float = 95) -> tuple[float, float]:
        vmin, vmax = np.percentile(arr, [low, high])
        if np.isclose(vmin, vmax):
            vmin = float(arr.min())
            vmax = float(arr.max())
            if np.isclose(vmin, vmax):
                vmin -= 1.0
                vmax += 1.0
        return vmin, vmax

    fig, ax = plt.subplots()
    vmin, vmax = percentile_limits(diag.expected_photons)
    img = ax.imshow(diag.expected_photons, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.imshow(diag.original_slice, alpha=0.4, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Expected Photons (No Noise)")
    cb = plt.colorbar(img, ax=ax)
    cb.set_label("Photons")
    plt.show()

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 5))

    vmin, vmax = percentile_limits(diag.expected_photons)
    img0 = ax0.imshow(diag.expected_photons, vmin=vmin, vmax=vmax, cmap="viridis")
    ax0.imshow(diag.original_slice, alpha=0.4, cmap="gray")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("Expected Photons")
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img0, cax=cax0)

    vmin, vmax = percentile_limits(diag.shot_noise)
    img1 = ax1.imshow(diag.shot_noise, vmin=vmin, vmax=vmax, cmap="viridis")
    ax1.imshow(diag.after_shot_noise, alpha=0.4, cmap="gray")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Shot Noise")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img1, cax=cax1)

    fig.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(diag.shot_noise.ravel(), bins=50)
    plt.xlabel("Photons per pixel")
    plt.ylabel("Frequency")
    plt.title("Photon Shot Noise Distribution")
    plt.show()

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 5))

    vmin, vmax = percentile_limits(diag.shot_noise)
    img0 = ax0.imshow(diag.shot_noise, vmin=vmin, vmax=vmax)
    ax0.imshow(diag.after_shot_noise, alpha=0.4, cmap="gray")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("Photons After Shot Noise")
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img0, cax=cax0)

    vmin, vmax = percentile_limits(diag.electrons)
    img1 = ax1.imshow(diag.electrons, vmin=vmin, vmax=vmax)
    ax1.imshow(diag.after_electrons, alpha=0.4, cmap="gray")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Electrons")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img1, cax=cax1)

    fig.tight_layout()
    plt.show()

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 5))

    vmin, vmax = percentile_limits(diag.electrons)
    img0 = ax0.imshow(diag.electrons, vmin=vmin, vmax=vmax)
    ax0.imshow(diag.after_electrons, alpha=0.4, cmap="gray")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("Electrons In")
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img0, cax=cax0)

    vmin, vmax = percentile_limits(diag.electrons_out)
    img1 = ax1.imshow(diag.electrons_out, vmin=vmin, vmax=vmax)
    ax1.imshow(diag.after_dark_noise, alpha=0.4, cmap="gray")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Electrons Out")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img1, cax=cax1)

    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    diff = diag.electrons_out - diag.electrons
    vmin, vmax = percentile_limits(diff)
    img = ax.imshow(diff, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Electron Difference (Out - In)")
    cb = plt.colorbar(img, ax=ax)
    cb.set_label("Electrons")
    plt.show()

    fig, ax = plt.subplots()
    vmin, vmax = percentile_limits(diag.adu)
    img = ax.imshow(diag.adu, vmin=vmin, vmax=vmax, cmap="plasma")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Final ADU Map")
    cb = plt.colorbar(img, ax=ax)
    cb.set_label("ADU")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate dark noise for a datacube.")
    default_input = Path(__file__).resolve().parents[1] / "ksc512 (1).npy"
    default_output = Path(__file__).resolve().parents[1] / "ksc512_dark_noise.npy"

    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to the input datacube (.npy).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to write the noisy datacube (.npy).",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=100,
        help="Slice index to visualise after simulation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for noise generation.",
    )
    parser.add_argument(
        "--bitdepth",
        type=int,
        default=12,
        help="Sensor bit depth used when clipping ADU values.",
    )
    parser.add_argument(
        "--quantum-efficiency",
        type=float,
        default=0.6,
        help="Quantum efficiency of the sensor.",
    )
    parser.add_argument(
        "--num-photons",
        type=float,
        default=500,
        help="Mean photons per pixel used for the shot-noise model.",
    )
    parser.add_argument(
        "--dark-noise-std",
        type=float,
        default=2.29,
        help="Gaussian dark noise standard deviation in electrons.",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=5.88,
        help="Conversion gain in ADU per electron.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of the before/after slice comparison.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=512,
        help="Threshold applied after each stage to zero affected pixels.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data = np.load(args.input)
    noisy_data, diagnostics = simulate_dark_noise(
        data,
        num_photons=args.num_photons,
        quantum_efficiency=args.quantum_efficiency,
        dark_noise_std=args.dark_noise_std,
        sensitivity=args.sensitivity,
        bitdepth=args.bitdepth,
        seed=args.seed,
        diagnostics_slice=args.slice,
        mask_threshold=args.mask_threshold,
    )

    np.save(args.output, noisy_data)
    print(f"Saved noisy datacube to {args.output}")

    if not args.no_plot:
        slice_index = diagnostics.index if diagnostics else int(
            np.clip(args.slice, 0, noisy_data.shape[2] - 1)
        )
        plot_slice(data, noisy_data, slice_index)
        if diagnostics:
            plot_diagnostics(diagnostics)


if __name__ == "__main__":
    main()
