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
    original_adu: np.ndarray
    expected_photons: np.ndarray
    shot_photons: np.ndarray
    signal_electrons: np.ndarray
    dark_current_electrons: np.ndarray
    read_noise: np.ndarray
    electrons_out: np.ndarray
    final_adu: np.ndarray


def simulate_dark_noise(
    data: np.ndarray,
    quantum_efficiency: float = 0.65,
    read_noise_std: float = 3.0,
    dark_current_rate: float = 0.005,
    exposure_time: float = 0.01,
    sensitivity: float = 5.88,
    bitdepth: int = 16,
    seed: int | None = 42,
    diagnostics_slice: int | None = None,
) -> tuple[np.ndarray, DiagnosticSlice | None]:
    """Apply a sensor-inspired dark noise model and return the noisy datacube."""
    rng = np.random.default_rng(seed)

    max_adu = int(2**bitdepth - 1)
    original = data.astype(np.float32, copy=False)

    # Convert recorded ADU values back to electrons to estimate the incoming signal.
    signal_electrons = np.clip(original / sensitivity, 0.0, None)
    expected_photons = np.where(
        quantum_efficiency > 0,
        signal_electrons / quantum_efficiency,
        0.0,
    )
    expected_photons = np.clip(expected_photons, 0.0, None)

    # Photon shot noise driven by the measured signal.
    shot_photons = rng.poisson(expected_photons)
    electrons_from_signal = shot_photons * quantum_efficiency

    # Dark current accumulates during the exposure and introduces additional electrons.
    mean_dark_electrons = max(dark_current_rate * exposure_time, 0.0)
    dark_current_electrons = rng.poisson(mean_dark_electrons, size=data.shape)

    electrons_total = electrons_from_signal + dark_current_electrons

    # Read noise is well-modelled as Gaussian.
    read_noise = rng.normal(loc=0.0, scale=read_noise_std, size=data.shape)
    electrons_out = np.clip(electrons_total + read_noise, 0.0, None)

    adu = np.clip(electrons_out * sensitivity, 0, max_adu)

    if np.issubdtype(data.dtype, np.integer):
        noisy = adu.astype(data.dtype)
    else:
        noisy = adu.astype(np.float32)

    diagnostics = None
    if diagnostics_slice is not None:
        slice_index = int(np.clip(diagnostics_slice, 0, data.shape[2] - 1))
        diagnostics = DiagnosticSlice(
            index=slice_index,
            original_adu=original[:, :, slice_index].copy(),
            expected_photons=expected_photons[:, :, slice_index].astype(
                np.float32, copy=True
            ),
            shot_photons=shot_photons[:, :, slice_index].astype(np.float32, copy=True),
            signal_electrons=electrons_from_signal[:, :, slice_index].astype(
                np.float32, copy=True
            ),
            dark_current_electrons=dark_current_electrons[:, :, slice_index].astype(
                np.float32, copy=True
            ),
            read_noise=read_noise[:, :, slice_index].astype(np.float32, copy=True),
            electrons_out=electrons_out[:, :, slice_index].astype(
                np.float32, copy=True
            ),
            final_adu=adu[:, :, slice_index].astype(np.float32, copy=True),
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
    """Visualise the intermediate states of the dark noise model."""

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
    ax.imshow(diag.original_adu, alpha=0.4, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Expected Photons (Derived from Signal)")
    cb = plt.colorbar(img, ax=ax)
    cb.set_label("Photons")
    plt.show()

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 5))

    vmin, vmax = percentile_limits(diag.expected_photons)
    img0 = ax0.imshow(diag.expected_photons, vmin=vmin, vmax=vmax, cmap="viridis")
    ax0.imshow(diag.original_adu, alpha=0.4, cmap="gray")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("Expected Photons")
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img0, cax=cax0)

    vmin, vmax = percentile_limits(diag.shot_photons)
    img1 = ax1.imshow(diag.shot_photons, vmin=vmin, vmax=vmax, cmap="viridis")
    ax1.imshow(diag.original_adu, alpha=0.4, cmap="gray")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Photon Shot Noise")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img1, cax=cax1)

    fig.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(diag.shot_photons.ravel(), bins=50)
    plt.xlabel("Photons per pixel")
    plt.ylabel("Frequency")
    plt.title("Photon Shot Noise Distribution")
    plt.show()

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 5))

    vmin, vmax = percentile_limits(diag.signal_electrons)
    img0 = ax0.imshow(diag.signal_electrons, vmin=vmin, vmax=vmax)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("Signal Electrons")
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img0, cax=cax0)

    vmin, vmax = percentile_limits(diag.dark_current_electrons)
    img1 = ax1.imshow(diag.dark_current_electrons, vmin=vmin, vmax=vmax)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Dark Current Electrons")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img1, cax=cax1)

    fig.tight_layout()
    plt.show()

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 5))

    vmin, vmax = percentile_limits(diag.read_noise)
    img0 = ax0.imshow(diag.read_noise, vmin=vmin, vmax=vmax, cmap="coolwarm")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("Read Noise (Electrons)")
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img0, cax=cax0)

    vmin, vmax = percentile_limits(diag.electrons_out)
    img1 = ax1.imshow(diag.electrons_out, vmin=vmin, vmax=vmax)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Electrons After Noise")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img1, cax=cax1)

    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    diff = diag.electrons_out - diag.signal_electrons
    vmin, vmax = percentile_limits(diff)
    img = ax.imshow(diff, vmin=vmin, vmax=vmax, cmap="coolwarm")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Electron Difference (Noise Contribution)")
    cb = plt.colorbar(img, ax=ax)
    cb.set_label("Electrons")
    plt.show()

    fig, ax = plt.subplots()
    vmin, vmax = percentile_limits(diag.final_adu)
    img = ax.imshow(diag.final_adu, vmin=vmin, vmax=vmax, cmap="plasma")
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
        default=16,
        help="Sensor bit depth used when clipping ADU values.",
    )
    parser.add_argument(
        "--quantum-efficiency",
        type=float,
        default=0.65,
        help="Quantum efficiency of the sensor (0-1).",
    )
    parser.add_argument(
        "--read-noise-std",
        type=float,
        default=3.0,
        help="Gaussian read-noise standard deviation in electrons.",
    )
    parser.add_argument(
        "--dark-current-rate",
        type=float,
        default=0.005,
        help="Mean dark current rate in electrons per pixel per second.",
    )
    parser.add_argument(
        "--exposure-time",
        type=float,
        default=0.01,
        help="Exposure time in seconds used to scale dark current.",
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data = np.load(args.input)
    noisy_data, diagnostics = simulate_dark_noise(
        data,
        quantum_efficiency=args.quantum_efficiency,
        read_noise_std=args.read_noise_std,
        dark_current_rate=args.dark_current_rate,
        exposure_time=args.exposure_time,
        sensitivity=args.sensitivity,
        bitdepth=args.bitdepth,
        seed=args.seed,
        diagnostics_slice=args.slice,
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
