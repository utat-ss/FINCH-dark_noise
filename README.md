<img src="img/logo.png" height="128">

# Dark Noise Simulator

The UTAT FINCH dark noise simulator models the signal loss and electronic noise present in hyperspectral cameras. It takes an input datacube (`.npy`) and passes it through a Poisson + Gaussian sensor model to generate a realistic noisy frame. The resulting datacube is saved back to disk for downstream testing or algorithm benchmarking.

<img src="img/utat-logo.png" height="64">

## What Is Dark Noise?
Dark noise represents electrons generated inside the sensor when no photons are present. It combines random photon arrival (shot noise), the camera’s quantum efficiency, thermally generated dark current, and downstream read noise. The simulator models each stage without artificially zeroing pixels so you can explore how different sensor characteristics affect your data.

## Simulator Parameters
All parameters can be supplied as command-line arguments when running `dark_noise_simulator/main.py`.

- `--input`: Path to the input datacube (`.npy`). Default: repository’s `ksc512 (1).npy`.
- `--output`: Destination for the noisy datacube (`.npy`). Default: `ksc512_dark_noise.npy`.
- `--quantum-efficiency`: Fraction of photons converted to electrons. Lower values amplify shot noise. Default: `0.65`.
- `--read-noise-std`: Standard deviation of the Gaussian read noise (electrons RMS). Default: `3.0`.
- `--dark-current-rate`: Dark current generation rate in electrons per pixel per second. Default: `0.005`.
- `--exposure-time`: Exposure duration in seconds used to scale dark current. Default: `0.01`.
- `--sensitivity`: Conversion gain from electrons to ADU. Controls brightness and potential saturation. Default: `5.88`.
- `--bitdepth`: Sensor bit depth used to clamp ADU values. Default: `16`.
- `--seed`: Random-number seed for reproducible noise. Default: `42`.
- `--slice`: Spectral slice index displayed in diagnostic plots. Default: `100`.
- `--no-plot`: Add this flag to skip generating diagnostic figures.

## Noise Model Details
The simulator follows a simple sensor-inspired pipeline. For every pixel in the input datacube:

1. **Signal estimate** – Convert the recorded digital number (DN/ADU) back to electrons using the gain `sensitivity` (`signal_e = ADU / sensitivity`). This serves as an estimate of the photo-electrons collected during exposure.
2. **Photon back-casting** – Infer the expected photon count by dividing by the quantum efficiency `QE` (`μ_ph = signal_e / QE`). Because `QE` < 1, this inflates the mean to reflect photons lost to imperfect conversion.
3. **Photon shot noise** – Draw the incident photon count from `Poisson(μ_ph)` to model the randomness of photon arrivals. Multiply by `QE` to obtain the electrons created from the signal.
4. **Dark current** – Dark current electrons accumulate even in the absence of light. We draw them from `Poisson(dark_current_rate × exposure_time)` for each pixel and add them to the signal electrons.
5. **Read noise** – Add zero-mean Gaussian noise with standard deviation `read_noise_std`. This captures downstream electronic noise in the sensor circuitry.
6. **Clamp and quantize** – Clip negative electron counts to zero, convert back to ADU through multiplication by `sensitivity`, and clamp to the configured `bitdepth` (`0 … 2^bitdepth − 1`).

The statistics of each intermediate quantity are optionally exported through the diagnostic plots (enable them by omitting `--no-plot`). These figures show the inferred photon map, shot-noise distribution, dark current contribution, read-noise map, electron deltas, and final ADU frame for the selected spectral slice.

## Setup
1. Install Python 3.10 from the [official downloads](https://www.python.org/downloads/).
2. Clone this repository and open a terminal in its root directory.
3. (Optional) Create and activate a virtual environment:
   - `python -m venv .venv`
   - On macOS/Linux: `source .venv/bin/activate`
   - On Windows: `.venv\Scripts\activate`
4. Install runtime dependencies with `pip install -r requirements.txt`.

## Running the Simulator
To simulate dark noise and save a new datacube:

```bash
python dark_noise_simulator/main.py \
  --input "ksc512 (1).npy" \
  --output noisy_cube.npy \
  --quantum-efficiency 0.65 \
  --read-noise-std 4.0 \
  --dark-current-rate 0.01 \
  --exposure-time 0.02
```

Omit any argument to use its default. Remove the `--no-plot` flag (or simply avoid adding it) to view the diagnostic figures that illustrate each noise stage.

## Contributing
Pull requests are welcome. Please open an issue to discuss substantial changes or new sensor models.
