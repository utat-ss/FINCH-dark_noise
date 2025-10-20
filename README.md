<img src="img/logo.png" height="128">

# Dark Noise Simulator

The UTAT FINCH dark noise simulator models the signal loss and electronic noise present in hyperspectral cameras. It takes an input datacube (`.npy`) and applies the same Poisson and Gaussian noise pipeline used by the team to generate noisy frames. The resulting datacube is saved back to disk for downstream testing.

<img src="img/utat-logo.png" height="64">

## What Is Dark Noise?
Dark noise represents electrons generated inside the sensor when no photons are present. It combines random photon arrival (shot noise), the camera’s quantum efficiency, and downstream read noise. The simulator models each stage and zeroes saturated pixels so you can explore how different sensor characteristics affect your data.

## Simulator Parameters
All parameters can be supplied as command-line arguments when running `dark_noise_simulator/main.py`.

- `--input`: Path to the input datacube (`.npy`). Default: repository’s `ksc512 (1).npy`.
- `--output`: Destination for the noisy datacube (`.npy`). Default: `ksc512_dark_noise.npy`.
- `--num-photons`: Mean photons per pixel used for the Poisson shot-noise draw. Higher values reduce relative photon noise. Default: `500`.
- `--quantum-efficiency`: Fraction of photons converted to electrons. Lower values amplify shot noise. Default: `0.6`.
- `--dark-noise-std`: Standard deviation of the Gaussian read/dark noise (in electrons). Default: `2.29`.
- `--sensitivity`: Conversion gain from electrons to ADU. Controls brightness and potential saturation. Default: `5.88`.
- `--bitdepth`: Sensor bit depth used to clamp ADU values. Default: `12`.
- `--mask-threshold`: Pixels with counts above this value are zeroed to mimic saturation. Default: `512`.
- `--seed`: Random-number seed for reproducible noise. Default: `42`.
- `--slice`: Spectral slice index displayed in diagnostic plots. Default: `100`.
- `--no-plot`: Add this flag to skip generating diagnostic figures.

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
  --num-photons 600 \
  --quantum-efficiency 0.65 \
  --dark-noise-std 3.0 \
  --sensitivity 6.0
```

Omit any argument to use its default. Remove the `--no-plot` flag (or simply avoid adding it) to view the diagnostic figures that illustrate each noise stage.

## Contributing
Pull requests are welcome. Please open an issue to discuss substantial changes or new sensor models.
