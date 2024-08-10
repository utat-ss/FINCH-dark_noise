# Author: Shivesh Prakash
# This file subtracts dark frames from the datacube, a way to correct for dark noise

import numpy as np

def calculate_master_dark(dark_frames):
    """Calculate the average dark frame."""
    return np.mean(dark_frames, axis=0)

def optimize_dark_frame_subtraction(datacube, master_dark):
    """Optimize the dark frame subtraction using entropy minimization."""
    best_entropy = float('inf')
    best_k = 1.0
    k_values = np.linspace(0.8, 1.2, 50)  # Read more about the optimal scaling factor

    for k in k_values:
        corrected_datacube = datacube - k * master_dark
        entropy = calculate_entropy(corrected_datacube)
        if entropy < best_entropy:
            best_entropy = entropy
            best_k = k

    return best_k

def calculate_entropy(image):
    """Calculate the entropy of an image."""
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    histogram = histogram / histogram.sum()
    entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
    return entropy

def df_subtract(datacube, dark_frames):
    # Calculate the master dark frame
    master_dark = calculate_master_dark(dark_frames)

    # Optimize the dark frame subtraction
    optimal_k = optimize_dark_frame_subtraction(datacube, master_dark)

    # Subtract the optimized dark frame from the datacube
    corrected_datacube = datacube - optimal_k * master_dark

    return corrected_datacube