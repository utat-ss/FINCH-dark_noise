import cv2
import numpy as np
import matplotlib.pyplot as plt

#used python 3.11

def per_pixel_snr_histogram(original_img, noisy_img):
    # Convert to grayscale
    original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    noisy = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    noise = noisy - original

    signal_power = original ** 2
    noise_power = noise ** 2

    with np.errstate(divide='ignore', invalid='ignore'):
        snr_map = 10 * np.log10(signal_power / noise_power)
        snr_map = np.nan_to_num(snr_map, nan=0.0, posinf=0.0, neginf=0.0)

    # Flatten and plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(snr_map.ravel(), bins=100, color='purple', alpha=0.75)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Per-Pixel SNR")
    plt.grid(True)
    plt.show()

    return snr_map

# Load images
original = cv2.imread("original_image.jpeg")
noisy = cv2.imread("noisy_image.jpeg")

snr_map = per_pixel_snr_histogram(original, noisy)

#python snr_histogram.py
