import tensorflow as tf
import cv2
import os

#used python 3.11
#used tensorflow 2.14.0
#used opcencev 4.7.0

def preprocess_grayscale_image(image):
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    return tf.expand_dims(tf.expand_dims(tensor, axis=0), axis=-1)

def compute_ssim_tf(image_path1, image_path2):
    # Check if files exist before loading
    if not os.path.exists(image_path1):
        raise FileNotFoundError(f"File not found: {image_path1}")
    if not os.path.exists(image_path2):
        raise FileNotFoundError(f"File not found: {image_path2}")

    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("Could not load one or both images.")

    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    img1_tensor = preprocess_grayscale_image(img1)
    img2_tensor = preprocess_grayscale_image(img2)


    #Testing images
    ssim_score = tf.image.ssim(img1_tensor, img2_tensor, max_val=255.0)
    return float(ssim_score.numpy()) 

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())

    try:
        score = compute_ssim_tf("noisy_image.jpeg", "denoised_image.jpeg")  
        print("TensorFlow SSIM Score:", score)
    except Exception as e:
        print("Error:", e)