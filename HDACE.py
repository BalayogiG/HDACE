# Required Packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from math import log2, log10, sqrt

def compute_hessian_eigen_map(img):
    """
    Computes the Hessian eigenvalue map for a grayscale image.
    
    For each pixel, this function computes the 2D Hessian matrix using second-order
    partial derivatives and extracts the dominant eigenvalue (in absolute magnitude).
    The result is a map highlighting regions with strong curvature, commonly useful 
    in feature detection, vessel enhancement, or edge analysis.

    Parameters:
        img (np.ndarray): 2D grayscale input image.

    Returns:
        eigen_map (np.ndarray): 2D map of the dominant Hessian eigenvalue at each pixel.
    """

    # Compute second-order partial derivatives using Sobel filters.
    I_xx = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=3)  # d²I/dx²
    I_yy = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=3)  # d²I/dy²
    I_xy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)  # d²I/dxdy

    # Compute the discriminant (under the square root) of the eigenvalue formula.
    temp = np.sqrt((I_xx - I_yy) ** 2 + 4 * (I_xy ** 2))

    # Eigenvalues of the 2x2 Hessian matrix at each pixel.
    lambda1 = (I_xx + I_yy + temp) / 2.0
    lambda2 = (I_xx + I_yy - temp) / 2.0

    # Select the eigenvalue with maximum absolute magnitude (dominant curvature).
    eigen_map = np.maximum(np.abs(lambda1), np.abs(lambda2))

    return eigen_map

def generate_chaotic_sequence(x0, length, mu=3.99):
    """
    Generates a chaotic pseudo-random sequence using the logistic map.

    The logistic map is defined as: x_{n+1} = mu * x_n * (1 - x_n),
    which exhibits chaotic behavior for mu in the range (3.57, 4).
    This function scales and quantizes the sequence to fit within 8-bit range [0, 255],
    suitable for applications such as encryption, watermarking, or randomized indexing.

    Parameters:
        x0 (float): Initial seed value in the range (0, 1). Must not be 0 or 1.
        length (int): Length of the output chaotic sequence.
        mu (float): Logistic map control parameter. Default is 3.99 (chaotic regime).

    Returns:
        chaotic_seq (np.ndarray): Array of uint8 values representing the chaotic sequence.
    """

    # Initialize sequence storage
    seq = np.zeros(length)
    x = x0

    # Iteratively apply logistic map
    for i in range(length):
        x = mu * x * (1 - x)
        seq[i] = x

    # Scale to [0, 255], apply floor and modulo to ensure valid uint8 range
    chaotic_seq = np.floor(seq * 256) % 256

    return chaotic_seq.astype(np.uint8)


def encrypt_single_channel(img_gray, secret_key=0.5):
    """
    Encrypts a single-channel grayscale image using Hessian-based permutation and
    chaotic diffusion.

    The encryption pipeline follows two primary stages:
    1. Permutation: Pixels are reordered based on the ascending order of their
       corresponding normalized Hessian eigenvalues, creating spatial disarrangement
       informed by image structure.
    2. Diffusion: The permuted pixel values are XOR-ed with a chaotic sequence
       generated from a logistic map initialized with a secret-derived seed.

    Parameters:
        img_gray (np.ndarray): 2D grayscale image array of dtype uint8.
        secret_key (float): A user-defined key (0 < key < 1) that influences the
                            chaotic seed and thereby the encryption process.

    Returns:
        cipher_img (np.ndarray): Encrypted image with the same shape as `img_gray`.
        aux_data (dict): Dictionary containing metadata required for decryption:
            - 'perm_indices': Order of pixel permutation.
            - 'seed': Initial seed used for the chaotic sequence.
            - 'shape': Original image shape.
    """

    # Step 1: Compute dominant Hessian eigenvalue map to inform spatial permutation
    eigen_map = compute_hessian_eigen_map(img_gray)

    # Step 2: Normalize the eigenvalue map to the [0, 1] range for fair ranking
    eigen_map_norm = cv2.normalize(eigen_map, None, alpha=0, beta=1,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    # Step 3: Flatten and sort by normalized eigenvalues to derive pixel permutation
    flat_eigen = eigen_map_norm.flatten()
    perm_indices = np.argsort(flat_eigen)

    # Step 4: Apply permutation to the image's flattened intensity values
    flat_img = img_gray.flatten()
    permuted_flat_img = flat_img[perm_indices]

    # Step 5: Derive chaotic seed from image structure and secret key
    seed = (np.mean(eigen_map_norm) + secret_key) % 1.0
    if seed == 0:
        seed = 0.1  # Avoid degenerate logistic map behavior

    # Step 6: Generate a chaotic sequence for pixel-wise diffusion
    chaotic_seq = generate_chaotic_sequence(seed, flat_img.size)

    # Step 7: Apply XOR-based diffusion to permuted pixel values
    diffused_flat = np.bitwise_xor(permuted_flat_img, chaotic_seq)

    # Step 8: Reshape the encrypted flat array to original image dimensions
    cipher_img = diffused_flat.reshape(img_gray.shape)

    # Step 9: Return encrypted image and auxiliary data for decryption
    aux_data = {
        'perm_indices': perm_indices,
        'seed': seed,
        'shape': img_gray.shape
    }
    return cipher_img, aux_data

def decrypt_single_channel(cipher_img, aux_data):
    """
    Decrypts a single-channel grayscale image previously encrypted using 
    Hessian-based permutation and chaotic diffusion.

    This function performs the inverse operations of:
    1. Diffusion: Reverses the XOR operation with the identical chaotic sequence.
    2. Permutation: Restores the original spatial arrangement using the stored permutation indices.

    Parameters:
        cipher_img (np.ndarray): 2D encrypted grayscale image of dtype uint8.
        aux_data (dict): Dictionary containing metadata from encryption:
            - 'perm_indices': Original permutation indices used during encryption.
            - 'seed': Seed for generating the chaotic sequence.
            - 'shape': Original image shape before encryption.

    Returns:
        plain_img (np.ndarray): Decrypted grayscale image matching the original input.
    """

    # Step 1: Extract decryption metadata
    perm_indices = aux_data['perm_indices']
    seed = aux_data['seed']
    shape = aux_data['shape']

    # Step 2: Flatten the cipher image to apply inverse operations
    flat_cipher = cipher_img.flatten()

    # Step 3: Regenerate the identical chaotic sequence used during encryption
    chaotic_seq = generate_chaotic_sequence(seed, flat_cipher.size)

    # Step 4: Undo diffusion by XOR-ing with the chaotic sequence
    permuted_flat_img = np.bitwise_xor(flat_cipher, chaotic_seq)

    # Step 5: Recover original pixel ordering via inverse permutation
    inv_perm = np.argsort(perm_indices)
    flat_img = permuted_flat_img[inv_perm]

    # Step 6: Reshape back to the original image dimensions
    plain_img = flat_img.reshape(shape)

    return plain_img

def encrypt_image(img, secret_key=0.5):
    """
    Encrypts an image (grayscale or RGB).
    """
    if len(img.shape) == 2:
        cipher_img, aux_data = encrypt_single_channel(img, secret_key)
        return cipher_img, [aux_data]
    elif len(img.shape) == 3:
        channels = cv2.split(img)
        encrypted_channels = []
        aux_data_list = []
        for ch in channels:
            encrypted_ch, aux_data = encrypt_single_channel(ch, secret_key)
            encrypted_channels.append(encrypted_ch)
            aux_data_list.append(aux_data)
        return cv2.merge(encrypted_channels), aux_data_list


def decrypt_image(cipher_img, aux_data_list):
    """
    Decrypts an image (grayscale or RGB).
    """
    if len(cipher_img.shape) == 2:
        return decrypt_single_channel(cipher_img, aux_data_list[0])
    elif len(cipher_img.shape) == 3:
        channels = cv2.split(cipher_img)
        decrypted_channels = []
        for i in range(3):
            decrypted_ch = decrypt_single_channel(channels[i], aux_data_list[i])
            decrypted_channels.append(decrypted_ch)
        return cv2.merge(decrypted_channels)

def plot_histograms(original, encrypted, decrypted):
    """
    Plots the histograms of the original, encrypted, and decrypted images.
    """
    plt.figure(figsize=(18, 5))

    # Histogram for Original Image
    plt.subplot(1, 3, 1)
    plt.hist(original.ravel(), bins=256, range=[0,256], color='blue', alpha=0.7)
    plt.title("Histogram of Original Image")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # Histogram for Encrypted Image
    plt.subplot(1, 3, 2)
    plt.hist(encrypted.ravel(), bins=256, range=[0,256], color='red', alpha=0.7)
    plt.title("Histogram of Encrypted Image")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # Histogram for Decrypted Image
    plt.subplot(1, 3, 3)
    plt.hist(decrypted.ravel(), bins=256, range=[0,256], color='green', alpha=0.7)
    plt.title("Histogram of Decrypted Image")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    img = cv2.imread('Lenna.png')
    if img is None:
        print("Error: Image file not found.")
    else:
        secret_key = 0.5
        cipher_img, aux_data_list = encrypt_image(img, secret_key)
        decrypted_img = decrypt_image(cipher_img, aux_data_list)

        # Convert BGR to RGB for proper display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dec_img_rgb = cv2.cvtColor(decrypted_img, cv2.COLOR_BGR2RGB)

        # Display the images
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(img_rgb)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cipher_img, cmap='gray')
        plt.savefig("encrypted_img.png")
        plt.title("Encrypted Image")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(dec_img_rgb)
        plt.title("Decrypted Image")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

plot_histograms(img, cipher_img, decrypted_img)
