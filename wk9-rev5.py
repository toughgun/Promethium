import torch
import cv2
import torchvision.io as io
from torchvision.io import ImageReadMode
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from skimage import feature
from scipy.ndimage import generic_filter
import multiprocessing


# to calculate score based on mask area
def calculate_mask_area_score(mask):
    # calc's the tamper score based on the percentage of the mask's area
    if mask is None or mask.size == 0:
        return 0.0

    # count non-zero (white) pixels in the mask
    suspect_area = np.count_nonzero(mask)
    total_area = mask.size

    # score is the ratio of suspect area to total area, maxed at 1.0
    score = suspect_area / total_area
    return min(score, 1.0) # ensure it doesn't exceed 1.0 due to any calculation oddity


# to calculate score based on average intensity
def calculate_intensity_score(intensity_map):
    # calc's the tamper score based on the average intensity of the map
    if intensity_map is None or intensity_map.size == 0:
        return 0.0

    # calc avg intensity
    mean_intensity = np.mean(intensity_map)

    # normalize the average intensity to a 0.0 - 1.0 score
    # normalize by 255 (max intensity)
    score = mean_intensity / 255.0
    return min(score, 1.0) # make sure it doesn't exceed 1.0


def canny_display(img):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(script_dir, img)
    img_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    tamper_score = 0.0 # Initialize score

    if img_gray is None:
        print(f"Error: Could not load image at {fpath}. Check file path.")
        return None, 0.0, None

    blurred_img = cv2.GaussianBlur(img_gray, (5, 5), 0)
    low_threshold = 50
    high_threshold = 150
    canny_output = cv2.Canny(blurred_img, low_threshold, high_threshold)

    # canny output is already a binary mask (edges are 255, background is 0)
    canny_mask = canny_output

    # --- Scoring Logic: Canny ---
    tamper_score = calculate_mask_area_score(canny_mask)

    # find contours for visualization
    contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_color_overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # The output image is the one with the contours drawn (GREEN)
    cv2.drawContours(img_color_overlay, contours, -1, (0, 255, 0), 1)

    return canny_mask, tamper_score, cv2.cvtColor(img_color_overlay, cv2.COLOR_BGR2RGB)


def ela_and_canny(img_path, quality=90, scale_factor=15):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, img_path)
    temp_ela_path = os.path.join(script_dir, "temp_ela_contour.jpg")
    tamper_score = 0.0 # init score

    try:
        # load a clean copy 
        img_orig_bgr = cv2.imread(full_path)
        if img_orig_bgr is None:
            print(f"Error: Could not load image at {full_path}.")
            return None, 0.0, None

        # ELA calculations
        cv2.imwrite(temp_ela_path, img_orig_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img_recompressed = cv2.imread(temp_ela_path)
        os.remove(temp_ela_path)

        diff_img = cv2.absdiff(img_orig_bgr, img_recompressed)
        ela_map_bgr = np.clip(diff_img * scale_factor, 0, 255).astype(np.uint8)
        ela_map_gray = cv2.cvtColor(ela_map_bgr, cv2.COLOR_BGR2GRAY)

        # automatic thresholding (otsu)
        otsu_threshold, ela_mask = cv2.threshold(
            ela_map_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"ELA-based Otsu Threshold for Contours: {otsu_threshold:.2f}")

        # --- Scoring Logic: ELA Mask Area ---
        tamper_score = calculate_mask_area_score(ela_mask)

        # find the boundires of the suspect areas
        contours, _ = cv2.findContours(ela_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # draw them on the original image (CYAN)
        # The output image is the one with the contours drawn
        cv2.drawContours(img_orig_bgr, contours, -1, (255, 255, 0), 2) # line thickness of 2

        return ela_mask, tamper_score, cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"An error occurred during ELA contouring: {e}")
        if os.path.exists(temp_ela_path):
            os.remove(temp_ela_path)
        return None, 0.0, None

def lbp_texture(img_path, radius=1, n_points=8, window_size=9):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, img_path)
    tamper_score = 0.0 # init score

    try:
        # load a clean copy
        img_orig_bgr = cv2.imread(full_path)
        img_gray = cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2GRAY)

        if img_orig_bgr is None:
            print(f"Error: Could not load image at {full_path}.")
            return None, 0.0, None

        lbp_map = feature.local_binary_pattern(
            img_gray, P=n_points, R=radius, method='uniform'
        ).astype(np.float32)

        texture_variance_map = generic_filter(lbp_map, np.std, size=window_size)

        max_val = np.max(texture_variance_map)
        if max_val == 0:
            print("Texture variance is zero. No meaningful patterns found.")
            
            # Create a zero mask to avoid errors later
            lbp_mask = np.zeros_like(img_gray, dtype=np.uint8)
            return lbp_mask, 0.0, cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2RGB)

        variance_map_8bit = (texture_variance_map * (255.0 / max_val)).astype(np.uint8)

        # otsu thresholding on variance map
        otsu_threshold, lbp_mask = cv2.threshold(
            variance_map_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"LBP Variance Otsu Threshold for Contours: {otsu_threshold:.2f}")

        # --- Scoring Logic: LBP Mask Area ---
        tamper_score = calculate_mask_area_score(lbp_mask)

        # find contours and draw
        contours, _ = cv2.findContours(lbp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # draw boundaries in red
        # The output image is the one with the contours drawn
        cv2.drawContours(img_orig_bgr, contours, -1, (0, 0, 255), 2)

        return lbp_mask, tamper_score, cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"An error occurred during LBP contouring: {e}")
        # Create a zero mask on error
        if 'img_gray' in locals():
             lbp_mask = np.zeros_like(img_gray, dtype=np.uint8)
        else:
             lbp_mask = None

        return lbp_mask, 0.0, None

# intensity mask for dark areas/shadows
def shadow_localization(img_path, threshold=50):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(script_dir, img_path)
    img_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        print(f"Error: Could not load image for shadow localization at {fpath}.")
        return None

    # pixels with intensity below the threshold are marked as 255 (white)
    # if intensity < threshold, output = 255 (white)
    _, shadow_mask = cv2.threshold(
        img_gray, threshold, 255, cv2.THRESH_BINARY_INV
    )
    
    # calculate a score based on the area of the detected dark regions
    shadow_score = calculate_mask_area_score(shadow_mask)
    
    print(f"Shadow Localization Threshold: {threshold}. Area Score: {shadow_score:.4f}")

    return shadow_mask, shadow_score

# combination logic remains the same
def combined_mask_logic(canny_mask, ela_mask, lbp_mask):

    # make sure all masks exist and are the same shape
    # otherwise return empty masks
    if canny_mask is None or ela_mask is None or lbp_mask is None:
        print("Error: One or more masks were not generated. Cannot combine.")
        return None, None

    if canny_mask.shape != ela_mask.shape or canny_mask.shape != lbp_mask.shape:
        print("Error: Masks have incompatible shapes. Cannot combine.")
        return None, None

    # overlay mask (Logical OR / Addition):
    combined_sum = canny_mask.astype(np.float32) + ela_mask.astype(np.float32) + lbp_mask.astype(np.float32)

    # normalize the sum map for visualization (max 3 votes = 255)
    normalized_overlay_mask = np.clip(combined_sum / 3.0, 0, 255).astype(np.uint8)

    # consensus mask (logical AND):
    # Finds areas where AT LEAST 2 methods agree (higher confidence)
    consensus_mask = np.zeros_like(canny_mask, dtype=np.uint8)
    consensus_mask[combined_sum >= (2 * 255)] = 255

    return normalized_overlay_mask, consensus_mask

# get the original image for the final plot
def load_original_image(img_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(script_dir, img_path)
    img_orig_bgr = cv2.imread(fpath)
    if img_orig_bgr is None:
        return None
    return cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2RGB)


# combine and display all results, now including shadow mask
def combined_display(img_orig, canny_result, ela_result, lbp_result, shadow_mask, shadow_score, overlay_mask, consensus_mask, final_score):

    canny_output, canny_score, canny_img = canny_result
    ela_mask, ela_score, ela_img = ela_result
    lbp_mask, lbp_score, lbp_img = lbp_result

    # 2x4 grid to show original, 4 Masks, 2 combined results
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Image Tampering & Shadow Detection | Combined AVG Tamper Score: {final_score:.4f}", fontsize=16)

    # Original Image
    plt.subplot(2, 4, 1)
    if img_orig is not None:
        plt.imshow(img_orig)
        plt.title('Original Image (RGB)')
    else:
        plt.title('Original Image (Error)')
    plt.axis('off')

    # Canny Edge Map
    plt.subplot(2, 4, 2)
    if canny_output is not None:
        plt.imshow(canny_output, cmap='gray')
        plt.title(f'Canny Mask ({canny_score:.4f})')
    else:
        plt.title('Canny Mask (Error)')
    plt.axis('off')

    # ELA Binary Mask
    plt.subplot(2, 4, 3)
    if ela_mask is not None:
        plt.imshow(ela_mask, cmap='gray')
        plt.title(f'ELA Mask ({ela_score:.4f})')
    else:
        plt.title('ELA Mask (Error)')
    plt.axis('off')

    # LBP Suspect Mask
    plt.subplot(2, 4, 4)
    if lbp_mask is not None:
        plt.imshow(lbp_mask, cmap='gray')
        plt.title(f'LBP Mask ({lbp_score:.4f})')
    else:
        plt.title('LBP Mask (Error)')
    plt.axis('off')

    # combined localization results

    # shadow/Dark Area Mask
    plt.subplot(2, 4, 5)
    if shadow_mask is not None:
        plt.imshow(shadow_mask, cmap='gray')
        plt.title(f'Shadow Mask (T=50) ({shadow_score:.4f})')
    else:
        plt.title('Shadow Mask (Error)')
    plt.axis('off')

    # combined overlay Mask (weighted sum)
    plt.subplot(2, 4, 6)
    if overlay_mask is not None:
        plt.imshow(overlay_mask, cmap='hot') # show intensity (1, 2, or 3 votes)
        plt.title('Tamper Suspect Map heat map')
    else:
        plt.title('Tamper Suspect Map (Error)')
    plt.axis('off')

    # consensus Mask and Logic
    plt.subplot(2, 4, 7)
    if consensus_mask is not None:
        plt.imshow(consensus_mask, cmap='gray')
        plt.title('Consensus Tamper Mask')
    else:
        plt.title('Consensus Tamper Mask (Error)')
    plt.axis('off')

    # shadow mask overlay
    plt.subplot(2, 4, 8)
    if shadow_mask is not None and img_orig is not None:

        # create a blended image with a blue overlay on the original image
        blended = img_orig.copy()

        shadow_mask_3ch = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)

        # set mask color
        blue = np.array([0, 0, 255], dtype=np.uint8) # set color to blue

        condition_3d = (shadow_mask_3ch[:,:,0] == 255)[:, :, np.newaxis]

        blended = np.where(condition_3d, blue, blended)

        # blend with original image
        alpha = 0.5
        final_overlay = cv2.addWeighted(img_orig, 1 - alpha, blended, alpha, 0)

        plt.imshow(final_overlay)
        plt.title('Shadow Localization in Blue')
    else:
        plt.title('Shadow Overlay (Error)')

    plt.axis('off')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.show()

# will run the detection and store the result in the shared dictionary.

def run_canny(img, shared_scores):
    _, score, _ = canny_display(img)
    shared_scores['canny'] = score

def run_ela_canny(img, shared_scores):
    _, score, _ = ela_and_canny(img)
    shared_scores['ela-canny'] = score

def run_lbp(img, shared_scores):
    _, score, _ = lbp_texture(img)
    shared_scores['lbp'] = score


def multi_run(tampered):
    # this function is not fully updated to handle the new shadow logic 
    # and is only used for the 'debug' path. Leaving it as is.
    
    # use a manager to create a shared dictionary for communication
    manager = multiprocessing.Manager()
    shared_scores = manager.dict()

    # define processes with the new wrapper functions and shared dictionary
    p_canny = multiprocessing.Process(
        target=run_canny,
        args=(tampered, shared_scores)
    )

    p_ela_contour = multiprocessing.Process(
        target=run_ela_canny,
        args=(tampered, shared_scores)
    )

    p_lbp = multiprocessing.Process(
        target=run_lbp,
        args=(tampered, shared_scores)
    )

    # only using defined functions
    processes = [p_canny, p_ela_contour, p_lbp]

    # start and join processes
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    # Calc Final Score
    scores = [s for s in shared_scores.values() if isinstance(s, (int, float))]

    if not scores:
        print("Error: No scores were collected from the concurrent processes.")
        return 0.0, shared_scores

    average_score = sum(scores) / len(scores)

    return average_score, shared_scores


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python wk9.py <tampered_image> [method]")
        sys.exit(1)

    tampered_file = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) == 3 else "def" # Defaults to 'def' if not specified
    totalscore = [0.0000, 0.0000, 0.0000]

    if method == 'def':
        # run all methods sequentially and collect results

        # canny
        print("[1/4] Canny edge detection.")
        canny_mask, canny_score, canny_img = canny_display(tampered_file)
        print(f"--- Canny Final Tamper Score: {canny_score:.4f} ---")
        totalscore[0] = canny_score
        canny_result = (canny_mask, canny_score, canny_img)

        # ela
        print("\n[2/4] ELA with contour mapping.")
        ela_mask, ela_score, ela_img = ela_and_canny(tampered_file)
        print(f"--- ELA Contour Final Tamper Score: {ela_score:.4f} ---")
        totalscore[1] = ela_score
        ela_result = (ela_mask, ela_score, ela_img)

        # lbp
        print("\n[3/4] Local Binary Patterns (LBP).")
        lbp_mask, lbp_score, lbp_img = lbp_texture(tampered_file)
        print(f"--- LBP Final Tamper Score: {lbp_score:.4f} ---")
        totalscore[2] = lbp_score
        lbp_result = (lbp_mask, lbp_score, lbp_img)
        
        # calculate tamper score (avg)
        final_avg_score = sum(totalscore) / 3.0
        print(f"\n--- Final Tamper [Texture] Score: {final_avg_score:.4f} ---")
        
        # shadow localization
        print("\n[4/4] Shadow Localization.")
        shadow_mask, shadow_score = shadow_localization(tampered_file, threshold=50)
        
        # generate the combined localization masks (based on Canny, ELA, LBP)
        overlay_mask, consensus_mask = combined_mask_logic(canny_mask, ela_mask, lbp_mask)

        # load the original image once for the plot
        img_orig = load_original_image(tampered_file)

        # call the new combined display function
        combined_display(
            img_orig, 
            canny_result, 
            ela_result, 
            lbp_result, 
            shadow_mask, 
            shadow_score, 
            overlay_mask, 
            consensus_mask, 
            final_avg_score
        )

    elif method == 'debug':
        print("Starting all processes...")

        # run all methods in parallel and get the average score
        final_score, individual_scores = multi_run(tampered_file)

        print("\n--- Individual Method Scores ---")
        for method_name, score in individual_scores.items():
            print(f"{method_name.ljust(10)}: {score:.4f}")

        print(f"\n--- Final Tamper [Texture] Score: {final_score:.4f} ---")

    else:
        raise ValueError(f"Invalid processing method: {method}. Choose 'def' or 'debug'.")

if __name__ == '__main__':
    main()