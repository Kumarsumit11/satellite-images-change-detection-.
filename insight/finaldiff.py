from PIL import Image
import cv2
import numpy as np

# === Step 1: Resize two images and save ===
def resize_images(image_path1, image_path2):
    img1 = Image.open('A:\\projects\\dev olympus hackathon\\images all sources\\img1.jpg').convert("RGB")
    img2 = Image.open('A:\\projects\\dev olympus hackathon\\images all sources\\img2.jpg').convert("RGB")

    width = min(img1.width, img2.width)
    height = min(img1.height, img2.height)

    img1 = img1.resize((width, height))
    img2 = img2.resize((width, height))

    img1.save("resized_image1.jpg")
    img2.save("resized_image2.jpg")
    print("Images resized and saved.")
    return "resized_image1.jpg", "resized_image2.jpg"

# === Step 2: Generate simple difference image ===
def create_difference_image(path1, path2, output_path="difference.jpg"):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    diff = cv2.absdiff(img1, img2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, thresh_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, thresh_diff)
    print("Difference image saved.")
    return output_path

# === Step 3: Overlay red highlights where changes were detected ===
def highlight_changes(base_image_path, diff_mask_path, output_path="highlighted_changes.jpg"):
    base = cv2.imread(base_image_path)
    mask = cv2.imread(diff_mask_path, cv2.IMREAD_GRAYSCALE)

    _, binary_mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

    red_overlay = np.zeros_like(base)
    red_overlay[:] = [0, 0, 255]  # Red in BGR

    blended = cv2.addWeighted(base, 1.0, red_overlay, 0.5, 0)
    result = base.copy()
    result[binary_mask == 255] = blended[binary_mask == 255]

    cv2.imwrite(output_path, result)
    print("Highlighted image saved.")
    return output_path

# === Run Everything ===
if __name__ == "__main__":
    # Replace these with your actual image paths
    img1_path = r"A:\projects\dev olympus hackathon\images all sources\img1.jpg"
    img2_path = r"A:\projects\dev olympus hackathon\images all sources\img2.jpg"

    resized1, resized2 = resize_images(img1_path, img2_path)
    diff_path = create_difference_image(resized1, resized2)
    highlighted_path = highlight_changes(resized1, diff_path)

    print(f"\n🎯 Done! Open this file to see the result: {highlighted_path}")
