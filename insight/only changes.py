#y vaala rkhna h
from PIL import Image
import cv2
import numpy as np

# === STEP 1: Resize two images to match ===
def resize_images(image_path1, image_path2):
    img1 = Image.open('A:\\projects\\dev olympus hackathon\\images all sources\\sat6.jpg').convert("RGB")
    img2 = Image.open('A:\\projects\\dev olympus hackathon\\images all sources\\sat7.jpg').convert("RGB")

    width = min(img1.width, img2.width)
    height = min(img1.height, img2.height)

    img1 = img1.resize((width, height))
    img2 = img2.resize((width, height))

    img1.save("resized_image1.jpg")
    img2.save("resized_image2.jpg")
    print("✅ Images resized and saved.")
    return "resized_image1.jpg", "resized_image2.jpg"

# === STEP 2: Generate difference mask ===
def create_difference_image(path1, path2, output_path="difference.jpg"):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    diff = cv2.absdiff(img1, img2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, thresh_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, thresh_diff)
    print("✅ Difference mask saved.")
    return output_path

# === STEP 3a: Highlight changes in red (overlay style) ===
def highlight_changes_with_contours(base_image_path, diff_mask_path, output_path="highlighted_better.jpg"):
    base = cv2.imread(base_image_path)
    mask = cv2.imread(diff_mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure proper type
    binary_mask = mask.astype(np.uint8)
    
    # Threshold to get binary change mask
    _, binary_mask = cv2.threshold(binary_mask, 30, 255, cv2.THRESH_BINARY)

    # Find contours (compatible with OpenCV 3 & 4)
    contours_info = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    # Draw red contours on top of base image
    highlighted = base.copy()
    cv2.drawContours(highlighted, contours, -1, (0, 0, 255), thickness=2)

    # Optional: blend with original
    result = cv2.addWeighted(base, 0.9, highlighted, 0.4, 0)

    cv2.imwrite(output_path, result)
    print("✅ Cleaner contour-highlighted image saved.")
    return output_path



# === STEP 3b: Only show changed parts (everything else black) ===
def highlight_changes_only(base_image_path, diff_mask_path, output_path="only_changes.jpg"):
    base = cv2.imread(base_image_path)
    mask = cv2.imread(diff_mask_path, cv2.IMREAD_GRAYSCALE)

    _, binary_mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)

    changes_only = np.zeros_like(base)
    changes_only[binary_mask == 255] = base[binary_mask == 255]

    cv2.imwrite(output_path, changes_only)
    print("✅ Image with only changed parts saved.")
    return output_path

# === RUN EVERYTHING ===
# === RUN EVERYTHING ===
if __name__ == "__main__":
    img1_path = r"A:\projects\dev olympus hackathon\images all sources\sat6.jpg"
    img2_path = r"A:\projects\dev olympus hackathon\images all sources\sat7.jpg"

    resized1, resized2 = resize_images(img1_path, img2_path)
    diff_path = create_difference_image(resized1, resized2)
    highlight_changes_with_contours(resized1, diff_path)  # ✅ This is the correct call now
    highlight_changes_only(resized1, diff_path)

    print("\n🎉 All outputs generated successfully!")

