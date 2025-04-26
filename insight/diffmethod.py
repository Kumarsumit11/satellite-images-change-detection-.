from PIL import Image, ImageChops
import numpy as np

# Load and convert images
img1 = Image.open("A:\\projects\\dev olympus hackathon\\images all sources\\sat6.jpg").convert("RGB")
img2 = Image.open("A:\\projects\\dev olympus hackathon\\images all sources\\sat7.jpg").convert("RGB")

# Compute the difference
diff = ImageChops.difference(img1, img2)

# Convert to numpy arrays
diff_np = np.array(diff)

# Set a stricter threshold to reduce background noise
threshold = 80

# Create a boolean mask of pixels where any channel exceeds the threshold
mask = np.any(diff_np > threshold, axis=2)

# Create an empty black image
output_np = np.zeros_like(diff_np)

# Apply the mask: only keep changed pixels
output_np[mask] = diff_np[mask]

# Convert back to image
changed_only_img = Image.fromarray(output_np)

# Show or save the result
changed_only_img.show()
# changed_only_img.save("A:\\projects\\dev olympus hackathon\\output\\diff_only.png")
