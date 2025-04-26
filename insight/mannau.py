import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- Load CLIP model ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Load SAM model ---
sam_checkpoint = "sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to("cuda" if torch.cuda.is_available() else "cpu")
mask_generator = SamAutomaticMaskGenerator(sam)

# --- Load and prepare image ---
img_path = "A:\\projects\\dev olympus hackathon\\images all sources\\sat3.jpg"
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(image_rgb)

# --- Generate masks ---
masks = mask_generator.generate(image_rgb)

# --- Labels ---
labels = ["building", "road", "car", "tree", "grass", "river", "field", "roof", "concrete", "vegetation", "shadows"]
man_made_labels = {"building", "road", "car", "roof", "concrete"}
natural_labels = {"tree", "grass", "river", "field", "vegetation"}

# --- Create two separate images ---
man_made_img = pil_img.copy()
natural_img = pil_img.copy()
draw_man = ImageDraw.Draw(man_made_img)
draw_nat = ImageDraw.Draw(natural_img)

# --- Process masks ---
for mask in masks:
    seg = mask["segmentation"]
    x, y, w, h = cv2.boundingRect(seg.astype(np.uint8))
    cropped = image_rgb[y:y+h, x:x+w]
    cropped_pil = Image.fromarray(cropped)

    # CLIP processing
    inputs = clip_processor(text=labels, images=cropped_pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    # Best label
    best_idx = torch.argmax(probs).item()
    best_label = labels[best_idx]

    # Draw separately
    if best_label in man_made_labels:
        draw_man.rectangle([x, y, x + w, y + h], outline="red", width=2)
        draw_man.text((x, y), best_label, fill="red")
    elif best_label in natural_labels:
        draw_nat.rectangle([x, y, x + w, y + h], outline="green", width=2)
        draw_nat.text((x, y), best_label, fill="green")

# --- Save both images ---
man_made_img.save("output_man_made.jpg")
natural_img.save("output_natural.jpg")

man_made_img.show()
natural_img.show()
