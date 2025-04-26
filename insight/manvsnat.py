import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt
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
img_path = "A:\\projects\\dev olympus hackathon\\images all sources\\sor1.jpg" 
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(image_rgb)

# --- Generate masks using SAM ---
masks = mask_generator.generate(image_rgb)

# --- Labels for classification ---
labels = ["building", "road", "car", "tree", "grass", "river", "field", "roof", "concrete", "vegetation","bike"]

# Man-made vs Natural
man_made_labels = {"building", "road", "car", "roof", "concrete", "bike"}
natural_labels = {"tree", "grass", "river", "field", "vegetation"}

# --- Start drawing ---
draw_img = pil_img.copy()
draw = ImageDraw.Draw(draw_img)

# --- Process each mask ---
for mask in masks:
    seg = mask["segmentation"]
    x, y, w, h = cv2.boundingRect(seg.astype(np.uint8))

    # Crop the region
    cropped = image_rgb[y:y+h, x:x+w]
    cropped_pil = Image.fromarray(cropped)

    # CLIP processing
    inputs = clip_processor(text=labels, images=cropped_pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    # Get the best matching label
    best_idx = torch.argmax(probs).item()
    best_label = labels[best_idx]

    # Classify and draw box
    color = "red" if best_label in man_made_labels else "green" if best_label in natural_labels else None
    if color:
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        draw.text((x, y), best_label, fill=color)

# --- Save the result ---
draw_img.save("sam_clip_output.jpg")
draw_img.show()
