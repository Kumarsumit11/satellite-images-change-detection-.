from PIL import Image, ImageChops
import numpy as np


img1 = Image.open("A:\\projects\\dev olympus hackathon\\images all sources\\sat6.jpg").convert("RGB")
img2 = Image.open("A:\\projects\\dev olympus hackathon\\images all sources\\sat7.jpg").convert("RGB")


diff = ImageChops.difference(img1, img2)


diff_np = np.array(diff)


threshold = 80


mask = np.any(diff_np > threshold, axis=2)


output_np = np.zeros_like(diff_np)


output_np[mask] = diff_np[mask]


changed_only_img = Image.fromarray(output_np)


changed_only_img.show()
