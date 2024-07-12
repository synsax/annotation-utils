import os
import random
from pycocotools.coco import COCO
import numpy as np
from PIL import Image, ImageDraw


def get_high_contrast_color(image):
    # Convert image to RGB mode
    rgb_image = image.convert("RGB")

    # Get image data as numpy array
    img_data = np.array(rgb_image)

    # Calculate average brightness
    avg_brightness = np.mean(img_data)

    # Choose a colorful high-contrast color based on average brightness
    if avg_brightness < 85:
        return (255, 0, 0)  # Bright red for very dark images
    elif avg_brightness < 170:
        return (0, 255, 0)  # Bright green for medium-dark images
    else:
        return (0, 0, 255)  # Bright blue for bright images


def draw_bbox(draw, ann, color):
    bbox = ann["bbox"]
    draw.rectangle(
        [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], outline=color, width=2
    )


def draw_polygon(draw, ann, color):
    for seg in ann["segmentation"]:
        poly = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
        draw.polygon(poly, outline=color, width=2)


def draw_keypoints(draw, ann, color):
    if "keypoints" in ann:
        keypoints = np.array(ann["keypoints"]).reshape(-1, 3)
        visible = keypoints[:, 2] > 0
        for x, y in keypoints[visible, :2]:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)


def draw_instance_segmentation(draw, ann, coco, color):
    mask = coco.annToMask(ann)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image = mask_image.convert("RGBA")
    mask_data = mask_image.getdata()
    r, g, b = color
    new_data = [(r, g, b, 128) for _, _, _, a in mask_data]
    mask_image.putdata(new_data)
    return mask_image


def draw_semantic_segmentation(draw, ann, coco, color):
    if "segmentation" in ann:
        mask = coco.annToMask(ann)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image = mask_image.convert("RGBA")
        mask_data = mask_image.getdata()
        r, g, b = color
        new_data = [(r, g, b, 128) for _, _, _, a in mask_data]
        mask_image.putdata(new_data)
        return mask_image


def generate_coco_preview(annotations_file, image_dir, output_dir, num_samples=0):
    """
    Generate preview images for MS COCO dataset with various annotation types.

    :param annotations_file: Path to COCO annotations JSON file
    :param image_dir: Directory containing COCO images
    :param output_dir: Directory to save preview images
    :param num_samples: Number of random samples to generate (default: 0)
        If 0, all images will be used
    """
    coco = COCO(annotations_file)
    img_ids = coco.getImgIds()
    if num_samples == 0:
        sample_img_ids = img_ids
    else:
        sample_img_ids = random.sample(img_ids, min(num_samples, len(img_ids)))

    os.makedirs(output_dir, exist_ok=True)

    for img_id in sample_img_ids:
        img = coco.loadImgs(img_id)[0]
        image_path = os.path.join(image_dir, img["file_name"])
        image = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(image)

        # Get high contrast color for this image
        high_contrast_color = get_high_contrast_color(image)

        ann_ids = coco.getAnnIds(imgIds=img["id"])
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if "bbox" in ann:
                draw_bbox(draw, ann, high_contrast_color)
            if "segmentation" in ann:
                if isinstance(ann["segmentation"], list):  # Polygonal segmentation
                    draw_polygon(draw, ann, high_contrast_color)
                else:  # RLE segmentation (instance or semantic)
                    mask_image = draw_instance_segmentation(
                        draw, ann, coco, high_contrast_color
                    )
                    image = Image.alpha_composite(image, mask_image)
            if "keypoints" in ann:
                draw_keypoints(draw, ann, high_contrast_color)

        output_path = os.path.join(output_dir, f"preview_{img['file_name']}")
        image = image.convert("RGB")
        image.save(output_path)

    print(f"Generated {len(sample_img_ids)} preview images in {output_dir}")
