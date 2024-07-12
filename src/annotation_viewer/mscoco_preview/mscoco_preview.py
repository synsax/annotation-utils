import os
import random
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def draw_bbox(ax, ann):
    bbox = ann['bbox']
    ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                fill=False, edgecolor='red', linewidth=2))

def draw_polygon(ax, ann):
    for seg in ann['segmentation']:
        poly = np.array(seg).reshape((int(len(seg)/2), 2))
        ax.add_patch(plt.Polygon(poly, fill=False, edgecolor='green', linewidth=2))

def draw_keypoints(ax, ann):
    if 'keypoints' in ann:
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)
        visible = keypoints[:, 2] > 0
        ax.scatter(keypoints[visible, 0], keypoints[visible, 1], c='blue', s=20)

def draw_instance_segmentation(ax, ann, coco):
    mask = coco.annToMask(ann)
    ax.imshow(mask, alpha=0.5, cmap='jet')

def draw_semantic_segmentation(ax, ann, coco):
    # Assuming semantic segmentation is stored as RLE in 'segmentation'
    if 'segmentation' in ann:
        mask = coco.annToMask(ann)
        ax.imshow(mask, alpha=0.5, cmap='rainbow')

def generate_coco_preview(annotations_file, image_dir, output_dir, num_samples=5):
    """
    Generate preview images for MS COCO dataset with various annotation types.
    
    :param annotations_file: Path to COCO annotations JSON file
    :param image_dir: Directory containing COCO images
    :param output_dir: Directory to save preview images
    :param num_samples: Number of random samples to generate (default: 5)
    """
    coco = COCO(annotations_file)
    img_ids = coco.getImgIds()
    sample_img_ids = random.sample(img_ids, min(num_samples, len(img_ids)))

    os.makedirs(output_dir, exist_ok=True)

    for img_id in sample_img_ids:
        img = coco.loadImgs(img_id)[0]
        image_path = os.path.join(image_dir, img['file_name'])
        I = np.array(Image.open(image_path))

        ann_ids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(ann_ids)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(I)
        ax.axis('off')

        for ann in anns:
            if 'bbox' in ann:
                draw_bbox(ax, ann)
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):  # Polygonal segmentation
                    draw_polygon(ax, ann)
                else:  # RLE segmentation (instance or semantic)
                    draw_instance_segmentation(ax, ann, coco)
            if 'keypoints' in ann:
                draw_keypoints(ax, ann)

        output_path = os.path.join(output_dir, f"preview_{img['file_name']}")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print(f"Generated {num_samples} preview images in {output_dir}")
