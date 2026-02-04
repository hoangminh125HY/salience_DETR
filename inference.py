import os
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.ops import nms

from util.lazy_load import Config
from util.utils import load_checkpoint, load_state_dict
from util.visualize import plot_bounding_boxes_on_image_cv2


# =========================
# Utils
# =========================
def is_image(path):
    try:
        Image.open(path).close()
        return True
    except:
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--show-dir", type=str, required=True)
    parser.add_argument("--conf-thres", type=float, default=0.3)
    parser.add_argument("--nms-thres", type=float, default=0.5)
    return parser.parse_args()


# =========================
# Main
# =========================
@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.show_dir, exist_ok=True)

    # 👉 TỰ KHAI BÁO CLASS (quan trọng)
    classes = ["UAV"]   # sửa nếu dataset bạn nhiều class

    # Load model
    model = Config(args.model_config).model.eval().cuda()
    ckpt = load_checkpoint(args.checkpoint)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    load_state_dict(model, ckpt)

    # Load images
    image_paths = [
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if is_image(os.path.join(args.image_dir, f))
    ]

    for img_path in tqdm(image_paths):
        img = cv2.imdecode(
            np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        h, w = img.shape[:2]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().cuda()
        img_tensor = img_tensor.unsqueeze(0)

        output = model(img_tensor)[0]

        boxes = output["boxes"]
        scores = output["scores"]
        labels = output["labels"]

        # =========================
        # Confidence filter
        # =========================
        keep = scores >= args.conf_thres
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if boxes.numel() == 0:
            continue

        # =========================
        # NMS
        # =========================
        keep = nms(boxes, scores, args.nms_thres)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # ⚠ FIX crash: ép label về 0 nếu dataset 1 class
        labels = torch.zeros_like(labels)

        # =========================
        # Draw
        # =========================
        vis = plot_bounding_boxes_on_image_cv2(
            image=img,
            boxes=boxes.cpu(),
            labels=labels.cpu(),
            scores=scores.cpu(),
            classes=classes,
            show_conf=args.conf_thres,
            box_thick=2,
        )

        save_path = os.path.join(args.show_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, vis)


if __name__ == "__main__":
    main()
