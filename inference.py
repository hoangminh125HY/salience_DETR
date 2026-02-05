import argparse
import os
from typing import Dict

import accelerate
import cv2
import numpy as np
import torch
import torch.utils.data as data
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from util.lazy_load import Config
from util.logger import setup_logger
from util.utils import load_checkpoint, load_state_dict
from util.visualize import plot_bounding_boxes_on_image_cv2


# ---------------- DATASET ----------------
def is_image(file_path):
    try:
        img = Image.open(file_path)
        img.close()
        return True
    except:
        return False


class InferenceDataset(data.Dataset):
    def __init__(self, root):
        self.images = [os.path.join(root, img) for img in os.listdir(root)]
        self.images = [img for img in self.images if is_image(img)]
        assert len(self.images) > 0, "No images found in folder"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        path = self.images[index]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)

        return torch.tensor(img), path


# ---------------- ARGUMENTS ----------------
def parse_args():
    parser = argparse.ArgumentParser("Salience-DETR inference")

    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--show-dir", type=str, default="output")
    parser.add_argument("--show-conf", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=2)

    return parser.parse_args()


# ---------------- MAIN ----------------
def main():
    args = parse_args()

    accelerator = Accelerator()
    accelerate.utils.set_seed(42)

    for logger_name in ["py.warnings", "accelerate", os.path.basename(os.getcwd())]:
        setup_logger(distributed_rank=accelerator.local_process_index, name=logger_name)

    # dataset
    dataset = InferenceDataset(args.image_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    # model
    model = Config(args.model_config).model.eval()
    checkpoint = load_checkpoint(args.checkpoint)

    if isinstance(checkpoint, Dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    load_state_dict(model, checkpoint)
    model = accelerator.prepare_model(model)

    os.makedirs(args.show_dir, exist_ok=True)

    # ---------------- INFERENCE ----------------
    with torch.inference_mode():
        for images, path in tqdm(loader):

            images = images.to(accelerator.device)
            outputs = model(images)[0]

            # move cpu
            for k in outputs:
                outputs[k] = outputs[k].cpu()

            # FILTER CONFIDENCE (QUAN TRỌNG NHẤT)
            if "scores" in outputs:
                keep = outputs["scores"] >= args.show_conf
                boxes = outputs["boxes"][keep]
                labels = outputs["labels"][keep]
                scores = outputs["scores"][keep]
            else:
                boxes = outputs["boxes"]
                labels = outputs["labels"]
                scores = None

            # convert image back
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img = plot_bounding_boxes_on_image_cv2(
                image=img,
                boxes=boxes,
                labels=labels,
                scores=scores,
                classes=model.CLASSES,
                show_conf=args.show_conf,
            )

            save_path = os.path.join(args.show_dir, os.path.basename(path[0]))
            cv2.imwrite(save_path, img)


if __name__ == "__main__":
    main()
