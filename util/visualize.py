import copy
import os
from functools import partial
from typing import List, Tuple, Union

import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.coco import CocoDetection


def label_colormap(n_label=256, value=None):
    """Label colormap.

    :param n_label: Number of labels, defaults to 256
    :param value: Value scale or value of label color in HSV space, defaults to None
    :return: Label id to colormap, numpy.ndarray, (N, 3), numpy.uint8
    """
    def bitget(byteval, idx):
        shape = byteval.shape + (8,)
        return np.unpackbits(byteval).reshape(shape)[..., -1 - idx]

    i = np.arange(n_label, dtype=np.uint8)
    r = np.full_like(i, 0)
    g = np.full_like(i, 0)
    b = np.full_like(i, 0)

    i = np.repeat(i[:, None], 8, axis=1)
    i = np.right_shift(i, np.arange(0, 24, 3)).astype(np.uint8)
    j = np.arange(8)[::-1]
    r = np.bitwise_or.reduce(np.left_shift(bitget(i, 0), j), axis=1)
    g = np.bitwise_or.reduce(np.left_shift(bitget(i, 1), j), axis=1)
    b = np.bitwise_or.reduce(np.left_shift(bitget(i, 2), j), axis=1)

    cmap = np.stack((r, g, b), axis=1).astype(np.uint8)

    if value is not None:
        hsv = cv2.cvtColor(cmap.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        cmap = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)
    return cmap


def generate_color_palette(n: int, contrast: bool = False):
    colors = label_colormap(n)
    hsv_colors = cv2.cvtColor(colors[None], cv2.COLOR_RGB2HSV)[0]

    if not contrast:
        return colors

    # generate contrast lighter and darker colors
    dark_colors = hsv_colors.copy()
    dark_colors[:, -1] //= 2
    light_colors = dark_colors.copy()
    light_colors[:, -1] += 128

    dark_colors = cv2.cvtColor(dark_colors[None], cv2.COLOR_HSV2RGB)[0]
    light_colors = cv2.cvtColor(light_colors[None], cv2.COLOR_HSV2RGB)[0]
    return colors, light_colors, dark_colors


def plot_bounding_boxes_on_image_cv2(
    image,
    boxes,
    labels,
    scores=None,
    classes=None,
    show_conf=0.3,
    font_scale=0.5,
    box_thick=2,
    **kwargs
):

    if len(labels) == 0:
        return image

    import numpy as np
    import cv2

    # convert numpy
    if any(not isinstance(t, np.ndarray) for t in (boxes, labels)):
        boxes, labels = map(np.array, (boxes, labels))
    if scores is not None and not isinstance(scores, np.ndarray):
        scores = np.array(scores)

    boxes = boxes.astype(np.int32)

    # filter confidence
    if scores is not None:
        keep = scores >= show_conf
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    if len(labels) == 0:
        return image

    if classes is None:
        classes = [str(i) for i in range(max(labels)+1)]

    # OpenCV dùng BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ===== COLOR STYLE (BGR) =====
    CLASS_STYLE = {
        "uav":  {"box": (51,82,255),  "txt": (255,255,255)},  # xanh đậm + chữ trắng
        "kite": {"box": (120,200,255),  "txt": (0,0,0)},        # xanh nhạt + chữ đen
    }

    H, W = image.shape[:2]

    for i, box in enumerate(boxes):

        cls_id = int(labels[i])          # QUAN TRỌNG
        cls_text = classes[cls_id]       # hiển thị UAV
        cls_key  = cls_text.strip().lower()      # map màu

        style = CLASS_STYLE.get(cls_key, {"box":(0,255,0),"txt":(0,0,0)})
        box_color = style["box"]
        txt_color = style["txt"]

        x1, y1, x2, y2 = box

        # clamp
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W-1, x2); y2 = min(H-1, y2)

        # 1️⃣ Bounding box
        cv2.rectangle(image, (x1,y1), (x2,y2), box_color, box_thick)

        # text
        if scores is not None:
            text = f"{cls_text} {scores[i]:.2f}"
        else:
            text = cls_text

        # text size
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # label position (tránh tràn ảnh)
        y_text = y1 - th - 6
        if y_text < 0:
            y_text = y1 + 3

        # 2️⃣ nền label
        cv2.rectangle(
            image,
            (x1, y_text),
            (x1 + tw + 4, y_text + th + 4),
            box_color,
            -1
        )

        # 3️⃣ chữ
        cv2.putText(
            image,
            text,
            (x1 + 2, y_text + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            txt_color,
            1,
            cv2.LINE_AA
        )

    # trả về RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image




def visualize_coco_bounding_boxes(
    data_loader: DataLoader,
    show_conf: float = 0.0,
    show_dir: str = None,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
):
    """Given a DataLoader of CocoDetection, plot bounding boxes, labels and save into given dir.

    :param data_loader: DataLoader of CocoDetection.
    :param show_conf: Only results with confidence > show_conf will be plot, defaults to 0.0
    :param show_dir: directory to save visualization results, defaults to None
    :param font_scale: scale factor to set font size, defaults to 1.0
    :param box_thick: scale factor to set box border weight, defaults to 3
    :param fill_alpha: alpha to filling the area in the bounding box, defaults to 0.2
    :param text_box_color: background color of the text box, defaults to (255, 255, 255)
    :param text_font_color: text color, will be set automatically if not given, defaults to None
    :param text_alpha: alpha to filling the area in the text box, defaults to 0.5
    """
    assert data_loader.batch_size in (None, 1), "batch_size of DataLoader for visualization must be 1"
    assert isinstance(data_loader.dataset, CocoDetection), "Only CocoDetection dataset is supported"
    os.makedirs(show_dir, exist_ok=True)
    dataset: CocoDetection = data_loader.dataset
    cat_ids = list(range(max(dataset.coco.cats.keys()) + 1))
    classes = tuple(dataset.coco.cats.get(c, {"name": "none"})["name"] for c in cat_ids)

    # multi-process on Windows does not support pickle local functions
    # we use functools.partial on global functools to workaround it
    data_loader.collate_fn = partial(
        _visualize_batch_in_coco,
        classes=classes,
        show_conf=show_conf,
        font_scale=font_scale,
        box_thick=box_thick,
        fill_alpha=fill_alpha,
        text_box_color=text_box_color,
        text_font_color=text_font_color,
        text_alpha=text_alpha,
        dataset=dataset,
        show_dir=show_dir,
    )
    [None for _ in tqdm(data_loader)]


def _visualize_batch_in_coco(
    batch: Tuple[np.ndarray, dict],
    dataset: CocoDetection,
    classes: List[str],
    show_conf: float = 0.0,
    show_dir: str = None,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
):
    image, output = batch[0]
    # plot bounding boxes on image
    image = image.numpy().transpose(1, 2, 0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = plot_bounding_boxes_on_image_cv2(
        image=image,
        boxes=output["boxes"],
        labels=output["labels"],
        scores=output.get("scores", None),
        classes=classes,
        show_conf=show_conf,
        font_scale=font_scale,
        box_thick=box_thick,
        fill_alpha=fill_alpha,
        text_box_color=text_box_color,
        text_font_color=text_font_color,
        text_alpha=text_alpha,
    )
    image_name = dataset.coco.loadImgs([output["image_id"]])[0]["file_name"]
    cv2.imwrite(os.path.join(show_dir, os.path.basename(image_name)), image)
