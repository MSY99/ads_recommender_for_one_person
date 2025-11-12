import torch
import cv2
from utils.common import *
from typing import Tuple, List
import logging
import inspect

CLASSES = (
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock",
    "vase","scissors","teddy bear","hair drier","toothbrush"
)

def build_palette(n: int) -> list[tuple]:
    # HSV에서 H만 균등 분할 → BGR 변환(OpenCV는 H:0~179)
    hues = np.linspace(0, 179, n, endpoint=False, dtype=np.uint8)
    sat  = np.full((n,), 200, dtype=np.uint8)
    val  = np.full((n,), 255, dtype=np.uint8)
    hsv  = np.stack([hues, sat, val], axis=1).reshape(1, n, 3)
    bgr  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(n, 3)
    return [tuple(map(int, c)) for c in bgr]

DET_PALETTE = build_palette(len(CLASSES))

log_level_dict = {
    0: logging.DEBUG,
    1: logging.INFO,
    2: logging.WARNING,
    3: logging.ERROR,
    4: logging.CRITICAL,
}


def set_logger(log_level: int):
    logging.basicConfig(
        level=log_level_dict[log_level],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    

def get_coco_class_num() -> int:
    return len(CLASSES)


def get_coco_label(idx: int) -> str:
    type_check(idx, int)
    return CLASSES[idx]


def get_coco_det_palette(idx: int) -> tuple:
    type_check(idx, int)
    return DET_PALETTE[idx]


def Tutorial_save(
    out_post_processed: List[torch.Tensor],
    img_size: Tuple[int],
    input_path: str,
    output_path: str = None,
    show: bool = False,
):
    img = cv2.imread(input_path)
    det = out_post_processed[0]
    num_det = det.shape[0]

    det[:, :4] = scale_boxes(img_size, det[:, :4], img.shape).round()
    for j in range(num_det):
        xyxy = det[j, :4].view(-1).tolist()
        conf = det[j, 4].cpu().numpy()
        cls = det[j, 5].cpu().numpy().item()
        cls_name = get_coco_label(int(cls))
        cls_color = get_coco_det_palette(int(cls))
        desc = f"{cls_name}: {round(conf.item(),3)}"
        img = draw_boxes(img, xyxy, desc, cls_color)
    cv2.imwrite(output_path, img)

    delay = 0
    if show:
        cv2.imshow('press "q" to exit...', img)
        key = cv2.waitKey(delay)
        if key == ord("q"):
            cv2.destroyAllWindows('press "q" to exit...')

def type_check(obj, class_input: List):
    error_msg = "class_input must be a class or list of class, but got "

    if isinstance(class_input, list):
        for class_ in class_input:
            if not inspect.isclass(class_):
                raise TypeError(
                    f"class_input must be a class or list of class, but got {type(class_)} in the list."
                )

        if not any([isinstance(obj, class_) for class_ in class_input]):
            class_list = set([t.__name__ for t in class_input])
            raise TypeError(
                f"Expected one of {class_list} input, but got: {type(obj)}."
            )
    else:
        if not inspect.isclass(class_input):
            raise TypeError(
                f"class_input must be a class or list of class, but got {type(class_)}."
            )

        if not isinstance(obj, class_input):
            raise TypeError(f"Got unexpected type: {type(obj)}.")
