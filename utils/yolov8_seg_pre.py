import cv2
import numpy as np
import math
from typing import Tuple, Optional

class YoloV8Preprocessor:
    """
    웹캠/동영상 등에서 들어오는 ndarray 프레임용 전처리

    Args:
        infer_shape: (H, W) - 모델 입력 크기
        src_is_bgr: 입력 프레임이 BGR(OpenCV 기본)인지 여부
    """
    def __init__(self, infer_shape: Tuple[int, int] = (640, 640), src_is_bgr: bool = True):
        self.infer_shape = infer_shape
        self.src_is_bgr = src_is_bgr

    def __call__(self, frame: np.ndarray):
        """
        Args:
            frame: HxWxC ndarray (uint8 or float). BGR가 기본.

        Returns:
            img: 전처리된 이미지 (float32, HxWxC, 0~1)
            meta: dict
              - r: scale ratio
              - pad: (pad_w, pad_h) = (dw, dh)
              - pad_sides: (left, right, top, bottom)
              - orig_shape: (h0, w0)
              - new_shape: (h, w)  # pad 전 resize된 크기
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a numpy ndarray.")

        # 1) 채널/타입 정리
        img = frame
        if img.ndim == 2:  # grayscale -> 3채널
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 입력이 BGR이라면 RGB로 변환 (모델이 RGB 기준이라면 권장)
        if self.src_is_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.dtype != np.uint8:
            # float 등 들어오면 0~255 범위로 가정하고 클리핑 후 uint8로
            img = np.clip(img, 0, 255).astype(np.uint8)

        h0, w0 = img.shape[:2]
        H, W = self.infer_shape

        # 2) 스케일 계산 (letterbox)
        r = min(H / h0, W / w0)
        new_w = int(round(w0 * r))
        new_h = int(round(h0 * r))

        # 3) resize (pad 전)
        if (new_w, new_h) != (w0, h0):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 4) padding 계산 (좌우/상하 반반)
        dw = W - new_w
        dh = H - new_h
        left  = dw // 2
        right = dw - left
        top   = dh // 2
        bottom= dh - top

        # 5) pad + 정규화
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        img = (img.astype(np.float32) / 255.0)

        meta = {
            "r": r,
            "pad": (dw, dh),
            "pad_sides": (left, right, top, bottom),
            "orig_shape": (h0, w0),
            "new_shape": (new_h, new_w),
            "infer_shape": (H, W),
        }
        return img, meta
