"""
face_detector_npu.py
모빌린트 NPU 기반 YOLOv8 얼굴 탐지 모듈
"""

import cv2
import numpy as np
import maccel
from pathlib import Path


class YOLOv8PreProcess:
    """YOLOv8 전처리 클래스"""
    
    def __init__(self, img_size=640):
        self.img_size = img_size
    
    def __call__(self, image):
        # 원본 크기 저장
        self.original_shape = image.shape[:2]
        
        # 리사이즈 (letterbox)
        img_resized = self.letterbox(image, new_shape=(self.img_size, self.img_size))
        
        # BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 정규화 (0-255 -> 0-1)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # ===== 수정: HWC to CHW와 배치 차원 추가 제거 =====
        # img_chw = np.transpose(img_normalized, (2, 0, 1))
        # img_batch = np.expand_dims(img_chw, axis=0)
        # return img_batch.astype(np.float32)
        
        # Tutorial_pre와 동일하게 (H, W, C) 형태로 반환
        return img_normalized  # (640, 640, 3)
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """
        이미지를 letterbox 방식으로 리사이즈
        """
        shape = img.shape[:2]  # 현재 shape [height, width]
        
        # 스케일 비율 계산
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # 새로운 크기 계산
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        # 중앙 정렬을 위한 패딩
        dw /= 2
        dh /= 2
        
        # 리사이즈
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # 패딩 추가
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                 cv2.BORDER_CONSTANT, value=color)
        
        return img


class FaceDetector:
    """NPU 기반 YOLOv8 얼굴 탐지 클래스"""
    
    def __init__(self, mxq_path, conf_threshold=0.5, iou_threshold=0.45, img_size=640):
        """
        Args:
            mxq_path: MXQ 모델 파일 경로
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            img_size: 입력 이미지 크기
        """
        self.mxq_path = mxq_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        # NPU 초기화
        self._init_npu()
        
        # 전처리/후처리 초기화
        self.preprocessor = YOLOv8PreProcess(img_size=img_size)
        self.postprocessor = None  # 첫 추론 시 초기화
        
    def _init_npu(self):
        """NPU 모델 초기화"""
        if not Path(self.mxq_path).exists():
            raise FileNotFoundError(f"MXQ 파일을 찾을 수 없습니다: {self.mxq_path}")
        
        # Accelerator 및 ModelConfig 설정
        self.acc = maccel.Accelerator()
        mc = maccel.ModelConfig()
        mc.exclude_all_cores()
        mc.include(maccel.Cluster.Cluster1, maccel.Core.Core0)
        
        # 모델 로드 및 실행
        self.model = maccel.Model(self.mxq_path, mc)
        self.model.launch(self.acc)
        
        print(f"NPU 모델 로드 완료: {self.mxq_path}")
    
    def detect_faces(self, frame):
        """
        프레임에서 얼굴 탐지
        
        Args:
            frame: OpenCV BGR 이미지 (numpy array)
            
        Returns:
            List[dict]: 탐지된 얼굴 정보 리스트
                - bbox: [x1, y1, x2, y2]
                - confidence: 신뢰도
                - cropped_face: 잘린 얼굴 이미지
        """
        # 전처리
        input_tensor = self.preprocessor(frame)
        
        # NPU 추론
        npu_out = self.model.infer([input_tensor])
        npu_out = [np.array(out) for out in npu_out]
        
        # ===== 디버깅: NPU 출력 형태 확인 =====
        #print(f"\n=== NPU 출력 디버깅 ===")
        #print(f"NPU 출력 개수: {len(npu_out)}")
        #for i, out in enumerate(npu_out):
        #    print(f"출력 {i} shape: {out.shape}, dtype: {out.dtype}")
        #print("=" * 50)
        # =====================================
        
        # 후처리 (첫 추론 시 초기화)
        if self.postprocessor is None:
            from utils.yolov8_detect_post import YOLOv8PostProcess
            self.postprocessor = YOLOv8PostProcess(
                imh=self.img_size,
                imw=self.img_size,
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
                nc=1  # 얼굴 클래스 1개
            )
        
        # 후처리 실행
        detections = self.postprocessor(npu_out)
        
        # 결과가 없으면 빈 리스트 반환
        if detections is None or detections[0].nelement() == 0:
            return []
        
        # 결과 변환
        faces = self._convert_detections(detections[0], frame)
        
        return faces
    
    def _convert_detections(self, detections, original_frame):
        """
        탐지 결과를 표준 형식으로 변환
        
        Args:
            detections: torch.Tensor [N, 6] (x1, y1, x2, y2, conf, cls)
            original_frame: 원본 프레임
            
        Returns:
            List[dict]: 변환된 탐지 결과
        """
        faces = []
        
        # 원본 이미지 크기
        orig_h, orig_w = self.preprocessor.original_shape
        
        # 스케일 계산
        scale_x = orig_w / self.img_size
        scale_y = orig_h / self.img_size
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            
            # 좌표를 원본 이미지 크기로 변환
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # 좌표 클리핑
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            
            # 유효한 박스만 추가
            if x2 > x1 and y2 > y1:
                # 얼굴 영역 크롭
                cropped_face = original_frame[y1:y2, x1:x2]
                
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'cropped_face': cropped_face
                })
        
        return faces
    
    def set_confidence_threshold(self, threshold):
        """신뢰도 임계값 변경"""
        self.conf_threshold = threshold
        if self.postprocessor is not None:
            self.postprocessor.conf_thres = threshold
    
    def __del__(self):
        """소멸자: NPU 리소스 정리"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'acc'):
            del self.acc