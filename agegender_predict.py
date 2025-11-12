"""
age_gender_predictor_npu.py
모빌린트 NPU 기반 나이/성별 예측 모듈 (GPU 버전과 동일한 전처리/후처리)
"""

import cv2
import numpy as np
import maccel
from pathlib import Path


class AgeGenderPreProcess:
    """나이/성별 모델 전처리 클래스 (GPU 버전과 동일)"""
    
    def __init__(self, img_size=96):
        self.img_size = img_size
        # GPU와 동일한 설정
        self.input_std = 1.0
        self.input_mean = 0.0
    
    def __call__(self, image):
        """
        이미지 전처리 - GPU 버전과 동일
        Args:
            image: OpenCV BGR 이미지 (numpy array)
        Returns:
            전처리된 이미지 (numpy array)
        """
        # 1. 이미지 크기 조정
        resized = cv2.resize(image, (self.img_size, self.img_size), 
                            interpolation=cv2.INTER_LINEAR)
        
        # 2. GPU와 동일한 방식으로 blob 생성
        # cv2.dnn.blobFromImage(image, scalefactor=1.0/std, size, mean, swapRB=True)
        # → (pixel - mean) * scalefactor
        # → (pixel - 0.0) * 1.0 = pixel (0~255 범위 유지)
        
        input_size = (self.img_size, self.img_size)
        blob = cv2.dnn.blobFromImage(
            resized,
            1.0 / self.input_std,  # 1.0 / 1.0 = 1.0
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),  # (0.0, 0.0, 0.0)
            swapRB=True
        )
        
        print(f"\n[전처리 디버그 - GPU 방식]")
        print(f"  입력 이미지 shape: {image.shape}")
        print(f"  Resized shape: {resized.shape}")
        print(f"  Blob shape: {blob.shape}")
        print(f"  Blob 값 범위: min={blob.min():.4f}, max={blob.max():.4f}, mean={blob.mean():.4f}")
        print(f"  Blob std: {blob.std():.4f}")
        print(f"  Sample blob values [0,0,:5,:5]:\n{blob[0, 0, :5, :5]}")
        
        return blob.astype(np.float32)


class AgeGenderPostProcess:
    """나이/성별 모델 후처리 클래스 (GPU 버전과 동일한 로직)"""
    
    def __call__(self, outputs):
        """
        모델 출력을 나이/성별 정보로 변환 - GPU 버전과 동일한 로직
        
        NPU 출력 형식:
          - outputs[0]: age (shape: 1,1,1,1) → GPU의 predictions[2]에 해당
          - outputs[1]: gender logits (shape: 1,2,1,1) → GPU의 predictions[:2]에 해당
        
        GPU 출력 형식:
          - predictions: shape (3,) = [gender_logit1, gender_logit2, age_normalized]
        
        Args:
            outputs: NPU 모델 출력 리스트
            
        Returns:
            tuple: (gender, age) - GPU 버전과 동일한 형식
        """
        print(f"\n{'='*70}")
        print(f"[나이/성별 NPU 원본 출력 디버깅]")
        print(f"{'='*70}")
        
        # 원본 출력 디버깅
        print(f"출력 개수: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"\n출력 {i}:")
            print(f"  - shape: {out.shape}")
            print(f"  - dtype: {out.dtype}")
            print(f"  - 원본 값:\n{out}")
            print(f"  - min: {out.min():.6f}, max: {out.max():.6f}, mean: {out.mean():.6f}")
            
            flat = out.flatten()
            if len(flat) <= 20:
                print(f"  - Flatten: {flat}")
        
        print(f"{'='*70}")
        
        try:
            # NPU 출력을 GPU 형식으로 재구성
            # outputs[0] = age, outputs[1] = [gender_logit1, gender_logit2]
            age_output = outputs[0].squeeze()  # (1,1,1,1) → scalar
            gender_output = outputs[1].squeeze()  # (1,2,1,1) → (2,)
            
            print(f"\n[Squeeze 후]")
            print(f"age_output shape: {age_output.shape}, 값: {age_output}")
            print(f"gender_output shape: {gender_output.shape}, 값: {gender_output}")
            
            # GPU 형식으로 결합: [gender_logit1, gender_logit2, age_normalized]
            # 주의: NPU 출력 순서를 확인해야 함
            if gender_output.ndim == 0:
                gender_logits = np.array([gender_output])
            else:
                gender_logits = gender_output
            
            if age_output.ndim == 0:
                age_value = float(age_output)
            else:
                age_value = float(age_output.item()) if age_output.size == 1 else float(age_output[0])
            
            # GPU와 동일한 predictions 형식으로 재구성
            predictions = np.concatenate([gender_logits, [age_value]])
            
            print(f"\n[GPU 형식으로 재구성된 predictions]")
            print(f"  predictions shape: {predictions.shape}")
            print(f"  predictions: {predictions}")
            print(f"  predictions[:2] (gender logits): {predictions[:2]}")
            print(f"  predictions[2] (age normalized): {predictions[2]:.6f}")
            
            # GPU와 동일한 후처리 로직
            print(f"\n[GPU 방식 후처리]")
            
            # 성별 예측
            gender = np.argmax(predictions[:2])
            print(f"  Gender argmax: {gender} ({'Male' if gender == 1 else 'Female'})")
            print(f"  Gender logits: {predictions[:2]}")
            
            # 나이 예측
            age = int(np.round(predictions[2] * 100))
            print(f"  Age normalized: {predictions[2]:.6f}")
            print(f"  Age * 100: {predictions[2]*100:.4f}")
            print(f"  Age rounded: {age}")
            
            print(f"\n{'='*70}")
            print(f"[최종 결과 - GPU 방식]")
            print(f"  Gender: {gender} ({'Male' if gender == 1 else 'Female'})")
            print(f"  Age: {age}세")
            print(f"{'='*70}\n")
            
            return gender, age
            
        except Exception as e:
            print(f"\n[에러] 후처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return None, None


class AgeGenderPredictor:
    """NPU 기반 나이/성별 예측 클래스"""
    
    def __init__(self, mxq_path, img_size=96):
        """
        Args:
            mxq_path: MXQ 모델 파일 경로
            img_size: 입력 이미지 크기 (기본값: 96)
        """
        self.mxq_path = mxq_path
        self.img_size = img_size
        
        # NPU 초기화
        self._init_npu()
        
        # 전처리/후처리 초기화
        self.preprocessor = AgeGenderPreProcess(img_size=img_size)
        self.postprocessor = AgeGenderPostProcess()
        
    def _init_npu(self):
        """NPU 모델 초기화"""
        if not Path(self.mxq_path).exists():
            raise FileNotFoundError(f"MXQ 파일을 찾을 수 없습니다: {self.mxq_path}")
        
        # Accelerator 및 ModelConfig 설정
        self.acc = maccel.Accelerator()
        mc = maccel.ModelConfig()
        mc.exclude_all_cores()
        mc.include(maccel.Cluster.Cluster1, maccel.Core.Core1)
        
        # 모델 로드 및 실행
        self.model = maccel.Model(self.mxq_path, mc)
        self.model.launch(self.acc)
        
        print(f"NPU 나이/성별 모델 로드 완료: {self.mxq_path}")
    
    def predict(self, face_image):
        """
        얼굴 이미지에서 나이와 성별 예측 - GPU 버전과 동일한 반환 형식
        
        Args:
            face_image: 크롭된 얼굴 이미지 (OpenCV BGR 이미지)
            
        Returns:
            tuple: (gender, age)
                - gender: 0 (Female) or 1 (Male)
                - age: 예측된 나이 (정수)
                또는 (None, None) (실패 시)
        """
        try:
            # 입력 이미지가 비어있는지 확인
            if face_image is None or face_image.size == 0:
                print("경고: 빈 얼굴 이미지")
                return None, None
            
            # 전처리 (GPU 방식)
            input_tensor = self.preprocessor(face_image)
            
            # NPU 추론
            npu_out = self.model.infer([input_tensor])
            npu_out = [np.array(out) for out in npu_out]
            
            # 후처리 (GPU 방식)
            gender, age = self.postprocessor(npu_out)
            
            return gender, age
            
        except Exception as e:
            print(f"예측 오류: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def __del__(self):
        """소멸자: NPU 리소스 정리"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'acc'):
            del self.acc