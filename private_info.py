#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
나이/성별 탐지 모듈
얼굴 탐지 및 나이/성별 예측 기능을 제공
"""

import os
import sys

# OpenCV와 PyQt5의 Qt 충돌 방지
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

# Python 모듈 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# models 폴더를 Python 경로에 추가 (절대 임포트를 위해)
models_dir = os.path.join(current_dir, 'models')
if os.path.exists(models_dir) and models_dir not in sys.path:
    sys.path.insert(0, models_dir)
    print(f"[INFO] Added to sys.path: {models_dir}")

import cv2

class AgeGenderDetectionManager:
    """
    얼굴 탐지 및 나이/성별 예측을 관리하는 클래스
    """
    
    def __init__(self, face_model_path, age_gender_model_path):
        """
        Args:
            face_model_path (str): 얼굴 탐지 모델 경로
            age_gender_model_path (str): 나이/성별 예측 모델 경로
        """
        self.face_detector = None
        self.age_gender_predictor = None
        self.is_initialized = False
        
        try:
            print("[INFO] Python sys.path:")
            for i, path in enumerate(sys.path[:5], 1):
                print(f"  {i}. {path}")
            
            # ===== 중요: 절대 임포트 사용 =====
            print("[INFO] FaceDetector 임포트 시도...")
            from face_detector import FaceDetector  # models/ 폴더 기준
            
            print("[INFO] AgeGenderPredictor 임포트 시도...")
            from agegender_predict import AgeGenderPredictor  # models/ 폴더 기준
            # ===================================
            
            print("[INFO] FaceDetector 초기화 중...")
            self.face_detector = FaceDetector(
                mxq_path=face_model_path,
                conf_threshold=0.5,
                iou_threshold=0.45,
                img_size=640
            )
            print("[INFO] FaceDetector 초기화 완료")
            
            print("[INFO] AgeGenderPredictor 초기화 중...")
            self.age_gender_predictor = AgeGenderPredictor(
                mxq_path=age_gender_model_path,
                img_size=96
            )
            print("[INFO] AgeGenderPredictor 초기화 완료")
            
            self.is_initialized = True
            
        except ImportError as e:
            print(f"[ERROR] 모듈 임포트 실패: {e}")
            print(f"[ERROR] 현재 작업 디렉토리: {os.getcwd()}")
            self.is_initialized = False
        except Exception as e:
            print(f"[ERROR] 모델 초기화 실패: {e}")
            import traceback
            traceback.print_exc()
            self.is_initialized = False
    
    def process_frame(self, frame):
        """
        프레임에서 얼굴을 탐지하고 나이/성별을 예측
        
        Args:
            frame: OpenCV 이미지 프레임
        
        Returns:
            dict: {
                'frame': 결과가 그려진 프레임,
                'detections': 탐지 결과 리스트,
                'summary': 요약 텍스트 (str)
            }
        """
        if not self.is_initialized:
            return {
                'frame': frame,
                'detections': [],
                'summary': "모델이 초기화되지 않았습니다."
            }
        
        try:
            # 얼굴 탐지
            faces = self.face_detector.detect_faces(frame)
            
            detections = []
            result_frame = frame.copy()
            
            # 각 얼굴에 대해 나이/성별 예측
            for i, face_info in enumerate(faces):
                bbox = face_info['bbox']
                confidence = face_info['confidence']
                cropped_face = face_info['cropped_face']
                
                x1, y1, x2, y2 = bbox
                
                # 나이/성별 예측
                age_gender_result = self.age_gender_predictor.predict(cropped_face)
                
                if age_gender_result is not None:
                    gender_idx, age = age_gender_result
                    gender = 'Male' if gender_idx == 1 else 'Female'
                    
                    # 결과 저장
                    detections.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'gender': gender,
                        'age': age
                    })
                    
                    # 프레임에 그리기
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{gender}, {age}"
                    cv2.putText(result_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # 예측 실패
                    detections.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'gender': 'Unknown',
                        'age': 0
                    })
                    
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(result_frame, "Face", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 요약 생성
            summary = self._generate_summary(detections)
            
            return {
                'frame': result_frame,
                'detections': detections,
                'summary': summary
            }
            
        except Exception as e:
            print(f"[ERROR] 프레임 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                'frame': frame,
                'detections': [],
                'summary': f"오류: {str(e)}"
            }
    
    def _generate_summary(self, detections):
        """
        탐지 결과 요약 텍스트 생성
        
        Args:
            detections: 탐지 결과 리스트
        
        Returns:
            str: 요약 텍스트
        """
        if len(detections) == 0:
            return "탐지된 얼굴 없음"
        
        summary_lines = [f"탐지된 얼굴: {len(detections)}명"]
        
        for i, det in enumerate(detections):
            if det['gender'] != 'Unknown':
                gender_kr = "남성" if det['gender'] == 'Male' else "여성"
                line = f"얼굴 {i+1}: {gender_kr}, {det['age']}세 (신뢰도: {det['confidence']:.2f})"
            else:
                line = f"얼굴 {i+1}: 예측 실패 (신뢰도: {det['confidence']:.2f})"
            
            summary_lines.append(line)
        
        return "\n".join(summary_lines)