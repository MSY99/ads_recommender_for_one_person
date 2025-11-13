#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
나이/성별 타겟 광고 시스템
- MainWindow: 메인 GUI 프레임 구성 및 전체 시스템 통합
- RealTimeDetecter: 웹캠 실시간 얼굴 탐지 및 나이/성별 추론
- AdsContent: 타겟 광고 표시 및 LLM 추론
"""

import sys
import os
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                             QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                             QPushButton, QComboBox, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QUrl
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import cv2
import numpy as np

# 기존 모듈 임포트
from webcam_connect import CameraManager
from private_info import AgeGenderDetectionManager
from llm_infer import LLMInferenceManager
from llm_worker import LLMInferenceWorkerThread


# ============================================================
# 상수 정의
# ============================================================
ADS_PATH = "./sample_ads/imgNvideos"
FACE_MODEL_PATH = "./models/cv/yolov8n-face-lindevs.mxq"
AGE_GENDER_MODEL_PATH = "./models/cv/genderage.mxq"
LLM_MODEL_PATH = "./models/llm/mblt-exaone"


# ============================================================
# RealTimeDetecter: 실시간 얼굴 탐지 및 나이/성별 추론
# ============================================================
class RealTimeDetecter(QObject):
    """실시간 얼굴 탐지 및 나이/성별 추론 클래스"""
    
    # 시그널 정의
    frame_updated = pyqtSignal(object)  # QImage 프레임
    status_updated = pyqtSignal(bool, str)  # 성공 여부, 메시지
    detection_result_updated = pyqtSignal(str)  # 탐지 결과 텍스트
    age_gender_extracted = pyqtSignal(int, str)  # 나이, 성별
    
    def __init__(self, face_model_path, age_gender_model_path):
        """
        Args:
            face_model_path: 얼굴 탐지 모델 경로
            age_gender_model_path: 나이/성별 추론 모델 경로
        """
        super().__init__()
        
        self.face_model_path = face_model_path
        self.age_gender_model_path = age_gender_model_path
        
        # 카메라 매니저
        self.camera_manager = CameraManager()
        
        # 탐지 매니저 (초기화 전)
        self.detection_manager = None
        
        # 현재 탐지된 나이/성별
        self.current_age = None
        self.current_gender = None
        
        # 초기화 상태
        self.is_initialized = False
    
    def initialize_models(self):
        """CV 모델 초기화 (얼굴 탐지 + 나이/성별)"""
        print("\n[RealTimeDetecter] CV 모델 로딩 시작...")
        
        try:
            self.detection_manager = AgeGenderDetectionManager(
                face_model_path=self.face_model_path,
                age_gender_model_path=self.age_gender_model_path
            )
            
            if not self.detection_manager.is_initialized:
                print("[RealTimeDetecter] ❌ CV 모델 초기화 실패")
                self.is_initialized = False
                return False
            
            print("[RealTimeDetecter] ✅ CV 모델 로딩 완료")
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"[RealTimeDetecter] ❌ CV 모델 로딩 예외: {e}")
            import traceback
            traceback.print_exc()
            self.is_initialized = False
            return False
    
    def start_camera(self, camera_id):
        """
        웹캠 시작 (나이/성별 탐지 자동 활성화)
        
        Args:
            camera_id: 카메라 디바이스 ID
            
        Returns:
            bool: 시작 성공 여부
        """
        if not self.is_initialized or self.detection_manager is None:
            print("[RealTimeDetecter] ❌ 모델이 초기화되지 않았습니다")
            return False
        
        print(f"\n[RealTimeDetecter] 카메라 {camera_id} 시작")
        
        # 카메라 시작
        camera_thread = self.camera_manager.start_camera(
            camera_id=camera_id,
            detection_manager=self.detection_manager
        )
        
        # 시그널 연결
        camera_thread.frame_update.connect(self._on_frame_update)
        camera_thread.connection_status.connect(self._on_connection_status)
        camera_thread.detection_result.connect(self._on_detection_result)
        
        return True
    
    def stop_camera(self):
        """웹캠 중지"""
        print("[RealTimeDetecter] 카메라 중지")
        self.camera_manager.stop_camera()
        
        # 탐지 정보 초기화
        self.current_age = None
        self.current_gender = None
    
    def is_camera_running(self):
        """카메라 실행 상태 확인"""
        return self.camera_manager.is_running()
    
    def get_current_detection(self):
        """
        현재 탐지된 나이/성별 정보 반환
        
        Returns:
            tuple: (나이, 성별) 또는 (None, None)
        """
        return self.current_age, self.current_gender
    
    def _on_frame_update(self, qt_image):
        """프레임 업데이트 시그널 전달"""
        self.frame_updated.emit(qt_image)
    
    def _on_connection_status(self, success, message):
        """연결 상태 시그널 전달"""
        self.status_updated.emit(success, message)
    
    def _on_detection_result(self, result_text):
        """
        탐지 결과 처리
        - 텍스트에서 나이/성별 추출
        - 시그널 발송
        """
        # 나이/성별 정보 추출
        age, gender = self._extract_age_gender_from_text(result_text)
        
        if age is not None and gender is not None:
            self.current_age = age
            self.current_gender = gender
            
            # 나이/성별 추출 시그널 발송
            self.age_gender_extracted.emit(age, gender)
        
        # 탐지 결과 텍스트 시그널 발송
        self.detection_result_updated.emit(result_text)
    
    def _extract_age_gender_from_text(self, result_text):
        """
        탐지 결과 텍스트에서 나이/성별 정보 추출 (정규표현식 사용)
        
        Args:
            result_text: 탐지 결과 텍스트
            
        Returns:
            tuple: (나이, 성별) 또는 (None, None)
        """
        try:
            # 패턴 1: "얼굴 N: 성별, 나이세 (신뢰도: 0.xx)" 형식
            pattern = r'얼굴\s+(\d+):\s*(여성|남성),\s*(\d+)세\s*\(신뢰도:\s*([\d.]+)\)'
            matches = re.findall(pattern, result_text)
            
            if matches:
                face_num, gender, age, confidence = matches[0]
                age_val = int(age)
                print(f"[RealTimeDetecter] ✓ 탐지 정보: 나이={age_val}세, 성별={gender}")
                print(f"  (얼굴 {face_num}, 신뢰도: {confidence})")
                return age_val, gender
            
            # 패턴 2: "나이: 25, 성별: 여성" 형식
            pattern_alt1 = r'나이:\s*([\d.]+),\s*성별:\s*(여성|남성)'
            match_alt1 = re.search(pattern_alt1, result_text)
            if match_alt1:
                age_str, gender = match_alt1.groups()
                age_val = int(float(age_str))
                print(f"[RealTimeDetecter] ✓ 탐지 정보 (대체 형식): 나이={age_val}세, 성별={gender}")
                return age_val, gender
            
            # 패턴 3: 성별과 나이를 따로 찾기
            age_pattern = r'(?:나이|Age|age):\s*([\d.]+)'
            gender_pattern = r'(?:성별|Gender|gender):\s*(\w+)'
            
            age_match = re.search(age_pattern, result_text)
            gender_match = re.search(gender_pattern, result_text)
            
            if age_match and gender_match:
                age_str = age_match.group(1)
                gender_str = gender_match.group(1).lower()
                
                age_val = int(float(age_str))
                
                # 성별 매핑
                if '여' in gender_str or 'female' in gender_str:
                    gender_val = "여성"
                elif '남' in gender_str or 'male' in gender_str:
                    gender_val = "남성"
                else:
                    return None, None
                
                print(f"[RealTimeDetecter] ✓ 탐지 정보 (분리 패턴): 나이={age_val}세, 성별={gender_val}")
                return age_val, gender_val
            
            # 모든 패턴 매칭 실패
            print(f"[RealTimeDetecter] ⚠ 탐지 정보 추출 실패")
            return None, None
                
        except Exception as e:
            print(f"[RealTimeDetecter] 나이/성별 정보 추출 오류: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def dispose(self):
        """리소스 정리"""
        print("[RealTimeDetecter] 리소스 정리")
        if self.is_camera_running():
            self.stop_camera()
        
        self.detection_manager = None
        self.is_initialized = False


# ============================================================
# AdsContent: 타겟 광고 표시 및 LLM 추론
# ============================================================
class AdsContent(QObject):
    """타겟 광고 표시 및 LLM 추론 클래스"""
    
    # 시그널 정의
    ad_image_ready = pyqtSignal(QPixmap)  # 광고 이미지 준비 완료
    ad_video_ready = pyqtSignal(str)  # 광고 비디오 준비 완료 (파일 경로)
    ad_video_frame = pyqtSignal(QPixmap)  # 비디오 프레임 업데이트
    ad_video_stopped = pyqtSignal()  # 비디오 재생 중지
    llm_text_ready = pyqtSignal(str)  # LLM 추론 텍스트 준비 완료
    error_occurred = pyqtSignal(str)  # 에러 발생
    
    def __init__(self, ad_base_path, llm_model_path):
        """
        Args:
            ad_base_path: 광고 이미지 디렉토리 경로
            llm_model_path: LLM 모델 경로
        """
        super().__init__()
        
        self.ad_base_path = ad_base_path
        self.llm_model_path = llm_model_path
        
        # LLM 매니저 (초기화 전)
        self.llm_manager = None
        
        # LLM 워커 스레드 참조
        self.llm_worker = None
        
        # OpenCV 비디오 재생 관련
        self.video_cap = None
        self.video_timer = None
        
        # 초기화 상태
        self.is_llm_initialized = False
    
    def initialize_llm(self):
        """LLM 모델 초기화"""
        print("\n[AdsContent] LLM 모델 로딩 시작...")
        
        try:
            self.llm_manager = LLMInferenceManager(
                model_path=self.llm_model_path
            )
            
            if not self.llm_manager.is_initialized:
                print("[AdsContent] ❌ LLM 초기화 실패")
                self.is_llm_initialized = False
                return False
            
            print("[AdsContent] ✅ LLM 로딩 완료")
            self.is_llm_initialized = True
            return True
            
        except Exception as e:
            print(f"[AdsContent] ❌ LLM 로딩 예외: {e}")
            self.is_llm_initialized = False
            return False
    
    def show_targeted_ad(self, age, gender):
        """
        나이/성별에 맞는 타겟 광고 표시
        
        Args:
            age: 탐지된 나이
            gender: 탐지된 성별 ("여성" 또는 "남성")
            
        Returns:
            bool: 광고 표시 성공 여부
        """
        if age is None or gender is None:
            error_msg = "탐지된 나이/성별 정보가 없습니다."
            print(f"[AdsContent] ❌ {error_msg}")
            self.error_occurred.emit(error_msg)
            return False
        
        # 연령대 결정
        age_group = self._get_age_group(age)
        
        # 성별을 영문으로 변환
        gender_en = "female" if gender == "여성" else "male"
        
        print(f"\n[AdsContent] 타겟: {age_group}대 {gender} (나이: {age}세)")
        print(f"[AdsContent] 광고 경로: {self.ad_base_path}")
        
        # 광고 경로 확인
        if not os.path.exists(self.ad_base_path):
            error_msg = f"광고 디렉토리를 찾을 수 없습니다.\n경로: {self.ad_base_path}"
            print(f"[AdsContent] ❌ {error_msg}")
            self.error_occurred.emit(error_msg)
            return False
        
        # ========================================
        # 1단계: 비디오 우선 확인 -> 없으면 이미지 표시
        # ========================================
        
        # 먼저 비디오 파일 확인
        video_found = self._load_and_show_ad_video(age_group, gender_en)
        
        if not video_found:
            # 비디오가 없으면 이미지 로드
            print("[AdsContent] 비디오 없음 - 이미지 로드 시도")
            image_found = self._load_and_show_ad_image(age_group, gender_en)
            
            if not image_found:
                return False
        
        # ========================================
        # 2단계: LLM 텍스트는 백그라운드에서 생성
        # ========================================
        if not self.is_llm_initialized or self.llm_manager is None:
            print("\n[AdsContent] LLM 모델이 없음 - 기본 설명 표시")
            self._show_default_explanation(age_group, gender, age)
            return True
        
        # LLM 비동기 추론 시작
        self._start_llm_inference_async(age_group, gender, age)
        
        return True
    
    def _load_and_show_ad_video(self, age_group, gender_en):
        """
        광고 비디오 로드 및 표시 (OpenCV 사용)
        
        Args:
            age_group: 연령대 ("20", "30", "40", "50")
            gender_en: 성별 영문 ("female", "male")
            
        Returns:
            bool: 비디오 로드 성공 여부
        """
        # mp4 확장자 확인
        video_filename = f"{age_group}_{gender_en}.mp4"
        video_path = os.path.join(self.ad_base_path, video_filename)
        
        print(f"[AdsContent] 비디오 시도: {video_path}")
        
        if os.path.exists(video_path):
            # 비디오 파일 발견
            print(f"[AdsContent] ✓ 비디오 발견: {video_filename}")
            
            # 기존 비디오 재생 중이면 중지
            self.stop_video()
            
            # OpenCV로 비디오 열기
            self.video_cap = cv2.VideoCapture(video_path)
            
            if not self.video_cap.isOpened():
                print(f"[AdsContent] ❌ 비디오를 열 수 없음: {video_path}")
                error_msg = f"비디오 파일을 열 수 없습니다.\n파일: {video_path}"
                self.error_occurred.emit(error_msg)
                return False
            
            # FPS 가져오기
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps > 120:  # 비정상적인 FPS
                fps = 30  # 기본값
            
            print(f"[AdsContent] 비디오 FPS: {fps}")
            
            # 비디오 준비 완료 시그널 발송 (MainWindow에서 UI 전환)
            abs_video_path = os.path.abspath(video_path)
            self.ad_video_ready.emit(abs_video_path)
            
            # 타이머 설정 (밀리초 단위)
            self.video_timer = QTimer()
            self.video_timer.timeout.connect(self._update_video_frame)
            self.video_timer.start(int(1000 / fps))
            
            print(f"[AdsContent] ✓ 비디오 재생 시작: {video_filename}")
            return True
        else:
            print(f"[AdsContent] 비디오 없음: {video_filename}")
            return False
    
    def _update_video_frame(self):
        """비디오 프레임 업데이트"""
        if self.video_cap is None or not self.video_cap.isOpened():
            self.stop_video()
            return
        
        ret, frame = self.video_cap.read()
        
        if ret:
            # OpenCV BGR을 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # numpy array를 QImage로 변환
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # QPixmap으로 변환하여 시그널 발송
            pixmap = QPixmap.fromImage(q_image)
            self.ad_video_frame.emit(pixmap)
        else:
            # 비디오 재생 완료 - 처음부터 다시 재생 (반복)
            print("[AdsContent] 비디오 재생 완료 - 반복 재생")
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 처음으로 되돌리기
    
    def stop_video(self):
        """비디오 재생 중지"""
        if self.video_timer is not None:
            self.video_timer.stop()
            self.video_timer = None
        
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        
        # 비디오 중지 시그널 발송
        self.ad_video_stopped.emit()
        print("[AdsContent] 비디오 재생 중지")
    
    def _load_and_show_ad_image(self, age_group, gender_en):
        """
        광고 이미지 로드 및 표시
        
        Args:
            age_group: 연령대 ("20", "30", "40", "50")
            gender_en: 성별 영문 ("female", "male")
            
        Returns:
            bool: 이미지 로드 성공 여부
        """
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.mp4']:
            image_filename = f"{age_group}_{gender_en}{ext}"
            image_path = os.path.join(self.ad_base_path, image_filename)
            
            print(f"[AdsContent] 이미지 시도: {image_path}")
            
            if os.path.exists(image_path):
                # 이미지 로드
                pixmap = QPixmap(image_path)
                
                if not pixmap.isNull():
                    # 이미지 준비 완료 시그널 발송
                    self.ad_image_ready.emit(pixmap)
                    
                    print(f"[AdsContent] ✓ 이미지 표시 성공: {image_filename}")
                    return True
                else:
                    error_msg = f"이미지를 로드할 수 없습니다.\n파일: {image_path}"
                    print(f"[AdsContent] ❌ {error_msg}")
                    self.error_occurred.emit(error_msg)
                    return False
        
        # 이미지를 찾지 못함
        error_msg = self._build_image_not_found_error(age_group, gender_en)
        print(f"[AdsContent] ❌ 이미지를 찾을 수 없음")
        self.error_occurred.emit(error_msg)
        return False
    
    def _build_image_not_found_error(self, age_group, gender_en):
        """이미지/비디오를 찾지 못했을 때 에러 메시지 생성"""
        try:
            available_files = os.listdir(self.ad_base_path)
            # 이미지와 비디오 파일 모두 검색
            ad_files = [f for f in available_files 
                       if any(f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.mp4'])]
        except:
            ad_files = []
        
        error_msg = f"광고 콘텐츠를 찾을 수 없습니다.\n\n"
        
        if ad_files:
            error_msg += f"사용 가능한 광고 파일:\n"
            for f in ad_files[:5]:
                error_msg += f"  - {f}\n"
            if len(ad_files) > 5:
                error_msg += f"  ... 외 {len(ad_files) - 5}개\n"
        else:
            error_msg += "광고 파일이 없습니다.\n"
        
        return error_msg
    
    def _start_llm_inference_async(self, age_group, gender, age):
        """
        LLM 추론 비동기 시작
        
        Args:
            age_group: 연령대
            gender: 성별
            age: 실제 나이
        """
        # 로딩 메시지 먼저 표시
        loading_msg = "🔄 AI가 광고를 분석하는 중입니다...\n잠시만 기다려주세요."
        self.llm_text_ready.emit(loading_msg)
        
        print("\n[AdsContent] LLM 추론 워커 스레드 시작")
        
        # LLM 추론 워커 스레드 생성 및 시작
        self.llm_worker = LLMInferenceWorkerThread(
            self.llm_manager,
            age_group,
            gender,
            age
        )
        
        # 시그널 연결
        self.llm_worker.result_ready.connect(self._on_llm_result_ready)
        self.llm_worker.error_occurred.connect(self._on_llm_error)
        
        # 스레드 시작 (백그라운드에서 추론 실행)
        self.llm_worker.start()
        print("[AdsContent] ✓ 백그라운드 실행 시작")
    
    def _on_llm_result_ready(self, result):
        """LLM 추론 완료 시 호출되는 슬롯"""
        print(f"[AdsContent] LLM 결과 받음 - 길이: {len(result)} 글자")
        
        # 결과 텍스트 생성
        explanation = "=== 광고 추천 이유 (AI 분석) ===\n\n"
        explanation += result
        
        # 시그널 발송
        self.llm_text_ready.emit(explanation)
        print("[AdsContent] ✓ LLM 결과 전달 완료")
        
        # 워커 스레드 정리
        self.llm_worker = None
    
    def _on_llm_error(self, error_msg):
        """LLM 추론 에러 발생 시 호출되는 슬롯"""
        print(f"[AdsContent] LLM 에러: {error_msg}")
        
        # 에러 메시지와 함께 기본 설명 표시
        explanation = "=== 광고 추천 이유 ===\n\n"
        explanation += f"⚠️ {error_msg}\n\n"
        explanation += "기본 설명을 표시합니다:\n\n"
        
        # 기본 설명은 마지막으로 사용된 정보로 생성할 수 없으므로 에러만 표시
        self.llm_text_ready.emit(explanation)
        
        # 워커 스레드 정리
        self.llm_worker = None
    
    def _show_default_explanation(self, age_group, gender, actual_age):
        """기본 설명을 즉시 표시 (LLM 초기화 실패 시)"""
        explanation = "=== 광고 추천 이유 ===\n\n"
        explanation += "⚠️ LLM 모델을 사용할 수 없습니다.\n"
        explanation += "기본 설명을 표시합니다.\n\n"
        explanation += self._get_default_explanation_text(age_group, gender, actual_age)
        
        self.llm_text_ready.emit(explanation)
    
    def _get_default_explanation_text(self, age_group, gender, actual_age):
        """기본 광고 추천 설명 생성"""
        return f"""탐지된 고객 정보:
• 실제 나이: {actual_age}세
• 연령대: {age_group}대
• 성별: {gender}

추천 근거:
{age_group}대 {gender} 고객을 위한 맞춤형 광고입니다.
이 연령대와 성별의 고객들이 선호하는 제품/서비스를 
기반으로 선정되었습니다.

타겟팅 분석:
• 연령 그룹: {age_group}대 ({age_group}세 ~ {int(age_group)+9}세)
• 성별 타겟: {gender}
• 실제 탐지 나이: {actual_age}세

※ LLM 모델을 사용하려면 모델 경로를 확인하세요.
현재 경로: {self.llm_model_path}"""
    
    def _get_age_group(self, age):
        """
        나이를 연령대로 변환 (20대, 30대, 40대, 50대)
        
        Args:
            age: 실제 나이
            
        Returns:
            str: 연령대 ("20", "30", "40", "50")
        """
        if age is None:
            return None
        
        if 20 <= age < 30:
            return "20"
        elif 30 <= age < 40:
            return "30"
        elif 40 <= age < 50:
            return "40"
        elif 50 <= age < 60:
            return "50"
        else:
            # 20대 미만이나 50대 이상은 가장 가까운 그룹으로
            if age < 20:
                return "20"
            else:
                return "50"
    
    def stop_llm_inference(self):
        """LLM 추론 중단 (필요 시)"""
        if self.llm_worker is not None and self.llm_worker.isRunning():
            print("[AdsContent] LLM 추론 워커 스레드 중단 대기...")
            self.llm_worker.wait(2000)  # 2초 대기
            if self.llm_worker.isRunning():
                self.llm_worker.terminate()
                print("[AdsContent] LLM 추론 워커 스레드 강제 종료")
            self.llm_worker = None
    
    def dispose(self):
        """리소스 정리"""
        print("[AdsContent] 리소스 정리")
        
        # 비디오 재생 중지
        self.stop_video()
        
        # LLM 추론 중단
        self.stop_llm_inference()
        
        # LLM 리소스 정리
        if self.llm_manager is not None:
            self.llm_manager.dispose()
            self.llm_manager = None
        
        self.is_llm_initialized = False


# ============================================================
# MainWindow: 메인 GUI 윈도우
# ============================================================
class MainWindow(QMainWindow):
    """메인 GUI 윈도우 클래스"""
    
    def __init__(self):
        super().__init__()
        
        # ========================================
        # 모듈 객체 생성
        # ========================================
        
        # 1. 실시간 탐지 모듈
        self.detecter = RealTimeDetecter(
            face_model_path=FACE_MODEL_PATH,
            age_gender_model_path=AGE_GENDER_MODEL_PATH
        )
        
        # 2. 광고 콘텐츠 모듈
        self.ads_content = AdsContent(
            ad_base_path=ADS_PATH,
            llm_model_path=LLM_MODEL_PATH
        )
        
        # ========================================
        # 비디오 재생 상태 (OpenCV 방식)
        # ========================================
        self.is_video_mode = False  # 현재 비디오 모드인지 여부
        
        # ========================================
        # UI 초기화
        # ========================================
        self.initUI()
        
        # ========================================
        # 시그널 연결
        # ========================================
        self._connect_signals()
        
        # ========================================
        # GUI 표시 후 모든 모델 로드
        # ========================================
        QTimer.singleShot(100, self.load_all_models_at_startup)
    
    def initUI(self):
        """UI 초기화"""
        # 메인 윈도우 설정
        self.setWindowTitle('System Modules - 나이/성별 타겟 광고')
        self.setGeometry(100, 100, 1200, 800)
        
        # 중앙 위젯 생성
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃 (수평)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 왼쪽 컬럼 생성 (실시간 웹캠 + 탐지 정보)
        left_column = self._create_left_column()
        
        # 오른쪽 컬럼 생성 (광고 화면 + LLM 텍스트)
        right_column = self._create_right_column()
        
        # 메인 레이아웃에 컬럼 추가 (비율 1:2)
        main_layout.addLayout(left_column, 1)
        main_layout.addLayout(right_column, 2)
    
    def _connect_signals(self):
        """시그널 연결"""
        # RealTimeDetecter 시그널 연결
        self.detecter.frame_updated.connect(self._on_frame_updated)
        self.detecter.status_updated.connect(self._on_status_updated)
        self.detecter.detection_result_updated.connect(self._on_detection_result_updated)
        self.detecter.age_gender_extracted.connect(self._on_age_gender_extracted)
        
        # AdsContent 시그널 연결
        self.ads_content.ad_image_ready.connect(self._on_ad_image_ready)
        self.ads_content.ad_video_ready.connect(self._on_ad_video_ready)
        self.ads_content.ad_video_frame.connect(self._on_ad_video_frame)
        self.ads_content.ad_video_stopped.connect(self._on_ad_video_stopped)
        self.ads_content.llm_text_ready.connect(self._on_llm_text_ready)
        self.ads_content.error_occurred.connect(self._on_ads_error)
    
    # ============================================================
    # 모델 로딩
    # ============================================================
    
    def load_all_models_at_startup(self):
        """앱 시작 시 모든 모델 로드 (LLM + CV 모델)"""
        print("\n" + "="*70)
        print("[시작] 모든 모델 로딩 시작")
        print("="*70)
        
        # 모든 버튼 비활성화
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.show_ad_button.setEnabled(False)
        
        # ========================================
        # 1단계: LLM 모델 로드
        # ========================================
        self._update_loading_status("🔄 모델 로딩 중...", "1/2: LLM 모델 로딩 중...")
        
        llm_success = self.ads_content.initialize_llm()
        
        if not llm_success:
            self._update_loading_status(
                "⚠️ LLM 로딩 실패", 
                "LLM 없이 계속 진행합니다..."
            )
        else:
            self._update_loading_status(
                "✅ LLM 로딩 완료", 
                "2/2: CV 모델 로딩 중..."
            )
        
        QApplication.processEvents()
        
        # ========================================
        # 2단계: CV 모델 로드 (얼굴 탐지 + 나이/성별)
        # ========================================
        cv_success = self.detecter.initialize_models()
        
        if not cv_success:
            self._update_loading_status(
                "❌ CV 모델 로딩 실패", 
                "얼굴 탐지를 사용할 수 없습니다"
            )
            QMessageBox.critical(
                self, "오류",
                "CV 모델 초기화에 실패했습니다.\n"
                "얼굴 탐지 기능을 사용할 수 없습니다.\n\n"
                "프로그램을 종료합니다."
            )
            self.close()
            return
        
        # ========================================
        # 3단계: 모든 로딩 완료
        # ========================================
        print("\n" + "="*70)
        print("[완료] 모든 모델 로딩 완료!")
        print("="*70 + "\n")
        
        # UI 업데이트
        self._update_loading_status(
            "✅ 모든 모델 로딩 완료!", 
            "웹캠을 시작할 수 있습니다"
        )
        
        # LLM 텍스트 영역 업데이트
        if self.ads_content.is_llm_initialized:
            self.llm_text.setText(
                "✅ AI 모델 로딩 완료!\n\n"
                "광고 추천 이유가 여기에 표시됩니다.\n\n"
                "1. 웹캠을 시작하세요\n"
                "2. 얼굴이 감지되면\n"
                "3. '탐지된 타겟 광고 표시' 버튼을 눌러주세요"
            )
        else:
            self.llm_text.setText(
                "⚠️ AI 모델을 사용할 수 없습니다\n\n"
                "기본 광고 추천 설명이 표시됩니다.\n\n"
                "1. 웹캠을 시작하세요\n"
                "2. 얼굴이 감지되면\n"
                "3. '탐지된 타겟 광고 표시' 버튼을 눌러주세요"
            )
        
        # 탐지 정보 영역 업데이트
        self.detection_text.setText(
            "✅ 나이/성별 탐지 모델 로딩 완료!\n\n"
            "웹캠을 시작하면 실시간으로\n"
            "얼굴의 나이와 성별을 탐지합니다."
        )
        
        # 웹캠 시작 버튼 활성화
        self.start_button.setEnabled(True)
        self.start_button.setText("웹캠 시작")
        self.show_ad_button.setEnabled(True)
        
        print("[시작] GUI 준비 완료 - 사용자가 웹캠을 시작할 수 있습니다")
    
    def _update_loading_status(self, status_text, llm_text):
        """로딩 상태 업데이트"""
        self.status_label.setText(status_text)
        
        if "완료" in status_text or "✅" in status_text:
            self.status_label.setStyleSheet("color: green; padding: 5px; font-weight: bold;")
        elif "실패" in status_text or "❌" in status_text or "⚠️" in status_text:
            self.status_label.setStyleSheet("color: red; padding: 5px; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: orange; padding: 5px; font-weight: bold;")
        
        self.llm_text.setText(llm_text)
        QApplication.processEvents()
    
    # ============================================================
    # UI 생성 메서드
    # ============================================================
    
    def _create_left_column(self):
        """왼쪽 컬럼 생성 (실시간 웹캠 화면 + 나이/성별 탐지 정보)"""
        left_layout = QVBoxLayout()
        
        # 1. 실시간 웹캠 화면
        camera_frame = self._create_camera_frame()
        
        # 2. 나이/성별 탐지 정보 텍스트
        detection_frame = self._create_detection_frame()
        
        # 레이아웃에 추가 (비율 3:2)
        left_layout.addWidget(camera_frame, 3)
        left_layout.addWidget(detection_frame, 2)
        
        return left_layout
    
    def _create_camera_frame(self):
        """웹캠 화면 프레임 생성"""
        frame = QFrame()
        frame.setFrameShape(QFrame.Box)
        frame.setFrameShadow(QFrame.Plain)
        frame.setLineWidth(2)
        
        # 프레임 내부 레이아웃
        layout = QVBoxLayout()
        frame.setLayout(layout)
        
        # 제목
        title_label = QLabel("실시간 웹캠 화면 (나이/성별 탐지 자동 활성화)")
        title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
        # 카메라 선택 드롭다운
        camera_select_layout = QHBoxLayout()
        camera_label = QLabel("카메라 선택:")
        self.camera_combo = QComboBox()
        
        # 사용 가능한 카메라 검색
        available_cameras = CameraManager.get_available_cameras()
        if available_cameras:
            for cam_id in available_cameras:
                self.camera_combo.addItem(f"카메라 {cam_id}", cam_id)
        else:
            self.camera_combo.addItem("카메라를 찾을 수 없음", -1)
        
        camera_select_layout.addWidget(camera_label)
        camera_select_layout.addWidget(self.camera_combo)
        layout.addLayout(camera_select_layout)
        
        # 비디오 표시 라벨
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setScaledContents(True)
        layout.addWidget(self.video_label, stretch=1)
        
        # 상태 표시 라벨
        self.status_label = QLabel("모델을 로딩하고 있습니다...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: orange; padding: 5px; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        
        # 시작 버튼
        self.start_button = QPushButton("웹캠 시작 (모델 로딩 중...)")
        self.start_button.clicked.connect(self._on_start_camera_clicked)
        self.start_button.setEnabled(False)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-size: 12pt;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.start_button)
        
        # 중지 버튼
        self.stop_button = QPushButton("웹캠 중지")
        self.stop_button.clicked.connect(self._on_stop_camera_clicked)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 8px;
                font-size: 12pt;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        layout.setContentsMargins(10, 10, 10, 10)
        
        return frame
    
    def _create_detection_frame(self):
        """나이/성별 탐지 정보 프레임 생성"""
        frame = QFrame()
        frame.setFrameShape(QFrame.Box)
        frame.setFrameShadow(QFrame.Plain)
        frame.setLineWidth(2)
        
        # 프레임 내부 레이아웃
        layout = QVBoxLayout()
        frame.setLayout(layout)
        
        # 제목
        title_label = QLabel("나이/성별 탐지 정보")
        title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
        # 탐지 결과 텍스트 영역
        self.detection_text = QTextEdit()
        self.detection_text.setReadOnly(True)
        self.detection_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                padding: 10px;
                font-family: monospace;
                font-size: 11pt;
            }
        """)
        self.detection_text.setText("탐지 결과가 여기에 표시됩니다.\n\n탐지 기능을 활성화하고 웹캠을 시작하세요.")
        layout.addWidget(self.detection_text)
        
        layout.setContentsMargins(10, 10, 10, 10)
        
        return frame
    
    def _create_right_column(self):
        """오른쪽 컬럼 생성 (광고 화면 + LLM 텍스트)"""
        right_layout = QVBoxLayout()
        
        # 1. 광고 화면
        ad_frame = self._create_ad_frame()
        
        # 2. 광고 추천 이유 LLM 텍스트
        llm_frame = self._create_llm_frame()
        
        # 레이아웃에 추가 (비율 2:1)
        right_layout.addWidget(ad_frame, 2)
        right_layout.addWidget(llm_frame, 1)
        
        return right_layout
    
    def _create_ad_frame(self):
        """광고 화면 프레임 생성"""
        frame = QFrame()
        frame.setFrameShape(QFrame.Box)
        frame.setFrameShadow(QFrame.Plain)
        frame.setLineWidth(2)
        
        # 프레임 내부 레이아웃
        layout = QVBoxLayout()
        frame.setLayout(layout)
        
        # 제목
        title_label = QLabel("타겟 광고 화면")
        title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
        # 광고 표시 라벨 (이미지와 비디오 모두 표시)
        self.ad_image_label = QLabel()
        self.ad_image_label.setAlignment(Qt.AlignCenter)
        self.ad_image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.ad_image_label.setMinimumSize(400, 300)
        self.ad_image_label.setScaledContents(True)  # 비디오 프레임을 위해 True로 설정
        self.ad_image_label.setText("광고가 여기에 표시됩니다.")
        layout.addWidget(self.ad_image_label, stretch=1)
        
        # 광고 표시 버튼
        self.show_ad_button = QPushButton("탐지된 타겟 광고 표시")
        self.show_ad_button.clicked.connect(self._on_show_ad_clicked)
        self.show_ad_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                font-size: 14pt;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        layout.addWidget(self.show_ad_button)
        
        layout.setContentsMargins(10, 10, 10, 10)
        
        return frame
    
    def _create_llm_frame(self):
        """LLM 텍스트 프레임 생성"""
        frame = QFrame()
        frame.setFrameShape(QFrame.Box)
        frame.setFrameShadow(QFrame.Plain)
        frame.setLineWidth(2)
        
        # 프레임 내부 레이아웃
        layout = QVBoxLayout()
        frame.setLayout(layout)
        
        # 제목
        title_label = QLabel("광고 추천 이유 (LLM 분석)")
        title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
        # LLM 텍스트 표시 영역
        self.llm_text = QTextEdit()
        self.llm_text.setReadOnly(True)
        self.llm_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                padding: 10px;
                font-family: 'NanumGothic';
                font-size: 10pt;
                line-height: 1.5;
            }
        """)
        self.llm_text.setText("🔄 AI 모델을 로딩하고 있습니다...\n\n"
                             "처음 실행 시 시간이 걸릴 수 있습니다.\n"
                             "모델 로딩이 완료되면 웹캠을 시작할 수 있습니다.")
        layout.addWidget(self.llm_text)
        
        layout.setContentsMargins(10, 10, 10, 10)
        
        return frame
    
    # ============================================================
    # 버튼 이벤트 핸들러
    # ============================================================
    
    def _on_start_camera_clicked(self):
        """웹캠 시작 버튼 클릭"""
        camera_id = self.camera_combo.currentData()
        
        # 유효한 카메라 ID 확인
        if camera_id == -1:
            QMessageBox.warning(self, "오류", "사용 가능한 카메라가 없습니다.")
            return
        
        # 카메라 시작
        success = self.detecter.start_camera(camera_id)
        
        if success:
            # 버튼 상태 변경
            self.start_button.setEnabled(False)
            self.start_button.setText("웹캠 시작")
            self.stop_button.setEnabled(True)
            self.camera_combo.setEnabled(False)
            
            # 탐지 정보 초기화
            self.detection_text.clear()
            self.detection_text.setText("웹캠이 시작되었습니다.\n\n카메라 앞에 얼굴을 보여주세요...")
        else:
            QMessageBox.critical(self, "오류", 
                               "나이/성별 탐지 모델이 로드되지 않았습니다.\n"
                               "프로그램을 재시작해주세요.")
    
    def _on_stop_camera_clicked(self):
        """웹캠 중지 버튼 클릭"""
        self.detecter.stop_camera()
        
        # 화면 초기화
        self.video_label.clear()
        self.video_label.setText("웹캠이 중지되었습니다.")
        self.status_label.setText("웹캠이 연결되지 않았습니다.")
        self.status_label.setStyleSheet("color: gray; padding: 5px;")
        
        # 탐지 결과 초기화
        if self.detecter.is_initialized:
            self.detection_text.append("\n\n웹캠이 중지되었습니다.")
        
        # 버튼 상태 변경
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_combo.setEnabled(True)
    
    def _on_show_ad_clicked(self):
        """광고 표시 버튼 클릭"""
        # 현재 탐지된 나이/성별 가져오기
        age, gender = self.detecter.get_current_detection()
        
        if age is None or gender is None:
            # 화면에 표시된 텍스트에서 직접 추출 시도
            displayed_text = self.detection_text.toPlainText()
            if displayed_text:
                # RealTimeDetecter의 추출 메서드 재사용
                age, gender = self.detecter._extract_age_gender_from_text(displayed_text)
        
        if age is None or gender is None:
            msg = "탐지된 나이/성별 정보가 없습니다.\n\n"
            msg += "다음을 확인해주세요:\n"
            msg += "1. '웹캠 시작' 버튼을 눌렀는지\n"
            msg += "2. 카메라 앞에 얼굴이 보이는지\n"
            msg += "3. '나이/성별 탐지 정보' 영역에 탐지 결과가 표시되는지\n"
            msg += "4. 모델이 정상적으로 로드되었는지"
            QMessageBox.warning(self, "알림", msg)
            return
        
        # 광고 표시
        self.ads_content.show_targeted_ad(age, gender)
    
    # ============================================================
    # 시그널 슬롯 (RealTimeDetecter)
    # ============================================================
    
    def _on_frame_updated(self, qt_image):
        """프레임 업데이트"""
        pixmap = QPixmap.fromImage(qt_image)
        
        # 라벨 크기에 맞게 스케일링
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def _on_status_updated(self, success, message):
        """연결 상태 업데이트"""
        self.status_label.setText(message)
        
        if success:
            self.status_label.setStyleSheet("color: green; padding: 5px;")
        else:
            self.status_label.setStyleSheet("color: red; padding: 5px;")
            QMessageBox.warning(self, "연결 오류", message)
    
    def _on_detection_result_updated(self, result_text):
        """탐지 결과 업데이트"""
        # 텍스트 영역에 결과 표시
        self.detection_text.clear()
        
        # 탐지 결과 표시
        display_text = result_text
        
        # 현재 저장된 값이 있으면 추가 안내 메시지 표시
        age, gender = self.detecter.get_current_detection()
        if age is not None and gender is not None:
            display_text += f"\n✓ '탐지된 타겟 광고 표시' 버튼을 눌러\n  이 정보에 맞는 광고를 볼 수 있습니다."
        
        self.detection_text.setText(display_text)
    
    def _on_age_gender_extracted(self, age, gender):
        """나이/성별 추출 완료"""
        print(f"[MainWindow] 나이/성별 추출: {age}세, {gender}")
        # 필요시 추가 처리
    
    # ============================================================
    # 시그널 슬롯 (AdsContent)
    # ============================================================
    
    def _on_ad_image_ready(self, pixmap):
        """광고 이미지 준비 완료"""
        # 비디오 재생 중이면 중지
        if self.is_video_mode:
            self.ads_content.stop_video()
            self.is_video_mode = False
        
        # 라벨 크기에 맞게 스케일링
        scaled_pixmap = pixmap.scaled(
            self.ad_image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.ad_image_label.setPixmap(scaled_pixmap)
        self.ad_image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        print("[MainWindow] 광고 이미지 표시 완료")
    
    def _on_ad_video_ready(self, video_path):
        """광고 비디오 준비 완료 (OpenCV 방식)"""
        print(f"[MainWindow] 광고 비디오 재생 시작: {video_path}")
        
        # 비디오 모드로 전환
        self.is_video_mode = True
        
        # 배경을 검은색으로 설정 (비디오 재생 중)
        self.ad_image_label.setStyleSheet("background-color: black; border: 1px solid #ccc;")
        self.ad_image_label.setText("")  # 텍스트 제거
    
    def _on_ad_video_frame(self, pixmap):
        """비디오 프레임 업데이트"""
        if self.is_video_mode:
            # 비디오 프레임을 라벨에 표시
            self.ad_image_label.setPixmap(pixmap)
    
    def _on_ad_video_stopped(self):
        """비디오 재생 중지"""
        print("[MainWindow] 비디오 재생 중지됨")
        self.is_video_mode = False
        self.ad_image_label.clear()
        self.ad_image_label.setText("광고가 여기에 표시됩니다.")
        self.ad_image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
    
    def _on_llm_text_ready(self, text):
        """LLM 텍스트 준비 완료"""
        self.llm_text.setText(text)
        print("[MainWindow] LLM 텍스트 표시 완료")
    
    def _on_ads_error(self, error_msg):
        """광고 관련 에러 발생"""
        QMessageBox.warning(self, "오류", error_msg)
    
    # ============================================================
    # 종료 처리
    # ============================================================
    
    def closeEvent(self, event):
        """창 닫기 이벤트"""
        print("[MainWindow] 종료 시작...")
        
        # AdsContent 정리 (비디오 포함)
        self.ads_content.dispose()
        
        # RealTimeDetecter 정리
        self.detecter.dispose()
        
        print("[MainWindow] 종료 완료")
        event.accept()


# ============================================================
# 메인 실행
# ============================================================
def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 폰트 설정 (한글 지원)
    font = QFont("NanumGothic", 10)
    app.setFont(font)
    
    # GUI 실행
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()