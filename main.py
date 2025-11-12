#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                             QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                             QPushButton, QComboBox, QMessageBox, QTextEdit,
                             QCheckBox, QLineEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap

# 카메라 모듈 임포트
from webcam_connect import CameraManager
# 나이/성별 탐지 모듈 임포트
from private_info import AgeGenderDetectionManager
# LLM 추론 모듈 임포트
from llm_infer import LLMInferenceManager
# LLM 워커 스레드 임포트
from llm_worker import LLMInferenceWorkerThread

ADS_PATH = "./sample_ads"


class SystemModulesGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 카메라 매니저 초기화
        self.camera_manager = CameraManager()
        
        # 나이/성별 탐지 매니저 (GUI 표시 후 초기화)
        self.detection_manager = None
        
        # LLM 매니저 (GUI 표시 후 초기화)
        self.llm_manager = None
        
        # LLM 워커 스레드 참조
        self.llm_inference_worker = None
        
        # 현재 탐지된 나이/성별 정보 저장
        self.current_age = None
        self.current_gender = None
        
        # 고정된 경로 설정 (코드 내에서만 수정 가능)
        self.ad_base_path = ADS_PATH
        self.face_model_path = "./models/cv/yolov8n-face-lindevs.mxq"
        self.age_gender_model_path = "./models/cv/genderage.mxq"
        self.llm_model_path = "./models/llm/mblt-exaone"
        
        # UI 초기화
        self.initUI()
        
        # GUI 표시 후 모든 모델 로드
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self.load_all_models_at_startup)
    
    def initUI(self):
        # 메인 윈도우 설정
        self.setWindowTitle('System Modules - 나이/성별 타겟 광고')
        self.setGeometry(100, 100, 1200, 800)
        
        # 중앙 위젯 생성
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃 (수평)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 왼쪽 컬럼 생성
        left_column = self.create_left_column()
        
        # 오른쪽 컬럼 생성
        right_column = self.create_right_column()
        
        # 메인 레이아웃에 컬럼 추가 (비율 1:2)
        main_layout.addLayout(left_column, 1)
        main_layout.addLayout(right_column, 2)
    
    def load_all_models_at_startup(self):
        """앱 시작 시 모든 모델 로드 (LLM + CV 모델)"""
        print("\n" + "="*70)
        print("[시작] 모든 모델 로딩 시작")
        print("="*70)
        
        # 모든 버튼 비활성화
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.show_ad_button.setEnabled(False)
        
        # 상태 메시지 표시
        self.update_loading_status("🔄 모델 로딩 중...", "1/2: LLM 모델 로딩 중...")
        QApplication.processEvents()
        
        # ========================================
        # 1단계: LLM 모델 로드
        # ========================================
        print("\n[1/2] LLM 모델 로딩 시작...")
        try:
            self.llm_manager = LLMInferenceManager(
                model_path=self.llm_model_path
            )
            
            if not self.llm_manager.is_initialized:
                print("[1/2] ❌ LLM 초기화 실패")
                self.update_loading_status(
                    "⚠️ LLM 로딩 실패", 
                    "LLM 없이 계속 진행합니다..."
                )
                self.llm_manager = None
            else:
                print("[1/2] ✅ LLM 로딩 완료")
                self.update_loading_status(
                    "✅ LLM 로딩 완료", 
                    "2/2: CV 모델 로딩 중..."
                )
        
        except Exception as e:
            print(f"[1/2] ❌ LLM 로딩 예외: {e}")
            self.update_loading_status(
                "⚠️ LLM 로딩 실패", 
                "LLM 없이 계속 진행합니다..."
            )
            self.llm_manager = None
        
        QApplication.processEvents()
        
        # ========================================
        # 2단계: CV 모델 로드 (얼굴 탐지 + 나이/성별)
        # ========================================
        print("\n[2/2] CV 모델 로딩 시작...")
        try:
            self.detection_manager = AgeGenderDetectionManager(
                face_model_path=self.face_model_path,
                age_gender_model_path=self.age_gender_model_path
            )
            
            if not self.detection_manager.is_initialized:
                print("[2/2] ❌ CV 모델 초기화 실패")
                self.update_loading_status(
                    "❌ CV 모델 로딩 실패", 
                    "얼굴 탐지를 사용할 수 없습니다"
                )
                QMessageBox.critical(
                    self, "오류",
                    "CV 모델 초기화에 실패했습니다.\n"
                    "얼굴 탐지 기능을 사용할 수 없습니다."
                )
                self.detection_manager = None
                # CV 모델 없이는 앱을 사용할 수 없으므로 종료
                self.close()
                return
            else:
                print("[2/2] ✅ CV 모델 로딩 완료")
        
        except Exception as e:
            print(f"[2/2] ❌ CV 모델 로딩 예외: {e}")
            import traceback
            traceback.print_exc()
            
            self.update_loading_status(
                "❌ CV 모델 로딩 실패", 
                "얼굴 탐지를 사용할 수 없습니다"
            )
            QMessageBox.critical(
                self, "오류",
                f"CV 모델 초기화에 실패했습니다:\n{str(e)}\n\n"
                "프로그램을 종료합니다."
            )
            self.detection_manager = None
            self.close()
            return
        
        # ========================================
        # 3단계: 모든 로딩 완료
        # ========================================
        print("\n" + "="*70)
        print("[완료] 모든 모델 로딩 완료!")
        print("="*70 + "\n")
        
        # UI 업데이트
        self.update_loading_status(
            "✅ 모든 모델 로딩 완료!", 
            "웹캠을 시작할 수 있습니다"
        )
        
        # LLM 텍스트 영역 업데이트
        if self.llm_manager:
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
    
    def update_loading_status(self, status_text, llm_text):
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
    
    
    def create_left_column(self):
        """왼쪽 컬럼 생성 (실시간 웹캠 화면 + 나이/성별 탐지 정보)"""
        left_layout = QVBoxLayout()
        
        # 1. 실시간 웹캠 화면
        chatbot_frame = self.create_camera_frame()
        
        # 2. 나이/성별 탐지 정보 텍스트
        detection_frame = self.create_detection_frame()
        
        # 레이아웃에 추가 (비율 3:2)
        left_layout.addWidget(chatbot_frame, 3)
        left_layout.addWidget(detection_frame, 2)
        
        return left_layout
    
    def create_camera_frame(self):
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
        
        # 시작 버튼 (초기에는 비활성화 - 모델 로딩 대기)
        self.start_button = QPushButton("웹캠 시작 (모델 로딩 중...)")
        self.start_button.clicked.connect(self.start_camera)
        self.start_button.setEnabled(False)  # 초기에는 비활성화
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
        self.stop_button.clicked.connect(self.stop_camera)
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
        
        # 여백 설정
        layout.setContentsMargins(10, 10, 10, 10)
        
        return frame
    
    def create_detection_frame(self):
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
        
        # 여백 설정
        layout.setContentsMargins(10, 10, 10, 10)
        
        return frame
    
    def start_camera(self):
        """웹캠 시작 (나이/성별 탐지 자동 활성화)"""
        camera_id = self.camera_combo.currentData()
        
        # 유효한 카메라 ID 확인
        if camera_id == -1:
            QMessageBox.warning(self, "오류", "사용 가능한 카메라가 없습니다.")
            return
        
        # detection_manager가 이미 로드되어 있는지 확인
        if self.detection_manager is None:
            QMessageBox.critical(self, "오류", 
                              "나이/성별 탐지 모델이 로드되지 않았습니다.\n"
                              "프로그램을 재시작해주세요.")
            return
        
        # 이미 로드된 detection_manager 사용
        print(f"\n[웹캠 시작] 카메라 {camera_id} 시작")
        print(f"[웹캠 시작] 나이/성별 탐지 모델: 이미 로드됨 ✓")
        
        # 카메라 시작
        camera_thread = self.camera_manager.start_camera(
            camera_id=camera_id,
            detection_manager=self.detection_manager
        )
        
        # 시그널 연결
        camera_thread.frame_update.connect(self.update_frame)
        camera_thread.connection_status.connect(self.update_status)
        
        # 탐지 결과 시그널 연결
        camera_thread.detection_result.connect(self.update_detection_result)
        
        # 버튼 상태 변경
        self.start_button.setEnabled(False)
        self.start_button.setText("웹캠 시작")
        self.stop_button.setEnabled(True)
        self.camera_combo.setEnabled(False)
        
        # 탐지 정보 초기화
        self.detection_text.clear()
        self.detection_text.setText("웹캠이 시작되었습니다.\n\n카메라 앞에 얼굴을 보여주세요...")
    
    def stop_camera(self):
        """웹캠 중지"""
        self.camera_manager.stop_camera()
        
        # 화면 초기화
        self.video_label.clear()
        self.video_label.setText("웹캠이 중지되었습니다.")
        self.status_label.setText("웹캠이 연결되지 않았습니다.")
        self.status_label.setStyleSheet("color: gray; padding: 5px;")
        
        # 탐지 결과 초기화
        if self.detection_manager is not None:
            self.detection_text.append("\n\n웹캠이 중지되었습니다.")
        
        # 버튼 상태 변경
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_combo.setEnabled(True)
    
    def update_frame(self, qt_image):
        """프레임 업데이트"""
        pixmap = QPixmap.fromImage(qt_image)
        
        # 라벨 크기에 맞게 스케일링
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_status(self, success, message):
        """연결 상태 업데이트"""
        self.status_label.setText(message)
        
        if success:
            self.status_label.setStyleSheet("color: green; padding: 5px;")
        else:
            self.status_label.setStyleSheet("color: red; padding: 5px;")
            QMessageBox.warning(self, "연결 오류", message)
    
    def update_detection_result(self, result_text):
        """탐지 결과 업데이트"""
        # 탐지 결과에서 나이/성별 정보 추출
        self.extract_age_gender_from_result(result_text)
        
        # 텍스트 영역에 결과 표시
        self.detection_text.clear()
        
        # 탐지 결과와 현재 저장된 값 함께 표시
        display_text = result_text
        
        if self.current_age is not None and self.current_gender is not None:
            display_text += f"\n✓ '탐지된 타겟 광고 표시' 버튼을 눌러\n  이 정보에 맞는 광고를 볼 수 있습니다."
        
        self.detection_text.setText(display_text)
    
    def extract_age_gender_from_result(self, result_text):
        """탐지 결과 텍스트에서 나이/성별 정보 추출 (정규표현식 사용)"""
        import re
        
        try:
            # 패턴 1: "얼굴 N: 성별, 나이세 (신뢰도: 0.xx)" 형식
            # 예: "얼굴 1: 여성, 32세 (신뢰도: 0.85)"
            pattern = r'얼굴\s+(\d+):\s*(여성|남성),\s*(\d+)세\s*\(신뢰도:\s*([\d.]+)\)'
            matches = re.findall(pattern, result_text)
            
            if matches:
                # 첫 번째 얼굴의 정보를 사용
                face_num, gender, age, confidence = matches[0]
                self.current_gender = gender
                self.current_age = int(age)
                
                print(f"✓ 탐지 정보 업데이트: 나이={self.current_age}세, 성별={self.current_gender}")
                print(f"  (얼굴 {face_num}, 신뢰도: {confidence})")
                return
            
            # 패턴 2: "나이: 25, 성별: 여성" 형식
            pattern_alt1 = r'나이:\s*([\d.]+),\s*성별:\s*(여성|남성)'
            match_alt1 = re.search(pattern_alt1, result_text)
            if match_alt1:
                age_str, gender = match_alt1.groups()
                self.current_age = int(float(age_str))
                self.current_gender = gender
                print(f"✓ 탐지 정보 업데이트 (대체 형식): 나이={self.current_age}세, 성별={self.current_gender}")
                return
            
            # 패턴 3: 성별과 나이를 따로 찾기
            age_pattern = r'(?:나이|Age|age):\s*([\d.]+)'
            gender_pattern = r'(?:성별|Gender|gender):\s*(\w+)'
            
            age_match = re.search(age_pattern, result_text)
            gender_match = re.search(gender_pattern, result_text)
            
            if age_match and gender_match:
                age_str = age_match.group(1)
                gender_str = gender_match.group(1).lower()
                
                self.current_age = int(float(age_str))
                
                # 성별 매핑
                if '여' in gender_str or 'female' in gender_str:
                    self.current_gender = "여성"
                elif '남' in gender_str or 'male' in gender_str:
                    self.current_gender = "남성"
                else:
                    self.current_gender = None
                
                if self.current_gender:
                    print(f"✓ 탐지 정보 업데이트 (분리 패턴): 나이={self.current_age}세, 성별={self.current_gender}")
                    return
            
            # 모든 패턴 매칭 실패
            print(f"⚠ 탐지 정보 추출 실패")
            print(f"원본 텍스트:\n{result_text[:300]}")
            self.current_age = None
            self.current_gender = None
                
        except Exception as e:
            print(f"나이/성별 정보 추출 오류: {e}")
            import traceback
            traceback.print_exc()
            self.current_age = None
            self.current_gender = None
    
    def get_age_group(self, age):
        """나이를 연령대로 변환 (20대, 30대, 40대, 50대)"""
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
    
    def show_targeted_ad(self):
        """현재 탐지된 나이/성별에 맞는 광고 표시"""
        # 화면에 표시된 탐지 정보를 직접 파싱
        displayed_text = self.detection_text.toPlainText()
        
        if displayed_text:
            # 표시된 텍스트에서 나이/성별 정보 추출
            self.extract_age_gender_from_result(displayed_text)
        
        # 탐지 정보 확인
        if self.current_age is None or self.current_gender is None:
            msg = "탐지된 나이/성별 정보가 없습니다.\n\n"
            msg += "다음을 확인해주세요:\n"
            msg += "1. '웹캠 시작' 버튼을 눌렀는지\n"
            msg += "2. 카메라 앞에 얼굴이 보이는지\n"
            msg += "3. '나이/성별 탐지 정보' 영역에 탐지 결과가 표시되는지\n"
            msg += "4. 모델이 정상적으로 로드되었는지"
            QMessageBox.warning(self, "알림", msg)
            return
        
        # 연령대 결정
        age_group = self.get_age_group(self.current_age)
        
        # 성별을 영문으로 변환
        gender = "female" if self.current_gender == "여성" else "male"
        
        print(f"\n[광고 표시] 타겟: {age_group}대 {self.current_gender} (나이: {self.current_age}세)")
        print(f"[광고 표시] 광고 경로: {self.ad_base_path}")
        
        # 광고 경로 확인
        if not os.path.exists(self.ad_base_path):
            QMessageBox.warning(self, "오류", 
                              f"광고 디렉토리를 찾을 수 없습니다.\n경로: {self.ad_base_path}\n\n"
                              f"디렉토리를 생성하거나 경로를 확인해주세요.")
            return
        
        # ========================================
        # 1단계: 광고 이미지 먼저 표시 (즉시 실행)
        # ========================================
        image_found = False
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            image_filename = f"{age_group}_{gender}{ext}"
            image_path = os.path.join(self.ad_base_path, image_filename)
            
            print(f"[광고 표시] 시도: {image_path}")
            
            if os.path.exists(image_path):
                # 이미지 로드 및 표시
                pixmap = QPixmap(image_path)
                
                if not pixmap.isNull():
                    # 광고 라벨에 이미지 표시
                    scaled_pixmap = pixmap.scaled(
                        self.ad_image_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.ad_image_label.setPixmap(scaled_pixmap)
                    
                    print(f"[광고 표시] ✓ 이미지 표시 성공: {image_filename}")
                    
                    # 성공 메시지 (선택적)
                    self.ad_image_label.setToolTip(f"표시된 광고: {image_filename}")
                    
                    image_found = True
                    break
                else:
                    QMessageBox.warning(self, "오류", 
                                      f"이미지를 로드할 수 없습니다.\n파일: {image_path}\n\n"
                                      f"파일이 손상되었거나 지원하지 않는 형식일 수 있습니다.")
                    return
        
        if not image_found:
            # 사용 가능한 파일 목록 확인
            try:
                available_files = os.listdir(self.ad_base_path)
                ad_files = [f for f in available_files if any(f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp'])]
            except:
                ad_files = []
            
            error_msg = f"광고 이미지를 찾을 수 없습니다.\n\n"
            error_msg += f"필요한 파일: {age_group}_{gender}.[jpg|jpeg|png|gif|bmp]\n"
            error_msg += f"검색 경로: {self.ad_base_path}\n\n"
            
            if ad_files:
                error_msg += f"사용 가능한 광고 이미지:\n"
                for f in ad_files[:5]:  # 최대 5개만 표시
                    error_msg += f"  - {f}\n"
                if len(ad_files) > 5:
                    error_msg += f"  ... 외 {len(ad_files) - 5}개\n"
            else:
                error_msg += "광고 이미지가 없습니다.\n"
                error_msg += "'create_sample_ads.py'를 실행하여 샘플 이미지를 생성하세요."
            
            QMessageBox.warning(self, "오류", error_msg)
            print(f"[광고 표시] 실패: 이미지를 찾을 수 없음")
            return
        
        # ========================================
        # 2단계: LLM 텍스트는 백그라운드에서 생성
        # ========================================
        
        # LLM이 없으면 기본 설명 표시
        if self.llm_manager is None:
            print("\n[광고 표시] LLM 모델이 없음 - 기본 설명 표시")
            self.show_default_explanation(age_group, self.current_gender, self.current_age)
            return
        
        # LLM 텍스트 영역에 로딩 메시지 먼저 표시
        self.llm_text.clear()
        self.llm_text.setText("🔄 AI가 광고를 분석하는 중입니다...\n잠시만 기다려주세요.")
        print("\n[LLM 추론] 워커 스레드에서 추론 시작")
        
        # LLM 추론 워커 스레드 생성 및 시작
        print("[LLM 추론] 워커 스레드 생성 및 시작")
        self.llm_inference_worker = LLMInferenceWorkerThread(
            self.llm_manager,
            age_group,
            self.current_gender,
            self.current_age
        )
        
        # 시그널 연결
        self.llm_inference_worker.result_ready.connect(self.on_llm_result_ready)
        self.llm_inference_worker.error_occurred.connect(self.on_llm_error)
        
        # 스레드 시작 (백그라운드에서 추론 실행)
        self.llm_inference_worker.start()
        print("[LLM 추론] ✓ 백그라운드 실행 시작 - GUI는 계속 응답 가능")
    
    def on_llm_result_ready(self, result):
        """LLM 추론 완료 시 호출되는 슬롯"""
        print(f"[LLM 결과] 받음 - 길이: {len(result)} 글자")
        
        # 결과 텍스트 표시
        explanation = "=== 광고 추천 이유 (AI 분석) ===\n\n"
        explanation += result
        
        self.llm_text.setText(explanation)
        print("[LLM 결과] ✓ UI 업데이트 완료")
        
        # 워커 스레드 정리
        self.llm_inference_worker = None
    
    def on_llm_error(self, error_msg):
        """LLM 추론 에러 발생 시 호출되는 슬롯"""
        print(f"[LLM 에러] {error_msg}")
        
        # 에러 메시지와 함께 기본 설명 표시
        explanation = "=== 광고 추천 이유 ===\n\n"
        explanation += f"⚠️ {error_msg}\n\n"
        explanation += "기본 설명을 표시합니다:\n\n"
        
        # 현재 타겟 정보를 사용하여 기본 설명 생성
        age_group = self.get_age_group(self.current_age)
        explanation += self._get_default_explanation(age_group, self.current_gender, self.current_age)
        
        self.llm_text.setText(explanation)
        
        # 워커 스레드 정리
        self.llm_inference_worker = None
    
    def show_default_explanation(self, age_group, gender, actual_age):
        """기본 설명을 즉시 표시 (LLM 초기화 실패 시)"""
        explanation = "=== 광고 추천 이유 ===\n\n"
        explanation += "⚠️ LLM 모델을 사용할 수 없습니다.\n"
        explanation += "기본 설명을 표시합니다.\n\n"
        explanation += self._get_default_explanation(age_group, gender, actual_age)
        
        self.llm_text.setText(explanation)
    
    def _get_default_explanation(self, age_group, gender, actual_age):
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
    
    def create_right_column(self):
        """오른쪽 컬럼 생성 (광고 화면 + LLM 텍스트)"""
        right_layout = QVBoxLayout()
        
        # 1. 광고 화면
        ad_frame = self.create_ad_frame()
        
        # 2. 광고 추천 이유 LLM 텍스트
        llm_frame = self.create_llm_frame()
        
        # 레이아웃에 추가 (비율 2:1)
        right_layout.addWidget(ad_frame, 2)
        right_layout.addWidget(llm_frame, 1)
        
        return right_layout
    
    def create_ad_frame(self):
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
        
        # 광고 이미지 표시 라벨
        self.ad_image_label = QLabel()
        self.ad_image_label.setAlignment(Qt.AlignCenter)
        self.ad_image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.ad_image_label.setMinimumSize(400, 300)
        self.ad_image_label.setScaledContents(False)  # 비율 유지
        self.ad_image_label.setText("광고가 여기에 표시됩니다.")
        layout.addWidget(self.ad_image_label, stretch=1)
        
        # 광고 표시 버튼
        self.show_ad_button = QPushButton("탐지된 타겟 광고 표시")
        self.show_ad_button.clicked.connect(self.show_targeted_ad)
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
        
        # 여백 설정
        layout.setContentsMargins(10, 10, 10, 10)
        
        return frame
    
    def create_llm_frame(self):
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
        
        # 여백 설정
        layout.setContentsMargins(10, 10, 10, 10)
        
        return frame
    
    def closeEvent(self, event):
        """창 닫기 이벤트"""
        # LLM 추론 워커 스레드 정리
        if self.llm_inference_worker is not None and self.llm_inference_worker.isRunning():
            print("[종료] LLM 추론 워커 스레드 중단 대기...")
            self.llm_inference_worker.wait(2000)  # 2초 대기
            if self.llm_inference_worker.isRunning():
                self.llm_inference_worker.terminate()
                print("[종료] LLM 추론 워커 스레드 강제 종료")
        
        # 카메라 정리
        if self.camera_manager.is_running():
            self.camera_manager.stop_camera()
        
        # LLM 리소스 정리
        if self.llm_manager is not None:
            self.llm_manager.dispose()
            self.llm_manager = None
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # 폰트 설정 (한글 지원)
    font = QFont("NanumGothic", 10)
    app.setFont(font)
    
    # GUI 실행
    gui = SystemModulesGUI()
    gui.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()