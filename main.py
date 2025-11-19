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
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
							 QVBoxLayout, QHBoxLayout, QLabel, QFrame,
							 QPushButton, QComboBox, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap

from webcam_connect import CameraManager
from ads_player import UnifiedContentPlayer
from ads_content import AdsContent
from realtime_detecter import RealTimeDetecter
from model_manager import ModelManager

# QtWebEngine 샌드박스 비활성화 (Docker 환경용)
os.environ['QTWEBENGINE_DISABLE_SANDBOX'] = '1'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--no-sandbox --disable-setuid-sandbox --ignore-certificate-errors'

ADS_PATH = "./sample_ads/imgNvideos"
YOUTUBE_CSV_PATH = "./sample_ads/sample_ad_video_urls/ads.csv"
ADS_CSV_PATH = "/workspace/interactive_ads_gui/src-old/sample_ads/ads.csv"
FACE_MODEL_PATH = "./models/cv/yolov8n-face-lindevs.mxq"
AGE_GENDER_MODEL_PATH = "./models/cv/genderage.mxq"
LLM_MODEL_PATH = "./models/llm/mblt-exaone"

# 메인 윈도우
class MainWindow(QMainWindow):
    """메인 GUI 윈도우 클래스"""
    
    def __init__(self, camera_id = 0, window_title="Ad System"):
        super().__init__()    
        self.camera_id = camera_id
        self.setWindowTitle(window_title)
        
        # 1. 실시간 탐지 모듈 (왼쪽 컬럼)
        self.detecter = RealTimeDetecter()

        self.ads_content = None   # 초기화 필요

        # UI 초기화
        self.initUI()
        
        # 광고 콘텐츠 모듈 생성 (ad_player 준비된 이후)
        self.ads_content = AdsContent(
            ads_csv_path=ADS_CSV_PATH,
            content_player=self.ad_player,  # 광고 표시 위젯
        )
        
        self._connect_signals()

        QTimer.singleShot(100, self.load_all_models_at_startup)
    
    def initUI(self):
        """UI 초기화"""
        self.setWindowTitle('System Modules - 나이/성별 타겟 광고')
        self.setGeometry(100, 100, 1200, 800)
        
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
        # RealTimeDetecter 시그널 연결 (그대로 유지)
        self.detecter.frame_updated.connect(self._on_frame_updated)
        self.detecter.status_updated.connect(self._on_status_updated)
        self.detecter.detection_result_updated.connect(self._on_detection_result_updated)
        self.detecter.age_gender_extracted.connect(self._on_age_gender_extracted)
        
        # AdsContent 시그널 연결
        self.ads_content.llm_text_ready.connect(self._on_llm_text_ready)
        self.ads_content.error_occurred.connect(self._on_ads_error)
    
	# === AI 모델 로딩 ===
    def load_all_models_at_startup(self):
        """앱 시작 시 모델 상태 확인 (모델은 이미 main.py에서 로드됨)"""
        print("\n" + "="*70)
        print("[확인] 모델 상태 확인")
        print("="*70)
        
        # 모든 버튼 비활성화
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.show_ad_button.setEnabled(False)
        
        # ModelManager가 이미 초기화되었는지 확인
        self._update_loading_status("🔄 모델 확인 중...", "AI 모델 상태를 확인하고 있습니다...")
        
        try:
            model_mgr = ModelManager()
            
            if not model_mgr._initialized:
                raise Exception("모델이 초기화되지 않았습니다")
            
            # RealTimeDetecter 초기화
            print("[MainWindow] RealTimeDetecter 초기화 중...")
            if not self.detecter.initialize_models():
                raise Exception("RealTimeDetecter 초기화 실패")
            print("[MainWindow] RealTimeDetecter 초기화 완료")
            
            print("[MainWindow] AdsContent LLM 초기화 중...")
            self.ads_content.initialize_llm()
            
            print("\n" + "="*70)
            print("[완료] 모델 확인 완료!")
            print("="*70 + "\n")
            
            # UI 업데이트
            self._update_loading_status(
                "✅ 모든 모델 준비 완료!", 
                "웹캠을 시작할 수 있습니다"
            )
            
            # LLM 텍스트 영역 업데이트
            self.llm_text.setText(
                "✅ AI 모델 준비 완료!\n\n"
                "광고 추천 이유가 여기에 표시됩니다.\n\n"
                "1. 웹캠을 시작하세요\n"
                "2. 얼굴이 감지되면\n"
                "3. '탐지된 타겟 광고 표시' 버튼을 눌러주세요"
            )
            
            # 탐지 정보 영역 업데이트
            self.detection_text.setText(
                "✅ 나이/성별 탐지 모델 준비 완료!\n\n"
                "웹캠을 시작하면 실시간으로\n"
                "얼굴의 나이와 성별을 탐지합니다."
            )
            
            # 웹캠 시작 버튼 활성화
            self.start_button.setEnabled(True)
            self.start_button.setText("웹캠 시작")
            self.show_ad_button.setEnabled(True)
            
            print(f"[{self.windowTitle()}] GUI 준비 완료")
            
        except Exception as e:
            print(f"[ERROR] 모델 확인 실패: {e}")
            
            self._update_loading_status(
                "❌ 모델 확인 실패", 
                f"오류: {str(e)}"
            )
            
            QMessageBox.critical(
                self, "오류",
                "모델이 초기화되지 않았습니다.\n\n"
                f"오류: {str(e)}\n\n"
                "프로그램을 종료합니다."
            )
            self.close()
    
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
    
    # === 왼쪽 레이아웃 생성 ===
    def _create_left_column(self):
        """왼쪽 컬럼 생성 (실시간 웹캠 화면 + 나이/성별 탐지 정보)"""
        left_layout = QVBoxLayout()
        
        camera_frame = self._create_camera_frame()
        
        detection_frame = self._create_detection_frame()
        
        left_layout.addWidget(camera_frame, 3)
        left_layout.addWidget(detection_frame, 2)
        
        return left_layout
    
    def _create_camera_frame(self):
        """웹캠 화면 프레임 생성"""
        frame = QFrame()
        frame.setFrameShape(QFrame.Box)
        frame.setFrameShadow(QFrame.Plain)
        frame.setLineWidth(2)
        
        layout = QVBoxLayout()
        frame.setLayout(layout)
        
        title_label = QLabel("실시간 웹캠 화면 (나이/성별 탐지 자동 활성화)")
        title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
        camera_select_layout = QHBoxLayout()
        camera_label = QLabel("카메라 선택:")
        self.camera_combo = QComboBox()
        
        available_cameras = CameraManager.get_available_cameras()
        if available_cameras:
            for cam_id in available_cameras:
                self.camera_combo.addItem(f"카메라 {cam_id}", cam_id)
        else:
            self.camera_combo.addItem("카메라를 찾을 수 없음", -1)
        
        camera_select_layout.addWidget(camera_label)
        camera_select_layout.addWidget(self.camera_combo)
        layout.addLayout(camera_select_layout)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setScaledContents(True)
        layout.addWidget(self.video_label, stretch=1)
        
        self.status_label = QLabel("모델을 로딩하고 있습니다...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: orange; padding: 5px; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        button_layout = QHBoxLayout()
        
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
        # ⚠️ 그대로 유지
        frame = QFrame()
        frame.setFrameShape(QFrame.Box)
        frame.setFrameShadow(QFrame.Plain)
        frame.setLineWidth(2)
        
        layout = QVBoxLayout()
        frame.setLayout(layout)
        
        title_label = QLabel("나이/성별 탐지 정보")
        title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
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
    
    # === 오른쪽 레이아웃 생성 ===
    def _create_right_column(self):
        """오른쪽 컬럼 생성 (광고 화면 + LLM 텍스트)"""
        right_layout = QVBoxLayout()
        
        ad_frame = self._create_ad_frame()
        
        llm_frame = self._create_llm_frame()
        
        right_layout.addWidget(ad_frame, 2)
        right_layout.addWidget(llm_frame, 1)
        
        return right_layout
    
    def _create_ad_frame(self):
        """광고 화면 프레임 생성 (위쪽 영역: 광고 콘텐츠 표시)"""
        frame = QFrame()
        frame.setFrameShape(QFrame.Box)
        frame.setFrameShadow(QFrame.Plain)
        frame.setLineWidth(2)
        
        layout = QVBoxLayout()
        frame.setLayout(layout)
        
        title_label = QLabel("타겟 광고 화면")
        title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
        # 광고 콘텐츠 표시용 UnifiedContentPlayer 위젯
        self.ad_player = UnifiedContentPlayer(self)
        layout.addWidget(self.ad_player, stretch=1)
        
        # 광고 표시 버튼 (탐지된 나이/성별 기준으로 AdsContent 호출)
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
        """LLM 텍스트 프레임 생성 (아래쪽 영역: LLM 설명)"""
        frame = QFrame()
        frame.setFrameShape(QFrame.Box)
        frame.setFrameShadow(QFrame.Plain)
        frame.setLineWidth(2)
        
        layout = QVBoxLayout()
        frame.setLayout(layout)
        
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
        self.llm_text.setText(
            "🔄 AI 모델을 로딩하고 있습니다...\n\n"
            "처음 실행 시 시간이 걸릴 수 있습니다.\n"
            "모델 로딩이 완료되면 웹캠을 시작할 수 있습니다."
        )
        layout.addWidget(self.llm_text)
        
        layout.setContentsMargins(10, 10, 10, 10)
        
        return frame
    
    # === 버튼 ===
    def _on_start_camera_clicked(self):
        """웹캠 시작 버튼 클릭"""
        camera_id = self.camera_combo.currentData()
        
        if camera_id == -1:
            QMessageBox.warning(self, "오류", "사용 가능한 카메라가 없습니다.")
            return
        
        success = self.detecter.start_camera(camera_id)
        
        if success:
            self.start_button.setEnabled(False)
            self.start_button.setText("웹캠 시작")
            self.stop_button.setEnabled(True)
            self.camera_combo.setEnabled(False)
            
            self.detection_text.clear()
            self.detection_text.setText("웹캠이 시작되었습니다.\n\n카메라 앞에 얼굴을 보여주세요...")
        else:
            QMessageBox.critical(
                self, "오류", 
                "나이/성별 탐지 모델이 로드되지 않았습니다.\n"
                "프로그램을 재시작해주세요."
            )
    
    def _on_stop_camera_clicked(self):
        """웹캠 중지 버튼 클릭"""
        self.detecter.stop_camera()
        
        self.video_label.clear()
        self.video_label.setText("웹캠이 중지되었습니다.")
        self.status_label.setText("웹캠이 연결되지 않았습니다.")
        self.status_label.setStyleSheet("color: gray; padding: 5px;")
        
        if self.detecter.is_initialized:
            self.detection_text.append("\n\n웹캠이 중지되었습니다.")
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_combo.setEnabled(True)
    
    def _on_show_ad_clicked(self):
        """광고 표시 버튼 클릭"""
        age, gender = self.detecter.get_current_detection()
        
        if age is None or gender is None:
            displayed_text = self.detection_text.toPlainText()
            if displayed_text:
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
        
        # AdsContent → AdsSelector + UnifiedContentPlayer 사용
        self.ads_content.show_targeted_ad(age, gender)
    
    # === 실시간 웹캠 업데이트 ===
    def _on_frame_updated(self, qt_image):
        """프레임 업데이트"""
        pixmap = QPixmap.fromImage(qt_image)
        
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
        self.detection_text.clear()
        
        display_text = result_text
        
        age, gender = self.detecter.get_current_detection()
        if age is not None and gender is not None:
            display_text += (
                f"\n✓ '탐지된 타겟 광고 표시' 버튼을 눌러\n"
                f"  이 정보에 맞는 광고를 볼 수 있습니다."
            )
        
        self.detection_text.setText(display_text)
    
    def _on_age_gender_extracted(self, age, gender):
        """나이/성별 추출 완료"""
        print(f"[MainWindow] 나이/성별 추출: {age}세, {gender}")
        # 필요시 추가 처리
    
    # === 광고 추천 관련 에러 처리 ===
    def _on_llm_text_ready(self, text):
        """LLM 텍스트 준비 완료"""
        self.llm_text.setText(text)
        print("[MainWindow] LLM 텍스트 표시 완료")
    
    def _on_ads_error(self, error_msg):
        """광고 관련 에러 발생"""
        QMessageBox.warning(self, "오류", error_msg)
    
    def closeEvent(self, event):
        """창 닫기 이벤트"""
        print("[MainWindow] 종료 시작...")
        
        # AdsContent 정리
        if self.ads_content is not None:
            self.ads_content.dispose()
        
        # RealTimeDetecter 정리
        self.detecter.dispose()
        
        print("[MainWindow] 종료 완료")
        event.accept()


def main():
    # Qt 애플리케이션 생성 (전체에서 하나만)
    app = QApplication(sys.argv)
    
    # 모델 한 번만 로드
    print("Initializing shared models...")
    model_mgr = ModelManager()
    model_mgr.initialize_models(FACE_MODEL_PATH, AGE_GENDER_MODEL_PATH, LLM_MODEL_PATH)
    
    # 카메라 1번용 윈도우 생성
    window1 = MainWindow(camera_id=0, window_title="Camera 1 - Ad System")
    window1.setGeometry(100, 100, 800, 600)  # 위치와 크기 설정
    window1.show()
    
    # 카메라 2번용 윈도우 생성
    window2 = MainWindow(camera_id=1, window_title="Camera 2 - Ad System")
    window2.setGeometry(920, 100, 800, 600)  # 오른쪽에 배치
    window2.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()