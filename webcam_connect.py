#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# OpenCV와 PyQt5의 Qt 충돌 방지
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


class CameraThread(QThread):
    """
    로컬 웹캠 스트림을 처리하는 스레드
    """
    # 프레임 업데이트 시그널
    frame_update = pyqtSignal(QImage)
    # 연결 상태 시그널
    connection_status = pyqtSignal(bool, str)
    # 탐지 결과 시그널 (추가)
    detection_result = pyqtSignal(str)
    
    def __init__(self, camera_id=0, detection_manager=None):
        """
        Args:
            camera_id (int): 로컬 웹캠 ID (기본값: 0, 여러 카메라가 있으면 1, 2, ...)
            detection_manager: AgeGenderDetectionManager 인스턴스 (선택사항)
        """
        super().__init__()
        self.camera_id = camera_id
        self.detection_manager = detection_manager
        self.is_running = False
        self.capture = None
    
    def run(self):
        """스레드 실행 메서드"""
        self.is_running = True
        
        # 로컬 웹캠 연결
        self.capture = cv2.VideoCapture(self.camera_id)
        
        # 연결 확인
        if not self.capture.isOpened():
            self.connection_status.emit(False, f"웹캠 연결 실패 (카메라 ID: {self.camera_id})")
            return
        
        # 카메라 설정 최적화 (선택사항)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        self.connection_status.emit(True, f"웹캠 연결 성공 (카메라 ID: {self.camera_id})")
        
        # 프레임 읽기 루프
        while self.is_running:
            ret, frame = self.capture.read()
            
            if ret:
                # 탐지 기능이 활성화된 경우
                if self.detection_manager is not None:
                    # 나이/성별 탐지 수행
                    result = self.detection_manager.process_frame(frame)
                    
                    # 탐지 결과가 그려진 프레임 사용
                    processed_frame = result['frame']
                    
                    # 탐지 결과 텍스트 전송
                    self.detection_result.emit(result['summary'])
                else:
                    # 탐지 기능 없이 원본 프레임 사용
                    processed_frame = frame
                
                # BGR을 RGB로 변환
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # QImage로 변환
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # 시그널 발생
                self.frame_update.emit(qt_image)
            else:
                # 프레임 읽기 실패 시 재연결 시도
                self.connection_status.emit(False, "프레임 읽기 실패. 재연결 시도 중...")
                self.capture.release()
                
                self.capture = cv2.VideoCapture(self.camera_id)
                
                if not self.capture.isOpened():
                    self.connection_status.emit(False, "재연결 실패")
                    break
        
        # 정리
        if self.capture:
            self.capture.release()
    
    def stop(self):
        """스레드 중지"""
        self.is_running = False
        self.wait()


class CameraManager:
    """
    로컬 웹캠을 관리하는 클래스
    """
    def __init__(self):
        self.camera_thread = None
    
    def start_camera(self, camera_id=0, detection_manager=None):
        """
        웹캠 스트림 시작
        
        Args:
            camera_id (int): 로컬 웹캠 ID (0, 1, 2, ...)
            detection_manager: AgeGenderDetectionManager 인스턴스 (선택사항)
        
        Returns:
            CameraThread: 카메라 스레드 객체
        """
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()
        
        self.camera_thread = CameraThread(camera_id=camera_id, detection_manager=detection_manager)
        self.camera_thread.start()
        
        return self.camera_thread
    
    def stop_camera(self):
        """웹캠 스트림 중지"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
    
    def is_running(self):
        """카메라가 실행 중인지 확인"""
        return self.camera_thread and self.camera_thread.isRunning()
    
    @staticmethod
    def get_available_cameras(max_test=5):
        """
        사용 가능한 카메라 목록 확인
        
        Args:
            max_test (int): 테스트할 최대 카메라 ID 수
        
        Returns:
            list: 사용 가능한 카메라 ID 리스트
        """
        available_cameras = []
        
        for camera_id in range(max_test):
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                available_cameras.append(camera_id)
                cap.release()
        
        return available_cameras