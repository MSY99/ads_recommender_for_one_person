#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM 추론 워커 스레드
백그라운드에서 LLM 추론을 실행하여 GUI 블로킹 방지
"""

from PyQt5.QtCore import QThread, pyqtSignal
from model_manager import ModelManager

class LLMInferenceWorkerThread(QThread):
    """LLM 추론만 수행하는 워커 스레드 (광고 표시 시 사용)"""
    
    # 시그널 정의
    result_ready = pyqtSignal(str)  # 추론 완료 시 결과 전달
    error_occurred = pyqtSignal(str)  # 에러 발생 시
    
    def __init__(self, age_group, gender, actual_age):
        """
        Args:
            age_group (str): 연령대 (20, 30, 40, 50)
            gender (str): 성별 (남성, 여성)
            actual_age (int): 실제 나이
        """
        super().__init__()
        model_mgr = ModelManager()
        self.llm_manager, self.llm_lock = model_mgr.get_llm_manager()
        self.age_group = age_group
        self.gender = gender
        self.actual_age = actual_age
    
    def run(self):
        """스레드 실행 (백그라운드에서 LLM 추론)"""
        try:
            if self.llm_manager is None:
                error_msg = "LLM 매니저가 초기화되지 않았습니다."
                print(f"[LLM Inference Worker] ✗ {error_msg}")
                self.error_occurred.emit(error_msg)
                return
            
            print(f"\n[LLM Inference Worker] 추론 시작")
            print(f"[LLM Inference Worker] 타겟: {self.age_group}대 {self.gender} ({self.actual_age}세)")
            
            # LLM 추론 실행 (락 사용)
            with self.llm_lock:
                result = self.llm_manager.generate_ad_explanation(
                    self.age_group, 
                    self.gender, 
                    self.actual_age
                )
            
            print(f"[LLM Inference Worker] ✓ 추론 완료 - 결과 길이: {len(result)} 글자")
            
            # 결과를 메인 스레드로 전달
            self.result_ready.emit(result)
            
        except Exception as e:
            print(f"[LLM Inference Worker] ✗ 예외 발생: {e}")
            import traceback
            traceback.print_exc()
            
            # 에러 메시지를 메인 스레드로 전달
            error_msg = f"LLM 추론 중 오류 발생:\n{str(e)}"
            self.error_occurred.emit(error_msg)