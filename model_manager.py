import threading
from typing import Optional

class ModelManager:
    """싱글톤 패턴으로 구현된 모델 매니저"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # CV 모델들
        self.face_detector = None
        self.age_gender_predictor = None
        
        # LLM 모델
        self.llm_inference_manager = None
        
        # 모델 접근용 락
        self.face_detector_lock = threading.Lock()
        self.age_gender_lock = threading.Lock()
        self.llm_lock = threading.Lock()
        
        self._initialized = True
    
    def initialize_models(self, face_model_path, age_gender_model_path, llm_model_path):
        """모든 모델을 초기화 (앱 시작 시 한 번만 호출)"""
        from face_detector import FaceDetector
        from agegender_predict import AgeGenderPredictor
        from llm_infer import LLMInferenceManager
        
        print("Loading Face Detector...")
        self.face_detector = FaceDetector(
            mxq_path=face_model_path,
            conf_threshold=0.5,
            iou_threshold=0.45,
            img_size=640
        )
        
        print("Loading Age-Gender Predictor...")
        self.age_gender_predictor = AgeGenderPredictor(
            mxq_path=age_gender_model_path,
            img_size=96
        )
        
        print("Loading LLM Model...")
        self.llm_inference_manager = LLMInferenceManager(
            model_path=llm_model_path
        )
        
        print("All models loaded successfully!")
    
    def get_face_detector(self):
        """스레드 안전한 FaceDetector 접근"""
        return self.face_detector, self.face_detector_lock
    
    def get_age_gender_predictor(self):
        """스레드 안전한 AgeGenderPredictor 접근"""
        return self.age_gender_predictor, self.age_gender_lock
    
    def get_llm_manager(self):
        """스레드 안전한 LLM Manager 접근"""
        return self.llm_inference_manager, self.llm_lock