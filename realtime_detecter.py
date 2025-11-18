import re
from PyQt5.QtCore import pyqtSignal, QObject

from webcam_connect import CameraManager
from face_analysis import AgeGenderDetectionManager

# 실시간 나이/성별 추론
class RealTimeDetecter(QObject):
	"""실시간 얼굴 탐지 및 나이/성별 추론 클래스"""
	
	# 시그널 정의
	frame_updated = pyqtSignal(object)  # QImage 프레임
	status_updated = pyqtSignal(bool, str)  # 성공 여부, 메시지
	detection_result_updated = pyqtSignal(str)  # 탐지 결과 텍스트
	age_gender_extracted = pyqtSignal(int, str)  # 나이, 성별
	
	def __init__(self):
		"""
		Args:
			face_model_path: 얼굴 탐지 모델 경로
			age_gender_model_path: 나이/성별 추론 모델 경로
		"""
		super().__init__()
		
		self.camera_manager = CameraManager()
		
		self.detection_manager = None
		
		self.current_age = None
		self.current_gender = None
		
		# 초기화 상태
		self.is_initialized = False
	
	def initialize_models(self):
		"""CV 모델 초기화 (얼굴 탐지 + 나이/성별)"""
		print("\n[RealTimeDetecter] CV 모델 로딩 시작...")
		
		try:
			self.detection_manager = AgeGenderDetectionManager()
			
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
				print(f"[RealTimeDetecter] ✓ 탐지 정보: 나이={age_val}세, 성별={gender} - 1")
				print(f"  (얼굴 {face_num}, 신뢰도: {confidence})")
				return age_val, gender
			
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
