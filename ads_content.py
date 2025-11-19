import os
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QPixmap

from ads_player import UnifiedContentPlayer
from ads_recommender import AdSelector
from llm_infer import LLMInferenceManager
from llm_worker import LLMInferenceWorkerThread

from model_manager import ModelManager

# 광고 콘텐츠 생성
class AdsContent(QObject):
    """타겟 광고 표시 및 LLM 추론 클래스"""
    
    ad_image_ready = pyqtSignal(QPixmap)   # 이미지 준비
    ad_video_ready = pyqtSignal(str)       # 비디오 파일 경로 준비      

    llm_text_ready = pyqtSignal(str)       # LLM 추론 텍스트 준비
    error_occurred = pyqtSignal(str)       # 에러 발생 여부 확인
    
    def __init__(
        self, 
        ads_csv_path: str,
        content_player: UnifiedContentPlayer,
        ads_selector: AdSelector = None,
        parent=None
    ):
        """
        Args:
            ad_base_path: 광고 콘텐츠(이미지/영상/유튜브 csv 등) 기본 경로
            youtube_csv_path: 유튜브 CSV 경로
            content_player: 실제 광고를 재생할 UnifiedContentPlayer 인스턴스
            ads_selector: 광고 추천 로직을 담당하는 AdsSelector (없으면 ad_base_path로 내부 생성)
        """
        super().__init__(parent)
        
        self.ads_csv_path = ads_csv_path

        # 광고 선택/재생 관련
        self.content_player = content_player          # UnifiedContentPlayer
        self.ads_selector = ads_selector or AdSelector(ads_csv_path)
        
        # LLM 관련
        self.llm_manager = None
        self.llm_worker = None
        self.is_llm_initialized = False
    
    def initialize_llm(self):
        """LLM 모델 확인 (ModelManager에서 이미 로드됨)"""
        print("\n[AdsContent] LLM 모델 확인 중...")
        
        try:
            model_mgr = ModelManager()
            self.llm_manager, self.llm_lock = model_mgr.get_llm_manager()
            
            if self.llm_manager is None:
                print("[AdsContent] ❌ LLM 모델 없음")
                self.is_llm_initialized = False
                return False
            
            print("[AdsContent] ✅ LLM 준비 완료")
            self.is_llm_initialized = True
            return True
            
        except Exception as e:
            print(f"[AdsContent] ❌ LLM 확인 예외: {e}")
            self.is_llm_initialized = False
            return False

    def show_targeted_ad(self, age, gender):
        """
        나이/성별에 맞는 타겟 광고 표시

        Args:
            age: 탐지된 나이 (int)
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

        if self.content_player is None:
            error_msg = "광고를 재생할 UnifiedContentPlayer가 설정되어 있지 않습니다."
            print(f"[AdsContent] ❌ {error_msg}")
            self.error_occurred.emit(error_msg)
            return False
        
        # === AdSelector로 광고 선택 ===
        try:
            selection = self.ads_selector.select_ad(age_group, gender_en)
            
        except Exception as e:
            # print("전달 인자: ", {age_group}, type(age_group), {gender_en}, type(gender_en))
            error_msg = f"광고 추천 중 예외 발생: {e}"
            print(f"[AdsContent] ❌ {error_msg}")
            self.error_occurred.emit(error_msg)
            return False

        if not selection:
            # 선택 실패 → 기존 에러 메시지 생성 로직 재사용
            error_msg = self._build_image_not_found_error(age_group, gender_en)
            print(f"[AdsContent] ❌ 광고 콘텐츠 선택 실패")
            self.error_occurred.emit(error_msg)
            return False

        content_type, source, description = selection

        if not content_type or not source:
            error_msg = "AdsSelector에서 유효한 광고 정보를 받지 못했습니다."
            print(f"[AdsContent] ❌ {error_msg} selection={selection}")
            self.error_occurred.emit(error_msg)
            return False

        print(f"[AdsContent] ✓ 선택된 광고: type={content_type}, source={source}")
        print(f"[AdsContent] ✓ 광고 설명: {description}")

        # === UnifiedPlayer로 콘텐츠 타입에 따라 재생 ===
        self.content_player.show_content(content_type, source)

        if content_type == "img":
            pixmap = QPixmap(source)
            if not pixmap.isNull():
                self.ad_image_ready.emit(pixmap)
        elif content_type == "video":
            self.ad_video_ready.emit(source)
        # youtube는 별도 시그널 없이 Player가 직접 재생

        # === LLM 백그라운드 실행 ===
        if not self.is_llm_initialized or self.llm_manager is None:
            print("\n[AdsContent] LLM 모델이 없음 - 기본 설명 표시")
            self._show_default_explanation(age_group, gender, age)
            return True
        
        # LLM 비동기 추론 시작
        self._start_llm_inference_async(age_group, gender, age, description)
        
        return True

    # 에러 메시지
    def _build_image_not_found_error(self, age_group, gender_en):
        """이미지/비디오를 찾지 못했을 때 에러 메시지 생성"""
        try:
            available_files = os.listdir(self.ad_base_path)
            # 이미지와 비디오 파일 모두 검색
            ad_files = [
                f for f in available_files
                if any(f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.mp4'])
            ]
        except Exception:
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

    # === LLM 비동기 실행 ===
    def _start_llm_inference_async(self, age_group, gender, age, ad_description=""):
        """
        LLM 추론 비동기 시작
        
        Args:
            age_group: 연령대
            gender: 성별
            age: 실제 나이
            ad_description: 광고 정보
        """
        # 로딩 메시지 먼저 표시
        loading_msg = "🔄 AI가 광고를 분석하는 중입니다...\n잠시만 기다려주세요."
        self.llm_text_ready.emit(loading_msg)
        
        print("\n[AdsContent] LLM 추론 워커 스레드 시작")
        
        # LLM 추론 워커 스레드 생성 및 시작
        self.llm_worker = LLMInferenceWorkerThread(
            age_group,
            gender,
            age,
            ad_description
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

※ LLM 모델이 초기화되지 않았습니다."""
    
    # 나이 -> 연령대 매핑 함수
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
        
        # LLM 추론 중단
        self.stop_llm_inference()
        
        # LLM 리소스 정리
        if self.llm_manager is not None:
            self.llm_manager.dispose()
            self.llm_manager = None
        
        self.is_llm_initialized = False

