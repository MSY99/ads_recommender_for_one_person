#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM 추론 모듈
광고 추천 이유를 생성하기 위한 LLM 인터페이스
"""

import torch
from mblt_model_zoo.transformers import AutoTokenizer, AutoModelForCausalLM


class LLMInferenceManager:
    """LLM 추론 관리 클래스"""
    
    def __init__(self, model_path="/workspace/interactive_ads_gui/src-old/mblt-exaone", device_no=0):
        """
        Args:
            model_path (str): 모델 경로
            device_no (int): NPU 디바이스 번호
        """
        self.model_path = model_path
        self.device_no = device_no
        self.tokenizer = None
        self.model = None
        self.is_initialized = False
        
        print(f"\n{'='*60}")
        print(f"[LLM Manager] 초기화 시작")
        print(f"[LLM Manager] 모델 경로: {model_path}")
        print(f"[LLM Manager] NPU 디바이스: {device_no}")
        print(f"{'='*60}")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """모델 초기화"""
        try:
            print("[LLM Manager] 1/3 토크나이저 로딩 중...")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("[LLM Manager]   ↳ pad_token을 eos_token으로 설정")
            
            print("[LLM Manager]   ✓ 토크나이저 로딩 완료")
            
            print("[LLM Manager] 2/3 NPU 모델 로딩 중...")
            print("[LLM Manager]   (이 단계에서 시간이 걸립니다...)")
            
            # NPU 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            
            print("[LLM Manager]   ✓ NPU 모델 로딩 완료")
            
            print("[LLM Manager] 3/3 초기화 완료 확인 중...")
            
            self.is_initialized = True
            
            print(f"\n{'='*60}")
            print("[LLM Manager] ✅ 전체 초기화 완료!")
            print(f"{'='*60}\n")
            
        except FileNotFoundError as e:
            print(f"[LLM Manager] ❌ 파일을 찾을 수 없음: {e}")
            print(f"[LLM Manager]    모델 경로를 확인하세요: {self.model_path}")
            self.is_initialized = False
            
        except Exception as e:
            print(f"[LLM Manager] ❌ 모델 로딩 실패: {e}")
            print(f"[LLM Manager]    예외 타입: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            self.is_initialized = False
    
    def generate_ad_explanation(self, age_group, gender, actual_age):
        """
        광고 추천 이유 생성
        
        Args:
            age_group (str): 연령대 (20, 30, 40, 50)
            gender (str): 성별 (남성, 여성)
            actual_age (int): 실제 나이
        
        Returns:
            str: 생성된 광고 추천 이유
        """
        if not self.is_initialized:
            error_msg = "LLM 모델이 초기화되지 않았습니다."
            print(f"[LLM Manager] ❌ {error_msg}")
            return error_msg
        
        # 프롬프트 생성
        prompt = self._create_prompt(age_group, gender, actual_age)
        
        try:
            print(f"\n[LLM Manager] {'='*50}")
            print(f"[LLM Manager] 추론 시작")
            print(f"[LLM Manager] 타겟: {age_group}대 {gender} ({actual_age}세)")
            print(f"[LLM Manager] {'='*50}")
            
            # 추론 실행
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            print(f"[LLM Manager] 입력 토큰 수: {inputs['input_ids'].shape[1]}")
            print(f"[LLM Manager] 생성 중... (max_new_tokens=70)")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=70,  # 간단하게 설정
                    temperature=0.7,
                    do_sample=True,
                    use_cache=True,
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 프롬프트 부분 제거
            if prompt in result:
                result = result.replace(prompt, "").strip()
            
            print(f"[LLM Manager] ✓ 추론 완료 (출력 길이: {len(result)} 글자)")
            print(f"[LLM Manager] {'='*50}\n")
            
            return result
            
        except Exception as e:
            error_msg = f"LLM 추론 중 오류 발생: {str(e)}"
            print(f"[LLM Manager] ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg
    
    def _create_prompt(self, age_group, gender, actual_age):
        """
        광고 추천 프롬프트 생성
        
        Args:
            age_group (str): 연령대
            gender (str): 성별
            actual_age (int): 실제 나이
        
        Returns:
            str: 프롬프트
        """
        # 간단한 프롬프트 (사용자가 수정 예정)
        prompt = f"""다음 고객 정보를 바탕으로 광고 추천 이유를 설명해주세요.

고객 정보:
- 연령대: {age_group}대
- 성별: {gender}
- 실제 나이: {actual_age}세

광고 추천 이유:"""
        
        return prompt
    
    def dispose(self):
        """리소스 정리"""
        if self.model is not None:
            try:
                print("\n[LLM Manager] 리소스 정리 중...")
                self.model.dispose()
                print("[LLM Manager] ✓ NPU 리소스 정리 완료")
            except Exception as e:
                print(f"[LLM Manager] ⚠ 리소스 정리 중 오류: {e}")
            finally:
                self.model = None
                self.tokenizer = None
                self.is_initialized = False


# 싱글톤 인스턴스 (선택적)
_llm_instance = None


def get_llm_instance(model_path="/workspace/interactive_ads_gui/src-old/mblt-exaone", device_no=0):
    """
    LLM 인스턴스 가져오기 (싱글톤 패턴)
    
    Args:
        model_path (str): 모델 경로
        device_no (int): NPU 디바이스 번호
    
    Returns:
        LLMInferenceManager: LLM 인스턴스
    """
    global _llm_instance
    
    if _llm_instance is None:
        _llm_instance = LLMInferenceManager(model_path, device_no)
    
    return _llm_instance


if __name__ == "__main__":
    # 테스트 코드
    print("=== LLM 모듈 테스트 ===\n")
    
    llm = LLMInferenceManager()
    
    if llm.is_initialized:
        # 광고 추천 테스트
        result = llm.generate_ad_explanation("30", "남성", 32)
        print(f"\n[생성된 광고 추천 이유]\n{result}\n")
    
    # 리소스 정리
    llm.dispose()