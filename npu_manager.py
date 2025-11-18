# npu_config_manager.py
import json
import maccel

class NPUConfigManager:
    """NPU 코어 할당 설정 관리자"""
    
    def __init__(self, config_path="npu_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[NPU Config] {self.config_path} 파일이 없습니다. 기본 설정 사용")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"[NPU Config] JSON 파싱 오류: {e}. 기본 설정 사용")
            return self._get_default_config()
    
    def _get_default_config(self):
        """기본 NPU 설정"""
        return {
            "npu_allocation": {
                "camera1": {
                "face_detection": {
                    "cluster": "Cluster1",
                    "cores": [0]
                },
                "age_gender": {
                    "cluster": "Cluster1",
                    "cores": [1]
                }
                },
                "camera2": {
                "face_detection": {
                    "cluster": "Cluster1",
                    "cores": [2]
                },
                "age_gender": {
                    "cluster": "Cluster1",
                    "cores": [3]
                }
                }
            },
            "llm": {
                "cluster": "Cluster0",
                "cores": [0]
            }
        }
    
    def get_model_config(self, camera_id, model_type):
        """
        특정 카메라의 특정 모델용 NPU 설정 반환
        
        Args:
            camera_id: "camera1" or "camera2"
            model_type: "face_detection" or "age_gender"
        
        Returns:
            maccel.ModelConfig 객체
        """
        try:
            npu_config = self.config["npu_allocation"][camera_id][model_type]
            return self._create_model_config(npu_config)
        except KeyError as e:
            print(f"[NPU Config] 설정 키 오류: {e}. 기본값 사용")
            # 기본값: Cluster1, Core0
            return self._create_model_config({
                "cluster": "Cluster1",
                "cores": [0]
            })
    
    def get_llm_config(self):
        """LLM용 NPU 설정 반환"""
        try:
            npu_config = self.config["llm"]
            return self._create_model_config(npu_config)
        except KeyError:
            print("[NPU Config] LLM 설정 없음. 기본값 사용")
            return self._create_model_config({
                "cluster": "Cluster1",
                "cores": [0, 1, 2, 3]
            })
    
    def _create_model_config(self, npu_config):
        """
        JSON 설정을 maccel.ModelConfig 객체로 변환
        
        Args:
            npu_config: {"cluster": "Cluster1", "cores": [0, 1]}
        
        Returns:
            maccel.ModelConfig 객체
        """
        mc = maccel.ModelConfig()
        mc.exclude_all_cores()
        
        # Cluster 파싱
        cluster_name = npu_config.get("cluster", "Cluster1")
        cluster = getattr(maccel.Cluster, cluster_name, maccel.Cluster.Cluster1)
        
        # Cores 파싱 및 포함
        cores = npu_config.get("cores", [0])
        for core_num in cores:
            core = self._get_core_enum(core_num)
            if core:
                mc.include(cluster, core)
                print(f"[NPU Config] {cluster_name}.Core{core_num} 포함")
        
        return mc
    
    def _get_core_enum(self, core_num):
        """코어 번호를 maccel.Core enum으로 변환"""
        core_map = {
            0: maccel.Core.Core0,
            1: maccel.Core.Core1,
            2: maccel.Core.Core2,
            3: maccel.Core.Core3,
        }
        return core_map.get(core_num)
    
    def print_allocation_summary(self):
        """현재 NPU 할당 요약 출력"""
        print("\n" + "="*60)
        print("NPU 코어 할당 현황")
        print("="*60)
        
        for camera_id in ["camera1", "camera2"]:
            if camera_id in self.config.get("npu_allocation", {}):
                print(f"\n[{camera_id.upper()}]")
                cam_config = self.config["npu_allocation"][camera_id]
                
                for model_type, npu_cfg in cam_config.items():
                    cores_str = ", ".join(map(str, npu_cfg["cores"]))
                    print(f"  {model_type:20s}: {npu_cfg['cluster']} - Cores [{cores_str}]")
        
        if "llm" in self.config:
            llm_cfg = self.config["llm"]
            cores_str = ", ".join(map(str, llm_cfg["cores"]))
            print(f"\n[LLM]")
            print(f"  {'llm_model':20s}: {llm_cfg['cluster']} - Cores [{cores_str}]")
        
        print("="*60 + "\n")