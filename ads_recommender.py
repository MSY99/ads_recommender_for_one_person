import os
import csv
import random
from typing import List, Dict, Tuple, Optional

class AdSelector:
    """
    나이/성별 값을 받아, 타겟 광고 콘텐츠를 1개 선별하는 객체
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, media_dir: str, youtube_csv_path: str):
        """
        :param media_dir: 이미지/동영상 파일이 저장된 디렉토리 경로
        :param youtube_csv_path: YouTube 광고 정보가 들어있는 CSV 파일 경로
        """
        self.media_dir = media_dir
        self.youtube_csv_path = youtube_csv_path

    # ==== UTILS ====
    def _normalize_key(self, age: int, gender: str) -> Tuple[str, str]:
        """age, gender를 키로 쓸 수 있게 정규화."""
        gender = gender.lower().strip()
        age_str = str(age).strip()
        return age_str, gender

    def _load_media_ads(self, age: int, gender: str) -> List[Dict[str, str]]:
        """로컬 이미지/동영상 파일에서 해당 타겟의 광고를 가져온다."""
        age_str, gender_norm = self._normalize_key(age, gender)
        key_prefix = f"{age_str}_{gender_norm}"
        ads: List[Dict[str, str]] = []

        if not os.path.isdir(self.media_dir):
            return ads

        for fname in os.listdir(self.media_dir):
            if not fname.startswith(key_prefix):
                continue

            ext = os.path.splitext(fname)[1].lower()
            full_path = os.path.join(self.media_dir, fname)

            if ext in self.IMAGE_EXTS:
                ads.append({"type": "img", "value": full_path})
            elif ext in self.VIDEO_EXTS:
                ads.append({"type": "video", "value": full_path})

        return ads

    def _load_youtube_ads(self, age: int, gender: str) -> List[Dict[str, str]]:
        """CSV에서 해당 타겟의 YouTube 광고를 가져온다."""
        age_str, gender_norm = self._normalize_key(age, gender)
        ads: List[Dict[str, str]] = []

        if not os.path.isfile(self.youtube_csv_path):
            return ads

        with open(self.youtube_csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_gender = row["target_gender"].strip().lower()
                row_age = str(row["target_age"]).strip()
                if row_gender == gender_norm and row_age == age_str:
                    ads.append({"type": "youtube", "value": row["ads_url"].strip()})

        return ads

    # ==== External Methods ====
    def get_all_ads(self, age: int, gender: str) -> List[Dict[str, str]]:
        """
        나이/성별에 맞는 모든 광고 리스트를 반환.

        반환 예:
        [
            {"type": "img", "value": "/path/10_female_1.jpg"},
            {"type": "video", "value": "/path/10_female_ad.mp4"},
            {"type": "youtube", "value": "https://www.youtube.com/..."},
        ]
        """
        ads = []
        ads.extend(self._load_media_ads(age, gender))
        ads.extend(self._load_youtube_ads(age, gender))
        return ads

    def select_ad(self, age: str, gender: str) -> Optional[Tuple[str, str]]:
        """
        나이/성별에 맞는 광고 중 하나를 랜덤 선택해서 반환.

        반환값:
            (content_type, path_or_url)
            content_type ∈ {"img", "video", "youtube"}

        광고가 하나도 없으면 None 반환.
        """
        candidates = self.get_all_ads(int(age), gender)
        if not candidates:
            return None

        chosen = random.choice(candidates)
        return chosen["type"], chosen["value"]
