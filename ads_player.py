import os
import re
import cv2

from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

# from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
# from PyQt5.QtMultimediaWidgets import QVideoWidget

from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings


class ImagePlayer(QWidget):
    """단순 이미지 플레이어 (레이아웃에 그대로 올려서 사용)"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def load_and_play(self, file_path: str) -> bool:
        """이미지 로드 및 표시 (성공 여부 반환)"""
        if not os.path.exists(file_path):
            print(f"[ImagePlayer] ❌ 파일을 찾을 수 없습니다: {file_path}")
            self.label.clear()
            return False

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            print(f"[ImagePlayer] ❌ 이미지 로드 실패: {file_path}")
            self.label.clear()
            return False

        self.label.setPixmap(pixmap)
        print(f"[ImagePlayer] ✓ 이미지 표시: {file_path}")
        return True

    def stop(self):
        """표시 중지 (이미지 제거)"""
        print("[ImagePlayer] 이미지 표시 중지")
        self.label.clear()


class VideoPlayer(QWidget):
    """
    QtMultimedia(QMediaPlayer) 대신 OpenCV + QLabel로 동영상 재생.
    - 파일을 cv2.VideoCapture로 읽어서
    - QTimer로 주기적으로 프레임을 갱신하고
    - QLabel에 QPixmap으로 뿌려준다.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black;")
        self.label.setMinimumSize(400, 300)
        self.label.setScaledContents(True)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)

        self._current_path = None

    def load_and_play(self, file_path: str) -> bool:
        """동영상 로드 및 반복 재생 시작."""
        if not os.path.exists(file_path):
            print(f"[VideoPlayer] ❌ 파일을 찾을 수 없습니다: {file_path}")
            self.stop()
            return False

        # 기존 캡처 정리
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        full_path = os.path.abspath(file_path)
        print(f"[VideoPlayer] ▶ OpenCV 동영상 재생 시작: {full_path}")
        self._current_path = full_path

        self.cap = cv2.VideoCapture(full_path)
        if not self.cap.isOpened():
            print(f"[VideoPlayer] ❌ VideoCapture를 열 수 없습니다: {full_path}")
            self.stop()
            return False

        # FPS 가져와서 타이머 간격 설정 (기본 30fps)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0
        interval_ms = int(1000 / fps)
        print(f"[VideoPlayer] FPS={fps:.2f}, interval={interval_ms}ms")

        self.timer.start(interval_ms)
        return True

    def _next_frame(self):
        """다음 프레임 읽어서 QLabel에 표시 (루프 재생)."""
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            # 끝까지 갔으면 처음부터 다시 (루프 재생)
            # print("[VideoPlayer] 비디오 끝 - 처음으로 루프")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # 현재 VideoPlayer 위젯 크기에 맞게 스케일
        scaled = pixmap.scaled(
            self.label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        """윈도우 리사이즈시 현재 프레임도 같이 리사이즈."""
        super().resizeEvent(event)
        if self.label.pixmap() is not None:
            scaled = self.label.pixmap().scaled(
                self.label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.label.setPixmap(scaled)

    def stop(self):
        """재생 중지 및 리소스 정리."""
        print("[VideoPlayer] 동영상 재생 중지 (OpenCV)")
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._current_path = None
        self.label.clear()


class YouTubePlayer(QWidget):
    """
    단순 유튜브 플레이어.
    - 유튜브 URL 또는 비디오 ID를 받아서 embed URL로 재생
    - 컨트롤 버튼 없음, 바로 자동재생
    """
    _global_webengine_initialized = False
    
    def __init__(self, parent=None):
        # QWebEngineSettings 글로벌 설정
        if not YouTubePlayer._global_webengine_initialized:
            QWebEngineSettings.globalSettings().setAttribute(
                QWebEngineSettings.PluginsEnabled, True
            )
            QWebEngineSettings.globalSettings().setAttribute(
                QWebEngineSettings.JavascriptEnabled, True
            )
            QWebEngineSettings.globalSettings().setAttribute(
                QWebEngineSettings.AutoLoadImages, True
            )
            QWebEngineSettings.globalSettings().setAttribute(
                QWebEngineSettings.LocalStorageEnabled, True
            )
            YouTubePlayer._global_webengine_initialized = True

        super().__init__(parent)

        self.webview = QWebEngineView(self)

        layout = QVBoxLayout()
        layout.addWidget(self.webview)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def load_and_play(self, url_or_id: str) -> bool:
        """
        유튜브 URL/ID 로드 및 재생.
        - 전체 URL or 11자리 video_id 모두 허용
        """
        video_id = self._extract_video_id(url_or_id)
        if not video_id:
            print(f"[YouTubePlayer] ❌ 유효하지 않은 유튜브 URL 또는 ID: {url_or_id}")
            self.webview.setHtml("<html><body style='background-color:#000;'></body></html>")
            return False

        embed_url = self._build_embed_url(video_id)
        print(f"[YouTubePlayer] ▶ 유튜브 재생 시작: {embed_url}")

        self.webview.setUrl(QUrl(embed_url))
        return True

    def _extract_video_id(self, url_or_id: str) -> str:
        """유튜브 URL 또는 ID에서 video_id 추출 (실패 시 None)"""

        # 이미 ID만 있는 경우 (11자리 영숫자)
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
            return url_or_id

        # 다양한 유튜브 URL 패턴들
        patterns = [
            r'(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})',
            r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com/v/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)

        return None

    def _build_embed_url(self, video_id: str) -> str:
        """자동재생/루프 옵션이 붙은 embed URL 생성"""
        return (
            f"https://www.youtube.com/embed/{video_id}"
            f"?autoplay=1&loop=1&playlist={video_id}&controls=0&modestbranding=1&rel=0"
        )

    def stop(self):
        """재생 중지 (검은 화면으로 전환)"""
        print("[YouTubePlayer] 유튜브 재생 중지")
        # about:blank 도 좋고, 검은 배경 html도 가능
        self.webview.setHtml("<html><body style='background-color:#000;'></body></html>")


class UnifiedContentPlayer(QWidget):
    """
    3개의 Player 중 적절한 것을 메인 윈도우 광고 레이아웃에 적용

    content_type:
        - "img"     : 로컬 이미지 파일
        - "video"   : 로컬 동영상 파일
        - "youtube" : 유튜브 URL 또는 video_id
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # 내부 플레이어들
        self.image_player = ImagePlayer(self)
        self.video_player = VideoPlayer(self)
        self.youtube_player = YouTubePlayer(self)

        self._current_widget: QWidget = None

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

    def _set_active_widget(self, widget: QWidget):
        """현재 보여지는 위젯 교체"""
        if self._current_widget is widget:
            return

        if self._current_widget is not None:
            self._layout.removeWidget(self._current_widget)
            self._current_widget.hide()

        self._current_widget = widget
        self._layout.addWidget(widget)
        widget.show()

    def _stop_all(self):
        """모든 플레이어 정지/초기화"""
        self.image_player.stop()
        self.video_player.stop()
        self.youtube_player.stop()

    def show_content(self, content_type: str, source: str):
        """
        콘텐츠 타입에 따라 적절한 플레이어 선택 후 바로 표시/재생.
        메인윈도우에서는 이 함수만 호출해주면 됨.
        """
        self._stop_all()

        if content_type == "img":
            self._set_active_widget(self.image_player)
            self.image_player.load_and_play(source)

        elif content_type == "video":
            self._set_active_widget(self.video_player)
            self.video_player.load_and_play(source)

        elif content_type == "youtube":
            #self._set_active_widget(self.youtube_player)
            #self.youtube_player.load_and_play(source)
            print("[UnifiedContentPlayer] 아직 유튜브 실행 기능이 활성화되지 않았습니다.")

        else:
            print(f"[UnifiedContentPlayer] ❌ 알 수 없는 content_type: {content_type}")
            # 필요하면 여기서 기본 빈 화면 처리 가능
