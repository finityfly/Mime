from __future__ import annotations

import time
from typing import Callable, Optional

import cv2
import numpy as np

from .media_bridge import VideoFrameSource


class VideoClient(VideoFrameSource):
    """Unified visual source controller for the Zoom bridge.

    Modes supported:
    - 'blank' : black canvas with optional status text
    - 'webcam': forward a local webcam (cv2.VideoCapture)
    - 'abs'   : render using a provided renderer callable (takes no args,
                returns BGR frame)
    - 'text'  : render a simple text card

    The class implements `read()` so it can be passed directly to
    `ZoomMediaBridge.set_video_source()`.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30, log: Callable[[str], None] = print):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.log = log

        self.mode: str = "blank"
        self._webcam_index: int = 0
        self._cap: Optional[cv2.VideoCapture] = None

        # For ABS or custom rendering the user provides a callable that
        # returns a BGR frame when invoked.
        self._renderer: Optional[Callable[[], np.ndarray]] = None

        self._text: str = ""
        self._last_frame = self._make_blank()

    def set_mode(self, mode: str) -> None:
        self.mode = str(mode)

    def set_webcam_index(self, idx: int) -> None:
        self._webcam_index = int(idx)
        # (re)open capture on next read
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def set_renderer(self, renderer: Callable[[], np.ndarray]) -> None:
        self._renderer = renderer

    def set_text(self, text: str) -> None:
        self._text = str(text)

    def _make_blank(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame

    def _fit_frame_letterbox(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return self._last_frame

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        src_h, src_w = frame.shape[:2]
        dst_w, dst_h = self.width, self.height
        if src_w <= 0 or src_h <= 0:
            return self._last_frame

        scale = min(dst_w / src_w, dst_h / src_h)
        out_w = max(1, int(src_w * scale))
        out_h = max(1, int(src_h * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (out_w, out_h), interpolation=interp)

        canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        x = (dst_w - out_w) // 2
        y = (dst_h - out_h) // 2
        canvas[y : y + out_h, x : x + out_w] = resized
        return canvas

    def _render_text_card(self) -> np.ndarray:
        frame = self._make_blank()
        cv2.putText(frame, self._text or "MIME VIDEO CLIENT", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 120), 3)
        now = time.strftime("%H:%M:%S")
        cv2.putText(frame, now, (40, self.height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 220), 2)
        return frame

    def read(self) -> Optional[np.ndarray]:
        try:
            if self.mode == "webcam":
                if self._cap is None:
                    self._cap = cv2.VideoCapture(self._webcam_index)
                    try:
                        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
                        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
                    except Exception:
                        pass

                ret, frame = self._cap.read()
                if not ret:
                    return self._last_frame
                out = self._fit_frame_letterbox(frame)
                self._last_frame = out
                return out

            elif self.mode == "abs":
                if self._renderer is None:
                    return self._last_frame
                try:
                    frame = self._renderer()
                except Exception as e:
                    self.log(f"[VideoClient] ABS renderer error: {e}")
                    return self._last_frame
                if frame is None:
                    return self._last_frame
                out = self._fit_frame_letterbox(frame)
                self._last_frame = out
                return out

            elif self.mode == "text":
                out = self._render_text_card()
                self._last_frame = out
                return out

            else:
                # blank
                return self._last_frame

        except Exception as e:
            self.log(f"[VideoClient] read error: {e}")
            return self._last_frame

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


__all__ = ["VideoClient"]
