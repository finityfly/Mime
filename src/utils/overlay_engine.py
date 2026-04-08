from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
import pyvirtualcam


@dataclass
class OverlayConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: Optional[str] = None


class FrameSource:
    """Interface for pulling base video frames (webcam, file, render stream)."""

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError("FrameSource.read() is a placeholder.")


class BlendshapeSource:
    """Interface for retrieving current ARKit blendshape coefficients."""

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError("BlendshapeSource.read() is a placeholder.")


class BlendshapeOverlayRenderer:
    """Interface for rendering an overlay given a frame and blendshapes."""

    def render(self, frame_bgr: np.ndarray, blendshapes: np.ndarray) -> np.ndarray:
        raise NotImplementedError("BlendshapeOverlayRenderer.render() is a placeholder.")


class ZoomOverlayEngine:
    """
    Framework for Zoom overlay integration.

    Pipeline:
    1) Read base frames from FrameSource.
    2) Read blendshapes from BlendshapeSource.
    3) Render overlay with BlendshapeOverlayRenderer.
    4) Send final frames to virtual camera for Zoom.
    """

    def __init__(
        self,
        config: OverlayConfig,
        frame_source: Optional[FrameSource] = None,
        blendshape_source: Optional[BlendshapeSource] = None,
        overlay_renderer: Optional[BlendshapeOverlayRenderer] = None,
        log_callback: Callable[[str], None] = print,
    ):
        self.config = config
        self.frame_source = frame_source
        self.blendshape_source = blendshape_source
        self.overlay_renderer = overlay_renderer
        self.log = log_callback
        self.is_running = False

    def set_frame_source(self, frame_source: FrameSource) -> None:
        self.frame_source = frame_source

    def set_blendshape_source(self, blendshape_source: BlendshapeSource) -> None:
        self.blendshape_source = blendshape_source

    def set_overlay_renderer(self, overlay_renderer: BlendshapeOverlayRenderer) -> None:
        self.overlay_renderer = overlay_renderer

    def start(self) -> None:
        if self.frame_source is None:
            raise NotImplementedError("FrameSource is required to start Zoom overlay.")
        if self.blendshape_source is None:
            raise NotImplementedError("BlendshapeSource is required to start Zoom overlay.")
        if self.overlay_renderer is None:
            raise NotImplementedError("BlendshapeOverlayRenderer is required to start Zoom overlay.")

        self.is_running = True
        self.log("[Overlay] Starting virtual camera...")

        with pyvirtualcam.Camera(
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            backend=self.config.backend,
        ) as cam:
            self._run_loop(cam)

    def stop(self) -> None:
        self.is_running = False

    def _run_loop(self, cam: pyvirtualcam.Camera) -> None:
        while self.is_running:
            frame = self.frame_source.read()
            if frame is None:
                time.sleep(0.005)
                continue

            blendshapes = self.blendshape_source.read()
            if blendshapes is None:
                blendshapes = np.zeros(52, dtype=np.float32)

            output = self.overlay_renderer.render(frame, blendshapes)
            cam.send(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            cam.sleep_until_next_frame()


def alpha_blend(bg: np.ndarray, fg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Utility blend helper for overlay renderers."""
    mask_inv = cv2.bitwise_not(mask)
    bg_part = cv2.bitwise_and(bg, bg, mask=mask_inv)
    fg_part = cv2.bitwise_and(fg, fg, mask=mask)
    return cv2.add(bg_part, fg_part)
