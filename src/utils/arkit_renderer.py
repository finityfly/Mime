import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"

if "PYOPENGL_PLATFORM" in os.environ:
    del os.environ["PYOPENGL_PLATFORM"]

os.environ["PYOPENGL_ERROR_CHECKING"] = "0"

import cv2
import torch
import pyrender
import gltflib
import numpy as np
import time
import threading

# Performance and compatibility fixes
os.environ["PYOPENGL_ERROR_CHECKING"] = "0"
np.product = np.prod 

# 1. Canonical ARKit standard
ARKIT_BLENDSHAPE_NAMES = [
    'eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft', 
    'eyeSquintLeft', 'eyeWideLeft', 'eyeBlinkRight', 'eyeLookDownRight', 'eyeLookInRight', 
    'eyeLookOutRight', 'eyeLookUpRight', 'eyeSquintRight', 'eyeWideRight', 'jawForward', 
    'jawLeft', 'jawRight', 'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker', 
    'mouthRight', 'mouthLeft', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownRight', 
    'mouthFrownLeft', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 
    'mouthStretchRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 
    'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 
    'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'browDownLeft', 
    'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 
    'cheekSquintLeft', 'cheekSquintRight', 'noseSneerLeft', 'noseSneerRight', 'tongueOut'
]

# 2. BEAT training order
MODEL_OUTPUT_BLENDSHAPE_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut"
]

_metadata_name_to_idx = {name: i for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES)}
_model_name_to_idx = {name: i for i, name in enumerate(MODEL_OUTPUT_BLENDSHAPE_NAMES)}
BEAT_TO_METADATA_IDX = {
    beat_idx: _metadata_name_to_idx[name]
    for name, beat_idx in _model_name_to_idx.items()
    if name in _metadata_name_to_idx
}

class ARKitRenderer:
    def __init__(self, glb_path, engine=None, width=1280, height=720):
        self.engine = engine
        self.w, self.h = width, height
        self.glb_path = glb_path
        self.is_running = False
        self.renderer = None
        
        # Load GLB
        self.gltf_data = gltflib.GLTF.load(str(glb_path))
        self.scene = pyrender.Scene.from_gltflib_scene(self.gltf_data, bg_color=[0, 0, 0, 0])
        
        # Camera Setup (Adjusted to frame the face based on your previous logic)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=width/height)
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = [0.0, 1.3, 0.4] 
        self.scene.add(camera, pose=cam_pose)
        
        # Lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        self.scene.add(light, pose=cam_pose)

        # Mapping for target names
        self.mesh_target_names = self._build_mesh_target_name_map(self.gltf_data)
        
        # Identify morphable nodes
        self.morph_nodes = []
        for node in self.scene.nodes:
            if node.mesh is not None:
                for prim in node.mesh.primitives:
                    if hasattr(prim, "targets") and prim.targets is not None:
                        num_targets = int(prim.targets.positions.shape[0])
                        node.mesh.weights = np.zeros(num_targets, dtype=np.float32)
                        mesh_name = node.name or node.mesh.name or "Unnamed_Mesh"
                        self.morph_nodes.append((node, num_targets, self.mesh_target_names.get(mesh_name)))
                        break

    def _build_mesh_target_name_map(self, gltf_obj):
        mesh_target_names = {}
        if not gltf_obj.model.meshes: return mesh_target_names
        for mesh_def in gltf_obj.model.meshes:
            mesh_name = getattr(mesh_def, "name", "Unnamed_Mesh")
            extras = getattr(mesh_def, "extras", {})
            if isinstance(extras, dict) and "targetNames" in extras:
                mesh_target_names[mesh_name] = extras["targetNames"]
        return mesh_target_names

    def _apply_weights(self, raw_model_output):
        """Remaps model output to ARKit and applies to mesh weights."""
        remapped_weights = np.zeros(len(ARKIT_BLENDSHAPE_NAMES), dtype=np.float32)
        for beat_idx, arkit_idx in BEAT_TO_METADATA_IDX.items():
            if beat_idx < len(raw_model_output):
                remapped_weights[arkit_idx] = raw_model_output[beat_idx]

        for node, expected_length, target_names in self.morph_nodes:
            weights = np.zeros(expected_length, dtype=np.float32)
            if target_names:
                for target_idx, target_name in enumerate(target_names):
                    src_idx = _metadata_name_to_idx.get(target_name)
                    if src_idx is not None and src_idx < len(remapped_weights):
                        weights[target_idx] = float(remapped_weights[src_idx])
            else:
                copy_len = min(expected_length, len(remapped_weights))
                weights[:copy_len] = remapped_weights[:copy_len]
            
            node.mesh.weights = weights

    def render_frame(self) -> np.ndarray:
        """Called by threaded consumers to render a BGR frame."""
        if self.renderer is None:
            print(f"[Renderer] Initializing OpenGL context on thread: {threading.current_thread().name}")
            self.renderer = pyrender.OffscreenRenderer(self.w, self.h)

        if self.engine:
            model_weights = self.engine.get_latest_weights()
            if model_weights is not None:
                self._apply_weights(model_weights)

        # Render and convert to BGR for OpenCV
        color, _ = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        return cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)

    def start_display_loop(self):
        """Simple loop to show only the avatar window."""
        self.is_running = True
        window_name = "Avatar Renderer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print(f"[Renderer] Starting display loop. Press 'q' to exit.")
        while self.is_running:
            frame = self.render_frame()
            cv2.imshow(window_name, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.close()

    def close(self):
        self.is_running = False
        if self.renderer:
            self.renderer.delete()
        cv2.destroyAllWindows()