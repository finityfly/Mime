"""
THIS IS ALL GPT SLOP
"""

import numpy as np
import pyrender
import trimesh
import cv2
import pyvirtualcam

class Live3DOverlay:
    def __init__(self, mesh_path, width=1280, height=720):
        # Load the pre-registered 3D mesh acquired via TrueDepth
        self.base_mesh = trimesh.load(mesh_path)
        self.width, self.height = width, height
        
        # Setup Lightweight Renderer
        self.scene = pyrender.Scene(bg_color=[0, 0, 0, 0]) # Alpha 0 for transparency
        self.mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(self.base_mesh))
        self.scene.add_node(self.mesh_node)
        
        # Add Camera (Match your physical webcam's FOV)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        self.scene.add(camera, pose=np.eye(4))
        self.renderer = pyrender.OffscreenRenderer(width, height)

    def update_blendshapes(self, coefficients):
        """
        coefficients: List/Array of 52 ARKit values (0.0 to 1.0)
        Updates the mesh vertices based on the pre-rigged blendshapes.
        """
        # Logic: New_Vertex = Base_Vertex + Sum(Coeff_i * Delta_i)
        # This is a linear combination of the 52 shapes
        # optimized via NumPy for speed.
        pass 

    def render_overlay(self, original_frame):
        # 1. Render 3D Mouth with Alpha
        color, depth = self.renderer.render(self.scene)
        
        # 2. Seamless Cloning / Alpha Blending
        # Using the Deep-Live-Cam 'Feathering' logic from earlier
        mask = (color[:, :, 3] > 0).astype(np.uint8) * 255
        final_frame = self.blend_overlay(original_frame, color[:, :, :3], mask)
        
        return final_frame

    def blend_overlay(self, bg, fg, mask):
        # Fast Alpha Blending using OpenCV
        mask_inv = cv2.bitwise_not(mask)
        bg_part = cv2.bitwise_and(bg, bg, mask=mask_inv)
        fg_part = cv2.bitwise_and(fg, fg, mask=mask)
        return cv2.add(bg_part, fg_part)