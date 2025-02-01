import cv2
import numpy as np

class Camera():
    def __init__(self, sim, visionSensorHandle):
        self.sim = sim
        self.handle = visionSensorHandle
        self.K, self.near, self.far = self.get_intrinsics()
        self.extrinsic = self.sim.getObjectMatrix(self.handle, -1) # Returns list can be reshaped to 3X4

    def get_intrinsics(self):
        """Calculate the intrinsic matrix of the camera"""
        # Get resolution
        resX = self.sim.getObjectInt32Param(self.handle, self.sim.visionintparam_resolution_x)
        resY = self.sim.getObjectInt32Param(self.handle, self.sim.visionintparam_resolution_y)

        # Get perspective angle (x-axis)
        xAngle = self.sim.getObjectFloatParam(self.handle, self.sim.visionfloatparam_perspective_angle)

        # Compute y-axis angle
        ratio = resX / resY
        yAngle = xAngle if resX <= resY else 2 * np.arctan(np.tan(xAngle / 2) / ratio)

        # Calculate fov
        fx = resX / (2 * np.tan(xAngle / 2))
        fy = resY / (2 * np.tan(yAngle / 2))

        #  Principal points
        cx = resX / 2
        cy = resY / 2

        intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        # Get near & far values
        near = self.sim.getObjectFloatParam(self.handle, self.sim.visionfloatparam_near_clipping)
        far = self.sim.getObjectFloatParam(self.handle, self.sim.visionfloatparam_far_clipping)

        return intrinsic_matrix, near, far
    
    def get_rgb_img(self):
        """Get rgb image from vision sensor"""
        img, resolution = self.sim.getVisionSensorImg(self.handle)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
        flipped = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        return img, flipped, resolution[0], resolution[1]

    def get_depth_map(self):
        """Get depth map from vision sensor"""
        depth, resolution = self.sim.getVisionSensorDepth(self.handle)
        depth = np.frombuffer(depth, dtype=np.float32).reshape(resolution[1], resolution[0])
        depth_normalized = cv2.flip(depth, 0)

        # Normalize to 0-255 for visualization
        depth_normalized = cv2.normalize(depth_normalized, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        
        return depth, depth_normalized

    def pixel_to_world(self, pixel, depth):
        """Calculate 3d location of pixel"""
        # Get pixel parameters
        u, v = pixel
        z = depth[v][u]

        # Intrinsics
        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]

        # Scale depth
        z = self.near + z * (self.far - self.near)

        # Compute 3d coordinates
        x_c = (cx - u) * z / fx
        y_c = (v - cy) * z / fy
        z_c = z

        # Homogenous coords
        camera_coords = [x_c, y_c, z_c, 1]

        # Conver to world
        world_coords = self.sim.multiplyVector(self.extrinsic, camera_coords)

        return world_coords[:3]