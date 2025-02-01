import random
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def euler_to_quaternion(euler, seq='xyz'):
    """Conver euler angles to a quaternion"""
    return R.from_euler(seq, euler, degrees=False).as_quat()

def rotation_to_quaternion(rotation_matrix):
    """Convert a rotation matrix to a quaternion"""
    return R.from_matrix(rotation_matrix).as_quat()

def quaternion_to_rotation(quaternion):
    """Conver a quaternion to a rotation matrix"""
    return R.from_quat(quaternion).as_matrix()

def call_lua_function(sim, script_handle, func_name, *args):
    """Calls a Lua function from the simulation script."""
    try:
        result = sim.callScriptFunction(func_name, script_handle, *args)
        return result
    except Exception as e:
        print(f"Error calling Lua function '{func_name}': {e}")
        raise

def create_red_dot(sim, position, size=0.02):
    """Create a red dot (sphere) in the scene at a given position."""
    # Convert position to a Python list
    position = position.tolist() if isinstance(position, np.ndarray) else position

    # Create a red sphere
    sphere = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [size, size, size], 1)
    sim.setObjectPosition(sphere, -1, position)
    sim.setShapeColor(sphere, None, sim.colorcomponent_ambient_diffuse, [1, 0, 0])  # Red color
    return sphere

def detect_objects(sim, detector, camera, target=None, isTall=False, visualize=False):
    """Detect objects and return interest points"""

    # Get camera data
    rgb, flipped, resX, resY = camera.get_rgb_img()
    depth, _ = camera.get_depth_map()

    # Get yolo results
    annotated_img, results = detector.detect_objects(flipped, [target] if target != None else None)

    world_coordinates = None

    # Find interest points
    if results[0]:
        result = results[0].boxes
        if len(result) == 1:
            world_coordinates = get_ip(sim, detector, camera, depth, result, resY, isTall)
    else:
        print(f'Could not detect {target} in the scene')
        return False

    if visualize:
        # Display annotated image
        plt.imshow(annotated_img)
        plt.title("YOLO Detections")
        plt.show()

    return world_coordinates 

def get_ip(sim, detector, camera, depth, result, resY, isTall=False):
    """"Returns interest point for picking up the object"""

    x1, y1, x2, y2 = result.xyxy[0].tolist()  # Bounding box corners
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Flip upside down
    y1 = resY - y1 - 1
    y2 = resY - y2 - 1
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # If its a tall item return the top, else the center of the box is our interest point    
    if not isTall:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2) + 5  # Center of the bounding box
    else:
        cx, cy = int((x1 + x2) / 2), y2 - 3

    # Get the class name
    class_id = int(result.cls)
    class_name = detector.model.names[class_id]

    # Convert pixel to world coordinates
    print('Returning interest point for ', {class_name})
    world_coords = camera.pixel_to_world((cx, cy), depth)

    return world_coords
