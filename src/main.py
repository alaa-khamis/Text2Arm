import os
import time
import cv2
import argparse
import json
import utils

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from vision.yolo import YOLOv8Detector
from arm import RobotArm
from vision.camera import Camera
from nlp.llm import LLM

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = './vision/yolov8_combined.pt' # https://github.com/iki-wgt/yolov7_yolov8_benchmark_on_ycb_dataset
LLM_PATH = './nlp/flan-t5-finetuned'
CACHE_PATH_FILE = os.path.join(PROJECT_DIR,'./saved_paths.json')

# All available objects for detection
# You can load the wanted objects randomly using the add_random_object function from utils
YCB_OBJECTS = {
    "002": 'master_chef_can',
    "003": 'cracker_box',
    "004": 'sugar_box',
    "005": 'tomato_soup_can',
    "006": 'mustard_bottle',
    "007": 'tuna_fish_can',
    "008": 'pudding_box',
    "009": 'gelatin_box',
    "010": 'potted_meat_can',
    "011": 'banana',
    "019": 'pitcher_base',
    "021": 'bleach_cleanser',
    "024": 'bowl',
    "025": 'mug',
    "035": 'power_drill',
    "036": 'wood_block',
    "037": 'scissors',
    "040": 'large_marker',
    "051": 'large_clamp',
    "052": 'extra_large_clamp',
    "061": 'foam_brick'
}

# Which items exist in the scene - for finetuning llm
ITEMS_IN_SCENE = {
    'sugar_box',
    'large_clamp',
    'tuna_fish_can',
    'master_chef_can'
}

IS_TALL = {
    'sugar_box' : True,
    'large_clamp' : False,
    'tuna_fish_can' : False,
    'master_chef_can' : True
}

LOCATIONS = {
    'redBin' : [0.570, 0.375, 0.6],
    'yellowBin' : [0.050, 0.375, 0.6],
    'blueBin' : [-0.450, 0.375, 0.6]
}

def main():

    # Argument Parser
    parser = argparse.ArgumentParser(description="Run the robot arm simulation")
    parser.add_argument("--use_cached_paths", action="store_true", help="Use cached paths if available")
    parser.add_argument("--vis_path", action="store_true", help="Visualize the paths in the simulation")
    parser.add_argument("--vis_yolo", action="store_true", help="Visualize the YOLO detections")
    args = parser.parse_args()


    # Initialize the Remote API Client
    client = RemoteAPIClient()
    sim = client.require('sim')

    # Get script
    script = sim.getObject('/ArmControlScript')
    
    # Load yolo model
    yolo = YOLOv8Detector(os.path.join(PROJECT_DIR, YOLO_PATH))

    # Load llm
    llm = LLM(os.path.join(PROJECT_DIR, LLM_PATH), ITEMS_IN_SCENE, LOCATIONS, True)

    # Get vision sensor
    vision_sensor = sim.getObject('/camera/sensor')
    camera = Camera(sim, vision_sensor)

    # Start Simulation
    sim.startSimulation()
    time.sleep(2) # Wait 2 seconds until everything loads in the simulation

    # Main loop
    try:
        # Load arm controls
        arm = RobotArm(sim, script, args.vis_path)
        
        # Check if using cached_paths
        if args.use_cached_paths and os.path.exists(CACHE_PATH_FILE):
            with open(CACHE_PATH_FILE, 'r') as f:
                arm.target_params = json.load(f)
            print("Using saved paths")

            utils.call_lua_function(sim, script, 'toggleCollisionBox', False)
        else:
            # Calculate locations' paths
            arm.calculate_home_target_trajectories(LOCATIONS)

            # Save paths if caching enabled
            if args.use_cached_paths:
                with open(CACHE_PATH_FILE, 'w') as f:
                    json.dump(arm.target_params, f)
                print("Saved calculated paths")

        print("\n\n")

        # Print instructions
        print('Instructions: Enter your request from the model in a natural tone.\n \
                eg. "Move the tuna can to the red bin"\n \
                You can compound multiple requests in the same prompt.\n \
                eg. "Move the tuna can to the red bin and the sugar to the blue bin" \n \
                If you wish to check all of the detected items, type : "detect"\n\n')

        while True:
            
            # Get user request
            req = input('Enter you request: ')

            if req == 'detect':
                utils.detect_objects(sim, yolo, camera, visualize=True)

            
            elif req == 'exit':
                break

            else:
                res = llm.process_prompt(req)

                if not isinstance(res, list): 
                    print(res)
                    continue
                
                # Create a pick and place task for each one of the pairs
                for item, location in res:
                    print(f'Creating task for item: {item} and location: {location}')
    
                    item_coords = utils.detect_objects(sim, yolo, camera, item, isTall=IS_TALL[item], visualize=args.vis_yolo)

                    if not item_coords:
                        continue

                    success = arm.pick_and_place(item_coords, location)

                    if not success:
                        print('Something went wrong!')
                        break

                time.sleep(0.1)
        
    finally:
        print("Stopping the simulation...")
        sim.stopSimulation()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
