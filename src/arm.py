import time
import utils
import numpy as np

class RobotArm:
    def __init__(self, sim, script, vis_path=False, name='/UR5'):
        self.sim = sim
        self.script = script
        self.name = name
        self.num_joints = 6
        self.vis_path = vis_path
        
        # Get parameters
        self.params = utils.call_lua_function(self.sim, self.script, 'getParams')

        self.target_params = {}

    def _create_pose(self, position, quaternion):
        """Create a pose from position and quaternion"""
        return np.concatenate([position, quaternion])

    def _set_target_config(self, config):
        """Set target joint positions"""
        for joint, pos in zip(self.params['joints'], config):
            self.sim.setJointTargetPosition(joint, pos)

    def followPath(self, path):
        """Simulate the arm movement along the generated path in the simulation"""
        configs = [path[i:i+self.num_joints] for i in range(0, len(path), self.num_joints)]

        for config in configs:
            self._set_target_config(config)
            time.sleep(0.075)
        time.sleep(1.0)

    def get_target_params(self, location, config=None):
        """Get data for location and create passiveShape of config"""

        # Location data
        loc = self.target_params[location]
        config = config if config else loc['config']
        path = loc['path']

        # Create passive shape
        passiveShape = utils.call_lua_function(self.sim, self.script, 'createPassiveShape', config)

        return path, passiveShape 
    
    def moveWithPath(self, pose=None, location=None):
        """Find a path to pose and follow it"""

        if location:
            print(f"Moving to pose: {location}...")
        else:
            print(f"Moving to pose: {pose}...")
        
        if not location:
            # Get params from lua using OMPL
            result = utils.call_lua_function(self.sim, self.script, 'getPath', pose)
            if not result:
                return False
            
            path, passiveShape = result
        
        else:
            # If path is known, get saved params
            path, passiveShape = self.get_target_params(location)

        if self.vis_path:
            # Visualize path for 3 seconds before moving
            shapes = utils.call_lua_function(self.sim, self.script, 'visualizePath', path, 20)
            time.sleep(3)
            self.sim.removeObjects(shapes)

        # Simulate path movement
        self.followPath(path)
        self.sim.removeObjects([passiveShape])
        time.sleep(0.15)

        return path
    
    def moveHome(self, item_path=None, location=None):
        """Find path to home config from saved target location"""

        print("Moving Home...")

        if location:
            # Get data and flip path
            path, passiveShape = self.get_target_params(location, config=self.params['homeConfig'])
        else:
            # If we are going back from picking up an item
            path = item_path
            passiveShape = utils.call_lua_function(self.sim, self.script, 'createPassiveShape', self.params['homeConfig'])

        if self.vis_path:
            # Visualize path
            shapes = utils.call_lua_function(self.sim, self.script, 'visualizePath', path, 20)
            time.sleep(3)
            self.sim.removeObjects(shapes)

        # Simulate path movement
        path = sum([path[i:i + self.num_joints] for i in range(0, len(path), self.num_joints)][::-1],[])
        self.followPath(path)
        self.sim.removeObjects([passiveShape])
        time.sleep(0.15)

        return True
    
    def pick_and_place(self, pick, place):
        """Execute pick and place operation"""

        # Create scene poses        
        pick[2] += self.params['heightDiff']
        pickPose = self._create_pose(pick, self.params['downOriQuat'])
    
        # Move to pick
        success = self.moveWithPath(pickPose)
        if not success:
            return False

        item_path = success  # Save path to item to reverse it

        # Move down until item is detected
        print("Moving down to target...")
        while True:
            curr_pose = self.sim.getObjectPose(self.params['robotTarget'], -1)
            curr_pose[2] -= 0.001  # Step down incrementally
            self.sim.setObjectPose(self.params['robotTarget'], -1, curr_pose)

            # Move using IK
            success = utils.call_lua_function(self.sim, self.script, 'moveToPose', curr_pose)
            if not success:
                return False

            # Check the suction sensor
            object = utils.call_lua_function(self.sim, self.script, 'detectSuctionSensor')
            if object:
                print("Item detected! Stopping descent.")
                break

        # Pick item
        utils.call_lua_function(self.sim, self.script, 'toggleSuction', object, False)

        # Lift from table with IK
        success = utils.call_lua_function(self.sim, self.script, 'moveToPose', pickPose)
        if not success:
            return False

        # Move home
        success = self.moveHome(item_path)
        if not success:
            return False
        
        # Move to place
        success = self.moveWithPath(location=place)
        if not success:
            return False
        
        # Drop item
        utils.call_lua_function(self.sim, self.script, 'toggleSuction', object, True)
        
        # Move home
        success = self.moveHome(location=place)
        if not success:
            return False
        
        return True
    
    def calculate_home_target_trajectories(self, locations):
        """
            Calculate paths and trajectories from home config to each available location for reuse
        """
        
        print(f"Calculating paths from Home to {len(locations)} target locations\n")

        utils.call_lua_function(self.sim, self.script, 'initialParams', True)

        shapes = None
        
        for locName, locPos in locations.items():
            print(f'Finding path for {locName}')
            locPose = self._create_pose(locPos, self.params['downOriQuat'])

            self.target_params[locName] = utils.call_lua_function(self.sim, self.script, 'findHomeTargetPath', locPose)
            
            if self.vis_path:
                shapes = utils.call_lua_function(self.sim, self.script, 'visualizePath', self.target_params[locName]['path'], 20)
                time.sleep(10.0)
                self.sim.removeObjects(shapes)

            self.followPath(self.target_params[locName]['path'])
            time.sleep(2.0)
            
            self.moveHome(location=locName)
            time.sleep(2.0) 

        utils.call_lua_function(self.sim, self.script, 'initialParams', False)

