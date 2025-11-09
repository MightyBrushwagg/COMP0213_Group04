import numpy as np
import pandas as pd

class Data:
    def __init__(self, data=None, object_pos=[0,0,0]):
        if data == None:
            column_tmp  = {"x": [], "y": [], "z": [], "roll": [], "pitch": [], "yaw": [], "success": []}
            self.data = pd.DataFrame(column_tmp)
            all_points, all_angles = self.make_data(object_pos=object_pos, num_points=400)
            # print(all_angles.shape)
            self.data["x"] = all_points[:,0]
            self.data["y"] = all_points[:,1]
            self.data["z"] = all_points[:,2]
            self.data["roll"] = all_angles[:,0]
            self.data["pitch"] = all_angles[:,1]
            self.data["yaw"] = all_angles[:,2]
            print(self.data)

    def make_data(self, object_pos, num_points = 400, R = 0.5, height = 0):
        # Step 1: Generate random 3D Gaussian vectors
        coords = np.random.normal(size=(3, num_points))
        # Step 2: Calculate the magnitude of each vector
        distance_from_origin = np.linalg.norm(coords, ord=2, axis=0)
        # Step 3: Normalize the vectors
        normalised_coords = coords / distance_from_origin
        # Scale to desired radius
        points = R * normalised_coords
        # print(points.T.shape)
        # print(points)

        valid = np.array([[coord[0], coord[1], coord[2]] for coord in points.T if coord[2] >= height])
        # print(valid.shape)
        # print(valid)
        noise = np.random.normal(0, 0.01, size=(len(valid),3))
        positions = valid + noise
        # print(positions)
        
        orientations = np.array([self.generate_angle(position, object_pos) for position in positions])
        # print(orientations)
        return positions, orientations # Transpose to get shape (num_points, 3)
    
    def generate_angle(self, gripper_pos, object_pos):
        approach_pos = gripper_pos + np.array([0,0,0.1])
        off = 0.04 # Need to iterate on this value
        direction = np.array(approach_pos) - np.array(object_pos)
        direction[2] = 0 # No z term offset needed because offset doesn't change based on starting position 

        if np.linalg.norm(direction) > 0: # No zero lengths 
            direction = direction / np.linalg.norm(direction) # Normalize direction vector
            offset = direction * off # Scale by offset
        else:
            offset = np.array([0, 0, 0])

        print(offset)

        target_pos = np.array(object_pos) + offset # Apply offset
        direction = np.array(target_pos) - np.array(approach_pos)
        
        direction = direction/np.linalg.norm(direction)

        pitch = np.arcsin(-direction[2])
        yaw = np.arctan2(direction[1], direction[0])
        roll = np.pi

        return np.array([roll, pitch, yaw])

    def add_data(self, *args):
        self.data.loc[len(self.data)] = args
        print(self.data)