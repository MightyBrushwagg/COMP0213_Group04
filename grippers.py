import pybullet as p
import pybullet_data
import time
from abc import abstractmethod

class Gripper():
    """Base gripper class that defines common gripper behavior."""
    def __init__(self, urdf_path, base_position):
        self.urdf_path = urdf_path
        self.base_position = base_position
        self.id = None
        self.constraint_id = None
        self.grasp_moving = False

    def load(self):
        """Load gripper into the PyBullet world."""
        self.id = p.loadURDF(self.urdf_path, *self.base_position)
        return self.id

    def attach_fixed(self, offset):
        """Attach gripper to a fixed world position."""
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=offset,
            childFramePosition=self.base_position
        )

    def move(self, z, yaw=0.0):
        """Move gripper to a new height and yaw."""
        
        if self.constraint_id is None:
            raise ValueError("Gripper must be fixed before moving.")
        p.changeConstraint(
            self.constraint_id,
            jointChildPivot=[0.5, 0.3, z],
            jointChildFrameOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
            maxForce=50
        )
 
    def update_camera(self, z, yaw):
        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=50 + (yaw * 180 / 3.1416),
            cameraPitch=-60,
            cameraTargetPosition=[0.5, 0.3, z]
        )


class TwoFingerGripper(Gripper):
    """A specific two-finger gripper inheriting from Gripper class.
    Initialize the gripper using default base_position=(0.5, 0.3, 0.7) and the urdf name: pr2_gripper.urdf setting up the relevant attributes.
    You need to use the parent's initializer accordingly.
    """ 
    def __init__(self, base_position=(0,0,0)):
       # Setup the gripper
        super().__init__("pr2_gripper.urdf", base_position=base_position)

    def start(self):
        """Open gripper at start."""
        initial_positions = [0.550569, 0.0, 0.549657, 0.0]
        for i, pos in enumerate(initial_positions):
            p.resetJointState(self.id, i, pos)

    def open(self):
        """Open the gripper fingers."""
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.0, maxVelocity=1, force=10)

    def close(self):
        """Close the gripper fingers."""
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.1, maxVelocity=1, force=10)

    def grasp_and_lift(self, obj, lift_height=0.4, lift_steps=150):
        """Perform grasping and lifting sequence with sustained gripping force."""
        yaw_angle = 0.0

        grasp_height = obj.grasp_height

        # --- Set friction on contact surfaces ---
        p.changeDynamics(obj.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)
        p.changeDynamics(self.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)

        # --- Move above object by calling the move function inherited from the parent
        self.move(obj.position[2] + 0.1, yaw_angle)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)

        # --- Lower onto object ---
        self.move(grasp_height, yaw_angle)
        print("\033[93mmove to the grasp height")
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)

        # --- Close gripper strongly ---
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.12, force=300, maxVelocity=2)
        for _ in range(100):  # allow contact to form
            p.stepSimulation()
            time.sleep(1./240.)

        # --- Continuous hold while lifting step-by-step ---
        z_current = grasp_height
        z_target = lift_height
        z_step = (z_target - z_current) / lift_steps

        for _ in range(lift_steps):
            z_current += z_step
            self.move(z_current, yaw_angle)

            # Continuously reapply strong grip to prevent slip
            for joint in [0, 2]:
                p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                        targetPosition=0.12, force=400, maxVelocity=2)

            p.stepSimulation()
            time.sleep(1./240.)

    def get_position(self):
        pass