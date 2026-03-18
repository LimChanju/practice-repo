import numpy as np
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import PickPlaceController

class RobotManager:
    def __init__(self, world):
        self.robot = world.scene.add(
            Franka(
                prim_path="/World/Fancy_Franka",
                name="my_franka",
                position=np.array([-0.3, -0.3, 0.0])
            )
        )
        self.controller = None

    def initialize_robot(self):
        self.robot.initialize()
        print(f"로봇 연결 상태: {self.robot.get_joint_positions()}")

    def setup_controller(self):
        self.controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self.robot.gripper,
            robot_articulation=self.robot,
        )

    def get_action(self, target_pos, goal_pos):
        return self.controller.forward(
            picking_position=target_pos,
            placing_position=goal_pos,
            current_joint_positions=self.robot.get_joint_positions(),
        )

    def apply_action(self, actions):
        if actions is not None:
            self.robot.apply_action(actions)

    def is_done(self):
        return self.controller.is_done()

    def reset_controller(self):
        self.controller.reset()