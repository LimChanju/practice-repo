import numpy as np
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import XFormPrim

class SceneManager:
    def __init__(self, spawn_z=0.04026):
        self.spawn_x_range = [0.0, 0.2]
        self.spawn_y_range = [-0.1, 0.3]
        self.spawn_z = spawn_z
        self.placed_positions = []
        
        self.red_cube_paths = ["/World/Red_Cube01", "/World/Red_Cube02", "/World/Red_Cube03"]
        self.green_cube_paths = ["/World/Green_Cube_1", "/World/Green_Cube_2", "/World/Green_Cube_3"]
        
        self.red_cubes = self._load_cubes(self.red_cube_paths)
        self.green_cubes = self._load_cubes(self.green_cube_paths)

    def _load_cubes(self, paths):
        cubes = []
        for path in paths:
            if get_prim_at_path(path):
                cubes.append(XFormPrim(path))
        return cubes

    def get_safe_spawn_pos(self):
        for _ in range(50):
            new_pos = np.array([
                np.random.uniform(self.spawn_x_range[0], self.spawn_x_range[1]),
                np.random.uniform(self.spawn_y_range[0], self.spawn_y_range[1]),
                self.spawn_z
            ])
            if all(np.linalg.norm(new_pos[:2] - p[:2]) > 0.08 for p in self.placed_positions):
                return new_pos
        return new_pos

    def randomize_cubes(self):
        print("모든 큐브의 초기 위치를 무작위로 섞습니다...")
        self.placed_positions = []
        
        for cube in self.red_cubes:
            new_pos = self.get_safe_spawn_pos()
            self.placed_positions.append(new_pos)
            cube.set_world_pose(position=new_pos)
            
        print(f"✅ 인식된 초록 큐브 개수: {len(self.green_cubes)}개")
        for cube in self.green_cubes:
            new_pos = self.get_safe_spawn_pos()
            self.placed_positions.append(new_pos)
            cube.set_world_pose(position=new_pos)