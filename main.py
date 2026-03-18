import sys
print("🚨 [디버그 0] 진짜 main.py 실행 시작!", flush=True)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

print("🚨 [디버그 1] 엔진 부팅 완료! 기본 모듈 로드 중...", flush=True)

import os
import time
import numpy as np
import carb
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage, is_stage_loading
import omni.kit.viewport.utility as vp_utils

# 커스텀 모듈 임포트 (경로 엇갈림 방지)
sys.path.append(os.path.expanduser("~/isaac_vr_project"))
from ssvep_manager import SSVEPManager
from scene_manager import SceneManager
from robot_manager import RobotManager

print("🚨 [디버그 2] 모듈 임포트 완료! USD 로딩 시도...", flush=True)
usd_path = os.path.expanduser("~/isaac_vr_project/stack_blocks_with_human.usd")
open_stage(usd_path)

print("🚨 [디버그 3] USD 로딩 대기 중 (무한루프 방지 탑재)...", flush=True)
timeout = 0
while is_stage_loading() and timeout < 500:
    simulation_app.update()
    timeout += 1

if timeout >= 500:
    print("⚠️ [경고] 로딩이 너무 오래 걸립니다. 일단 강제 진행합니다.", flush=True)
else:
    print(f"🎉 [디버그 4] {timeout} 프레임 만에 USD 로딩 완료!", flush=True)

print("🚨 [디버그 5] 월드 및 로봇 매니저 초기화...", flush=True)
world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
robot_manager = RobotManager(world)

world.reset()
world.play()

print("🚨 [디버그 6] 물리 엔진 예열 중...", flush=True)
for _ in range(20):
    world.step(render=True)

print("🚨 [디버그 7] 컨트롤러 및 씬(큐브) 매니저 초기화...", flush=True)
ssvep_manager = SSVEPManager()
scene_manager = SceneManager(spawn_z=0.04026)

robot_manager.initialize_robot()
robot_manager.setup_controller()

settings = carb.settings.get_settings()
settings.set("/app/runLoops/main/rateLimitEnabled", True)
settings.set("/app/runLoops/main/rateLimitFrequency", 60)

print("🚨 [디버그 8] 큐브 랜덤 배치 실행 중...", flush=True)
scene_manager.randomize_cubes()

for _ in range(10):
    world.step(render=True)

print("🎉 [디버그 9] 모든 준비 완료! 메인 루프 진입!", flush=True)

# 시뮬레이션 제어 변수
current_cube_idx = 0
stack_goal_pos = np.array([-0.3, -0.1, 0.04026])

# 메인 루프
while simulation_app.is_running():
    t = time.time()
    
    # SSVEP 업데이트
    ssvep_manager.update(t)

    # 로봇 컨트롤 업데이트
    if current_cube_idx < len(scene_manager.red_cubes):
        target_cube = scene_manager.red_cubes[current_cube_idx]
        cube_pos, _ = target_cube.get_world_pose()
        
        actions = robot_manager.get_action(cube_pos, stack_goal_pos)
        
        if actions is not None:
            robot_manager.apply_action(actions)
        else:
            if int(t * 10) % 20 == 0:
                print(f"⚠️ IK 계산 포기! (로봇이 닿을 수 없는 각도입니다) - 큐브 위치: {cube_pos}")

        if robot_manager.is_done():
            print(f"🎉 {current_cube_idx + 1}번째 큐브 완료!")
            robot_manager.reset_controller()
            current_cube_idx += 1
            stack_goal_pos[2] += 0.05 # 위로 쌓기

    world.step(render=True)

simulation_app.close()