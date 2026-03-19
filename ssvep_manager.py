import numpy as np
from omni.isaac.core.utils.prims import get_prim_at_path, set_prim_attribute_value

class SSVEPManager:
    def __init__(self):
        self.stimuli_mat = {
            "Green_Mat": {"path": "/World/Looks/Green_Mat", "freq": 20.0, "max_intensity": 2000.0},
            "Red_Mat": {"path": "/World/Looks/Red_Mat", "freq": 10.0, "max_intensity": 2000.0},
        }

    def update(self, current_time):
        for name, info in self.stimuli_mat.items():
            raw_sine = np.sin(2 * np.pi * info["freq"] * current_time)
            pulse_ratio = (raw_sine + 1.0) / 2.0 
            current_intensity = info["max_intensity"] * pulse_ratio

            shader_path = f"{info['path']}/Shader"
            if get_prim_at_path(shader_path):
                try:
                    set_prim_attribute_value(shader_path, "inputs:emissive_intensity", current_intensity)
                except:
                    set_prim_attribute_value(info['path'], "inputs:emissive_intensity", current_intensity)