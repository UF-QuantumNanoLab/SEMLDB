from utils.registry import Registry

MODELS = Registry("MODELS")

from .CNTFET.cntfet import CNTFET  # noqa: F401
from .HFET.hfet import HFET  # noqa: F401
from .NMOS.nmos import NMOS  # noqa: F401
from .DiamondFET.diamondfet import DiamondFET
from .NanosheetFET.nanosheet import NanosheetFET  # noqa: F401


def create_model_config():
    """Create MODEL_CONFIG from registered models"""
    config = {}
    for device_name in MODELS.get_all_reg():
        print(device_name)
        device_class = MODELS.get(device_name)
        config[device_name] = {
            'simulation_func': device_class.simulation_func,
            'device_params': device_class.device_params,
            'postprocess': device_class.postprocess,
            'voltage_params': device_class.voltage_params,
        }
    return config

MODEL_CONFIG = create_model_config()
