from models import MODEL_CONFIG, MODELS

class DeviceWrapper:
    """Wrapper for device simulation"""

    def __init__(self, device_type, params):
        self.device_type = device_type
        self.params = params
        self.config = MODEL_CONFIG[device_type]

    def simulate(self, **voltage_params):
        """Run ML model simulation

        Args:
            **voltage_params: Voltage sweep parameters (e.g., Vg={'start': 0, 'end': 0.5, 'step': 21})

        Returns:
            dict: Simulation results with device_params and simulation_data
        """
        full_params = {**self.params, **voltage_params}
        return self.config['simulation_func'](full_params)

    def __repr__(self):
        voltage_info = self.config.get('voltage_params')
        if voltage_info:
            voltage_str = f", voltage_params={voltage_info}"
        else:
            voltage_str = " (no voltage sweep needed)"
        return f"DeviceWrapper(type='{self.device_type}', params={list(self.params.keys())}{voltage_str})"

    def help(self):
        """Print help for this device instance"""
        print(f"Device: {self.device_type}")
        print(f"Device parameters: {self.config['device_params']}")
        voltage_info = self.config.get('voltage_params')
        if voltage_info:
            print(f"Voltage parameters: {voltage_info}")
            print(f"  Example: device.simulate(Vg={{'start': 0, 'end': 0.5, 'step': 21}}, Vd={{'start': 0, 'end': 0.5, 'step': 11}})")
        else:
            print("Voltage parameters: None (returns fixed characteristic curves)")
            print("  Example: device.simulate()")


class DeviceFactory:
    """Factory for creating device instances"""

    @staticmethod
    def create(device_type, **params):
        """Create a device model instance

        Args:
            device_type (str): Device type (e.g., 'CNTFET', 'NMOS', 'HFET')
            **params: Device parameters

        Returns:
            DeviceWrapper: Device instance with simulate() method

        Example:
            >>> device = DeviceFactory.create('CNTFET', tox=2, Lg=12, eps_ox=21.0)
            >>> result = device.simulate(Vg={'start': 0, 'end': 0.5, 'step': 21})
        """
        if device_type not in MODEL_CONFIG:
            available = ', '.join(MODEL_CONFIG.keys())
            raise ValueError(f"Unknown device '{device_type}'. Available: {available}")

        return DeviceWrapper(device_type, params)

    @staticmethod
    def list_devices():
        """List all available device types

        Returns:
            list: Available device type names
        """
        return list(MODEL_CONFIG.keys())

    @staticmethod
    def get_device_params(device_type):
        """Get required parameters for a device type

        Args:
            device_type (str): Device type name

        Returns:
            list: List of required parameter names

        Example:
            >>> DeviceFactory.get_device_params('CNTFET')
            ['tox', 'Lg', 'eps_ox', 'd_cnt', 'V_th', 'sca_flag']
        """
        if device_type not in MODEL_CONFIG:
            available = ', '.join(MODEL_CONFIG.keys())
            raise ValueError(f"Unknown device '{device_type}'. Available: {available}")

        return MODEL_CONFIG[device_type]['device_params']

    @staticmethod
    def help(device_type=None):
        """Print help information for devices

        Args:
            device_type (str, optional): Specific device type, or None for all devices

        Example:
            >>> DeviceFactory.help()
            >>> DeviceFactory.help('CNTFET')
        """
        if device_type is None:
            print("Available devices:")
            for device in MODEL_CONFIG.keys():
                config = MODEL_CONFIG[device]
                params = config['device_params']
                voltage_info = config.get('voltage_params')
                if voltage_info:
                    print(f"  - {device}")
                    print(f"      Device params: {params}")
                    print(f"      Voltage params: {voltage_info} (required in simulate())")
                else:
                    print(f"  - {device}")
                    print(f"      Device params: {params}")
                    print(f"      Voltage params: None (fixed curves)")
        else:
            if device_type not in MODEL_CONFIG:
                available = ', '.join(MODEL_CONFIG.keys())
                print(f"Unknown device '{device_type}'. Available: {available}")
                return

            config = MODEL_CONFIG[device_type]
            params = config['device_params']
            voltage_info = config.get('voltage_params')
            print(f"{device_type}")
            print(f"  Device parameters: {params}")
            if voltage_info:
                print(f"  Voltage parameters: {voltage_info}")
                print(f"  Usage: device = DeviceFactory.create('{device_type}', ...)")
                print(f"         result = device.simulate(Vg={{...}}, Vd={{...}})")
            else:
                print(f"  Voltage parameters: None")
                print(f"  Usage: device = DeviceFactory.create('{device_type}', ...)")
                print(f"         result = device.simulate()")


__version__ = '0.1.0'
__all__ = ['DeviceFactory', 'DeviceWrapper', 'MODEL_CONFIG', 'MODELS']
