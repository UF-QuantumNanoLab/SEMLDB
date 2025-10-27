# semldb - Semiconductor Machine Learning Models and Database

Semiconductor device simulation using machine learning models. Supports multiple device types including CNTFET, NMOS, and HFET.

## Quick Start
## Table of Contents
- [Installation and Usage](#installation-and-usage)
- [Adding a New Device](#adding-a-new-device)
- [Testing Your Device](#testing-your-device)

---

## Installation and Usage

### Installation

#### Option 1: Development Mode (Recommended for Development)

```bash
cd /path/to/semldb/backend
pip install -e .
```

This creates a symbolic link, so code changes are immediately reflected.

#### Option 2: Regular Installation

```bash
pip install git+https://github.com/QimaoYang/SEMLDB.git
```

### Basic Usage

#### 1. List Available Devices

```python
from semldb import DeviceFactory

# Show all devices
DeviceFactory.help()

# Output:
# Available devices:
#   - CNTFET
#       Device params: ['tox', 'Lg', 'eps_ox', 'd_cnt', 'V_th', 'sca_flag']
#       Voltage params: ['Vg', 'Vd'] (required in simulate())
#   - NMOS
#       Device params: ['Lg', 'THF', 'XjSD']
#       Voltage params: None (fixed curves)
#   - HFET
#       Device params: ['Lsg', 'Lgd', 'Lg', 'hpas', 'hAlGaN', 'hch', 'hg']
#       Voltage params: None (fixed curves)
```

#### 2. Get Device Parameters

```python
# Method 1: Using help()
DeviceFactory.help('CNTFET')

# Method 2: Programmatically
params = DeviceFactory.get_device_params('CNTFET')
print(params)  # ['tox', 'Lg', 'eps_ox', 'd_cnt', 'V_th', 'sca_flag']
```

#### 3. Create and Use Device

```python
from semldb import DeviceFactory

# Create device instance
device = DeviceFactory.create(
    'CNTFET',
    tox=2.0,
    Lg=12.0,
    eps_ox=21.0,
    d_cnt=1.02,
    V_th=0.258,
    sca_flag=0
)

# Check device info
print(device)
device.help()

# Run simulation
result = device.simulate(
    Vg={'start': 0, 'end': 0.5, 'step': 21},
    Vd={'start': 0, 'end': 0.5, 'step': 11}
)

# Access results
Id = result['simulation_data']['Id']
Vg = result['simulation_data']['Vg']
Vd = result['simulation_data']['Vd']
```

### Flexible Voltage Input (CNTFET)

CNTFET supports multiple voltage input formats:

```python
device = DeviceFactory.create('CNTFET', tox=2.0, Lg=12.0, eps_ox=21.0)

# 1. Dict format (grid sweep)
result = device.simulate(
    Vg={'start': 0, 'end': 0.5, 'step': 21},
    Vd={'start': 0, 'end': 0.5, 'step': 11}
)

# 2. List format
result = device.simulate(
    Vg=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    Vd=[0.0, 0.25, 0.5]
)

# 3. NumPy array
import numpy as np
result = device.simulate(
    Vg=np.linspace(0, 0.5, 21),
    Vd=np.array([0.1, 0.3, 0.5])
)

# 4. Single value
result = device.simulate(Vg=0.3, Vd=0.5)

# 5. Mixed formats (Id-Vd curve at fixed Vg)
result = device.simulate(
    Vg=0.4,
    Vd={'start': 0, 'end': 0.5, 'step': 51}
)

# 6. Mixed formats (Id-Vg curve at fixed Vd)
result = device.simulate(
    Vg=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    Vd=0.5
)
```

### Working with Results

```python
import numpy as np
import matplotlib.pyplot as plt

# Run simulation
result = device.simulate(Vg=0.4, Vd={'start': 0, 'end': 0.5, 'step': 51})

# Extract data
Vd = np.array(result['simulation_data']['Vd'])
Id = np.array(result['simulation_data']['Id'])

# Plot Id-Vd curve
plt.plot(Vd, Id.flatten())
plt.xlabel('Vd (V)')
plt.ylabel('Id (A)')
plt.title('CNTFET Id-Vd Characteristics')
plt.show()
```

---

## Adding a New Device

Adding a new device to semldb is straightforward thanks to the registry system. Follow these steps:

### Step 1: Create Device Directory

Create a new directory under `models/`:

```bash
mkdir -p models/YourDevice
touch models/YourDevice/__init__.py
touch models/YourDevice/yourdevice.py
```

### Step 2: Implement Device Model

In `models/YourDevice/yourdevice.py`:

```python
import torch
import numpy as np
from .. import MODELS

def run_simulation(parameters):
    """
    Main simulation function using ML model

    Args:
        parameters (dict): Device parameters

    Returns:
        dict: Simulation results with 'simulation_data' and 'device_params'
    """
    # Extract device parameters
    param1 = parameters.get('param1')
    param2 = parameters.get('param2')

    # Load your ML model
    model = YourMLModel()
    model_dir = os.path.dirname(__file__)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, 'your_model.pth'),
        map_location=torch.device('cpu')
    ))
    model.eval()

    # Run inference
    with torch.no_grad():
        output = model(input_data)

    # Format results
    return {
        'simulation_data': {
            'Vg': vg_values.tolist(),
            'Id': id_values.tolist(),
            # ... other outputs
        },
        'device_params': parameters
    }

def get_simulation_data(db_helper, parameters):
    """
    Optional: For API server to fetch from database
    Only needed if using database backend
    """
    complete_data, exact_match, distance, matched_params = \
        db_helper.get_simulation_data('YourDevice', parameters)

    if not complete_data:
        return None, False, None, None

    return complete_data, exact_match, distance, matched_params

@MODELS.register()
class YourDevice:
    simulation_func = run_simulation
    device_params = ['param1', 'param2', 'param3']
    voltage_params = ['Vg', 'Vd']  # or None if no voltage sweep
    postprocess = get_simulation_data  # for database queries
```

### Step 3: Import in `models/__init__.py`

Add import to `models/__init__.py`:

```python
from .CNTFET.cntfet import CNTFET
from .NMOS.nmos import NMOS
from .HFET.hfet import HFET
from .YourDevice.yourdevice import YourDevice  # Add this line
```

### Step 4: Add Model Weights to Package

Update `setup.py` to include your model files:

```python
package_data={
    'models': [
        'CNTFET/*.pth',
        'NMOS/**/*.pkl',
        'YourDevice/*.pth',  # Add this
        'YourDevice/*.pkl',  # Add this if needed
    ],
},
```

### Step 5: Done!

That's it! Your device is now available:

```python
from semldb import DeviceFactory

DeviceFactory.help()  # Will show your device
device = DeviceFactory.create('YourDevice', param1=..., param2=...)
result = device.simulate()
```

---

## Testing Your Device

### Manual Testing

```python
from semldb import DeviceFactory

# Test device creation
device = DeviceFactory.create('YourDevice', param1=value1, param2=value2)

# Test simulation
result = device.simulate()

# Verify output structure
assert 'simulation_data' in result
assert 'device_params' in result
print("Output keys:", result['simulation_data'].keys())
```

### Example Test Script

Create `test_yourdevice.py`:

```python
import numpy as np
from semldb import DeviceFactory

def test_yourdevice():
    # Create device
    device = DeviceFactory.create(
        'YourDevice',
        param1=10.0,
        param2=5.0
    )

    # Run simulation
    result = device.simulate()

    # Check output structure
    assert 'simulation_data' in result
    assert 'device_params' in result

    # Check data shapes
    data = result['simulation_data']
    assert len(data['Vg']) > 0
    assert len(data['Id']) > 0

    print("✓ All tests passed")

if __name__ == '__main__':
    test_yourdevice()
```

---

## Tips and Best Practices

### 1. Model File Paths
Always use relative paths from the module directory:

```python
import os
model_dir = os.path.dirname(__file__)
model_path = os.path.join(model_dir, 'model.pth')
```

### 2. Device Parameters
- Keep parameter names consistent with your domain
- Document valid ranges in docstrings
- Use sensible defaults where possible

### 3. Output Format
Always return data in this structure:

```python
{
    'simulation_data': {
        'Vg': [...],
        'Id': [...],
        # ... other outputs
    },
    'device_params': parameters
}
```

### 4. Voltage Parameters
- Set `voltage_params = None` if device returns fixed curves
- Set `voltage_params = ['Vg', 'Vd']` if device requires voltage sweep
- Implement flexible input parsing if needed (see CNTFET example)

---

## Examples

See the `example/` directory for complete examples:
- `pip_package_example.py` - Basic usage for all devices
- `sim_inverter.py` - Circuit simulation using CNTFET

---

## Need Help?

- Check existing device implementations in `models/CNTFET/`, `models/NMOS/`, or `models/HFET/`
- Review the `DeviceFactory` and `DeviceWrapper` classes in `semldb.py`
- Open an issue on GitHub for questions
