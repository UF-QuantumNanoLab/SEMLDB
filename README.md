# semldb - Semiconductor Machine Learning Models and Database

Semiconductor device simulation using machine learning models. Supports multiple device types including carbon nanotube field-effect transistors (CNTFET), Silicon MOSFET, and heterojunction FET (HFET).

## Quick Start
## Table of Contents
- [Run nanoHUB Simulation](#run-nanohub-simulation)
- [API Usage](#api-usage)
- [Installation and Usage](#installation-and-usage)

---

## Run nanoHUB Simulation

You can directly run the **semldb** tool online via nanoHUB without installation:  
🔗 [https://nanohub.org/resources/semldb/](https://nanohub.org/resources/semldb/)

---
## API Usage

semldb provides a RESTful API for running device simulations remotely. This is useful for web applications, cloud-based workflows, or when you want to access the simulation engine without installing Python locally.

### API Endpoint

```
POST https://semldb.rc.ufl.edu:443/run_simulation
```

### Request Format

Send a POST request with JSON payload:

```python
import requests
import json
import numpy as np

# API configuration
server_url = 'https://semldb.rc.ufl.edu:443/'
run_sim_url = f'{server_url}/run_simulation'

# Device parameters
params = {
    "device_type": "CNTFET",
    "parameters": {
        'tox': 1.9,
        'eps_ox': 21.0,
        'Lg': 12,
        'd_cnt': 1.02,
        'V_th': 0.258,
        'sca_flag': 0,
        'Vg': {
            'start': 0,
            'end': 0.5,
            'step': 81
        },
        'Vd': {
            'start': 0,
            'end': 0.5,
            'step': 81
        }
    }
}

# Send request
response = requests.post(
    run_sim_url,
    headers={"Content-Type": "application/json"},
    data=json.dumps(params),
    timeout=300
)

# Parse response
if response.status_code == 200:
    complete_data = response.json().get('data', {})
    simulation_data = complete_data.get('simulation_data', {})

    Vg_data = np.array(simulation_data.get('Vg', []))
    Vd_data = np.array(simulation_data.get('Vd', []))
    Id_data = np.array(simulation_data.get('Id', []))

    print(f"Data shape: Vg{Vg_data.shape}, Vd{Vd_data.shape}, Id{Id_data.shape}")
else:
    print(f'Request failed: Status code {response.status_code}')
```

### Response Format

The API returns JSON with the following structure:

```json
{
    "data": {
        "simulation_data": {
            "Vg": [...],
            "Vd": [...],
            "Id": [...]
        },
        "device_params": {
            "tox": 1.9,
            "eps_ox": 21.0,
            ...
        }
    }
}
```

### Retrieving High-Precision Simulation Data

The API also provides access to high-precision simulation data (e.g., NEGF or Synopsys results) stored in the database:

```
GET https://semldb.rc.ufl.edu:443/simulation_data
```

**Example:**

```python
import requests

# Device parameters
params = {
    'device': 'CNTFET',
    'tox': 2.0,
    'Lg': 12,
    'eps_ox': 25,
    'V_th': 0.358,
    'sca_flag': 1
}

# Send GET request
response = requests.get(
    'https://semldb.rc.ufl.edu:443/simulation_data',
    params=params,
    timeout=30
)

if response.status_code == 200:
    data = response.json()
    simulation_data = data.get('simulation_data', {})

    # The API will return exact match or nearest neighbor from database
    exact_match = data.get('exact_match', False)
    distance = data.get('distance', None)

    if exact_match:
        print("Found exact match in database")
    else:
        print(f"Nearest match with distance: {distance}")
        print(f"Matched params: {data.get('matched_params', {})}")

    # Access high-precision data
    Vg = simulation_data.get('Vg', [])
    Vd = simulation_data.get('Vd', [])
    Id = simulation_data.get('Id', [])
else:
    print(f'Request failed: Status code {response.status_code}')
```

**Note:** This endpoint queries the database for pre-computed high-precision simulation results. If no exact match is found, it returns the nearest neighbor based on parameter distance.

### Complete Example: Inverter Circuit Simulation

See [example/sim_inverter.py](example/sim_inverter.py) for a complete circuit simulation example that:
1. Fetches device characteristics using the API (81×81 grid)
2. Creates RBF interpolation model for fast device evaluation
3. Simulates a CNTFET inverter circuit using the model
4. Plots the transfer characteristics

```bash
python example/sim_inverter.py
```

This example demonstrates how to use the API for circuit-level simulation where the device model is called thousands of times during iterative solving.

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

#### 3. Initialize and Use Device

```python
from semldb import DeviceFactory

# Initialize device instance
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

## Examples

See the `example/` directory for complete examples:
- `pip_package_example.py` - Basic usage for all devices
- `sim_inverter.py` - Circuit simulation using CNTFET

---

## Need Help?

- Check existing device implementations in `models/CNTFET/`, `models/NMOS/`, or `models/HFET/`
- Review the `DeviceFactory` and `DeviceWrapper` classes in `semldb.py`
- Open an issue on GitHub for questions
