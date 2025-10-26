"""
Example usage of semldb as a pip package
Install: pip install -e .
"""

import numpy as np
import matplotlib.pyplot as plt
from semldb import DeviceFactory

# Example 1: CNTFET simulation
print("="*60)
print("CNTFET Simulation Example")
print("="*60)

cntfet = DeviceFactory.create(
    'CNTFET',
    tox=2.0,
    Lg=12.0,
    eps_ox=21.0,
    d_cnt=1.02,
    V_th=0.258,
    sca_flag=0
)

result = cntfet.simulate(
    Vg={'start': 0, 'end': 0.5, 'step': 21},
    Vd={'start': 0, 'end': 0.5, 'step': 11}
)

Vg = np.array(result['simulation_data']['Vg'])
Vd = np.array(result['simulation_data']['Vd'])
Id = np.array(result['simulation_data']['Id'])

print(f"Vg shape: {Vg.shape}")
print(f"Vd shape: {Vd.shape}")
print(f"Id shape: {Id.shape}")
print(f"Max current: {np.max(Id):.6e} A")

# Example 2: NMOS simulation
print("\n" + "="*60)
print("NMOS Simulation Example")
print("="*60)

nmos = DeviceFactory.create(
    'NMOS',
    Lg=20.0,
    THF=4.0,
    XjSD=10.0
)

result = nmos.simulate()

IdVd = np.array(result['simulation_data']['Id_Vd']['Id'])
IdVg = np.array(result['simulation_data']['Id_Vg']['Id'])

print(f"IdVd shape: {IdVd.shape}")
print(f"IdVg shape: {IdVg.shape}")
print(f"Max IdVd: {np.max(IdVd):.6e} A")

# Example 3: HFET simulation
print("\n" + "="*60)
print("HFET Simulation Example")
print("="*60)

hfet = DeviceFactory.create(
    'HFET',
    Lsg=0.5,
    Lgd=1.0,
    Lg=0.5,
    hpas=5.0,
    hAlGaN=20.0,
    hch=15.0,
    hg=100.0
)

result = hfet.simulate()

IdVd = np.array(result['simulation_data']['Id_Vd']['Id'])
IdVg = np.array(result['simulation_data']['Id_Vg']['Id'])

print(f"IdVd shape: {IdVd.shape}")
print(f"IdVg shape: {IdVg.shape}")
print(f"Max IdVd: {np.max(IdVd):.6e} A")

print("\n" + "="*60)
print("All simulations completed successfully!")
print("="*60)
