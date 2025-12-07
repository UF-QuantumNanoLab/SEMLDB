#!/usr/bin/env python3
"""Test script to verify the registry implementation works correctly"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MODEL_CONFIG, MODELS

def test_registry():
    print("Testing backend registry implementation...")
    
    # Test that models are registered
    print(f"Registered models: {list(MODELS.get_all_reg())}")
    
    # Test that MODEL_CONFIG is created correctly
    print(f"MODEL_CONFIG keys: {list(MODEL_CONFIG.keys())}")
    
    for device_name, config in MODEL_CONFIG.items():
        print(f"\n{device_name} configuration:")
        # print(f"  - simulation_func keys: {list(config['simulation_func'].keys())}")
        print(f"  - simulation_func: {config['simulation_func']}")
        print(f"  - device_params: {config['device_params']}")
        print(f"  - postprocess: {config['postprocess']}")
    
    # Test accessing models through registry
    for device_name in MODELS.get_all_reg():
        model_class = MODELS.get(device_name)
        print(f"\n{device_name} class attributes:")
        print(f"  - simulation_func: {model_class.simulation_func}")
        print(f"  - device_params: {model_class.device_params}")
        print(f"  - postprocess: {model_class.postprocess}")
    
    print("\n✓ Registry implementation test completed successfully!")

if __name__ == "__main__":
    test_registry()