import numpy as np
import matplotlib.pyplot as plt
import time
import requests
import json
from scipy.optimize import fsolve, minimize
from scipy.interpolate import Rbf
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 22})

### API configuration
server_url = 'https://semldb.rc.ufl.edu:443/'
run_sim_url = f'{server_url}/run_simulation'

### Device parameters
device_params = {
    'tox': 1.9,
    'eps_ox': 21.0,
    'Lg': 12,
    'd_cnt': 1.02,
    'V_th': 0.258,
    'sca_flag': 0
}

def generate_device_characteristics(grid_size=81, voltage_range=(0, 0.5), save_to_file=True):
    print(f"generating {grid_size}x{grid_size}...")
    start_time = time.time()
    
    params = {
        "device_type": "CNTFET",
        'sim_type': 'rnn',
        "parameters": {
            **device_params,
            'Vg': {
                'start': voltage_range[0],
                'end': voltage_range[1],
                'step': grid_size
            },
            'Vd': {
                'start': voltage_range[0],
                'end': voltage_range[1],
                'step': grid_size
            },
        }
    }
    
    print(f"Vg: {voltage_range[0]:.3f}V to {voltage_range[1]:.3f}V, #steps: {grid_size}V")
    print(f"Vd: {voltage_range[0]:.3f}V to {voltage_range[1]:.3f}V, #steps: {grid_size}V")
    
    try:
        response = requests.post(
            run_sim_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(params),
            timeout=300
        )
        
        if response.status_code == 200:
            complete_data = response.json().get('data', {})
            simulation_data = complete_data.get('simulation_data', {})
            
            Vg_data = np.array(simulation_data.get('Vg', []))
            Vd_data = np.array(simulation_data.get('Vd', []))
            Id_data = np.array(simulation_data.get('Id', []))
            
            print(f"shape of data responsed by API: Vg{Vg_data.shape}, Vd{Vd_data.shape}, Id{Id_data.shape}")
            
            if len(Id_data.shape) == 2 and Id_data.shape == (grid_size, grid_size):
                voltage_points = np.linspace(voltage_range[0], voltage_range[1], grid_size)
                Vg_grid, Vd_grid = np.meshgrid(voltage_points, voltage_points)
                Id_grid = Id_data
            else:
                print(f"Warning: API returned data size mismatch, expected {grid_size*grid_size}, actual {Id_data.size}")
                raise

        else:
            print(f'Batch API request failed: Status code {response.status_code}')
            print("Falling back to point-by-point request method...")
            return

    except requests.RequestException as e:
        print(f'Batch API request error: {str(e)}')
        print("Falling back to point-by-point request method...")
        return
    except Exception as e:
        print(f'Batch processing error: {str(e)}')
        print("Falling back to point-by-point request method...")
        return

    generation_time = time.time() - start_time
    print(f"\nBatch data generation completed, time elapsed: {generation_time:.2f} seconds")
    print(f"Actual data shape: Vg{Vg_grid.shape}, Vd{Vd_grid.shape}, Id{Id_grid.shape}")

    # Save data
    if save_to_file:
        filename = f'device_data_{grid_size}x{grid_size}.npz'
        np.savez(filename, Vg=Vg_grid, Vd=Vd_grid, Id=Id_grid, voltage_range=voltage_range)
        print(f"Data saved to: {filename}")
    
    return Vg_grid, Vd_grid, Id_grid

def load_device_data(filename):
    try:
        data = np.load(filename)
        return data['Vg'], data['Vd'], data['Id']
    except FileNotFoundError:
        print(f"File {filename} does not exist, need to regenerate data")
        return None, None, None

# ================================================================
# Step 2: Create RBF interpolation model
# ================================================================

class DeviceRBFModel:
    """Device model based on RBF interpolation"""
    
    def __init__(self, Vg_grid, Vd_grid, Id_grid, rbf_function='multiquadric'):
        """
        
        Args:
            Vg_grid, Vd_grid, Id_grid: Grid data
            rbf_function: RBF function type
        """
        print(f"Creating RBF interpolation model (function: {rbf_function})...")
        
        self.Vg_flat = Vg_grid.flatten()
        self.Vd_flat = Vd_grid.flatten()
        self.Id_flat = Id_grid.flatten()
        
        self.rbf_model = Rbf(self.Vg_flat, self.Vd_flat, self.Id_flat, 
                            function=rbf_function, smooth=1e-10)
        
        self.rbf_function = rbf_function
        self.call_count = 0
    
    def predict_current(self, Vg, Vd):
        self.call_count += 1
        return float(self.rbf_model(Vg, Vd))
    
    def batch_predict(self, Vg_array, Vd_array):
        self.call_count += len(Vg_array)
        return self.rbf_model(Vg_array, Vd_array)

# ================================================================
# Step 3: Device model wrapper
# ================================================================

device_rbf_model = None

def initialize_rbf_model(grid_size=81, voltage_range=(0, 0.5), 
                        rbf_function='multiquadric', 
                        force_regenerate=False):
    """Initialize RBF model"""
    global device_rbf_model
    
    filename = f'device_data_{grid_size}x{grid_size}.npz'
    
    if not force_regenerate:
        Vg_grid, Vd_grid, Id_grid = load_device_data(filename)
    else:
        Vg_grid, Vd_grid, Id_grid = None, None, None
    
    if Vg_grid is None:
        Vg_grid, Vd_grid, Id_grid = generate_device_characteristics(
            grid_size, voltage_range)
    
    device_rbf_model = DeviceRBFModel(Vd_grid, Vg_grid, Id_grid, rbf_function)
    
    return device_rbf_model

def nsifet(Vg, Vd):
    # if Vd < 0 or Vg < 0:
    #     return 0.0
    return device_rbf_model.predict_current(Vg, Vd)

def psifet(Vg, Vd):
    return -nsifet(-Vg, -Vd)

# ================================================================
# Step 4: Inverter simulation (using RBF model)
# ================================================================

def solve_with_bounds(balance_func, initial_guess, args=(), bounds=(0.0, 0.5)):
    def objective(x):
        residuals = balance_func(x, *args)
        return np.sum(residuals**2)
    
    bounds_list = [bounds] * len(initial_guess)
    
    return fsolve(balance_func, initial_guess, args=args)

def inverter_balance(Vnodes, Vin, Rp=1e5):
    Vdd = 0.4
    r = np.zeros_like(Vnodes)
    
    # KCL
    r[0] = Vnodes[0]/Rp - nsifet(Vin-Vnodes[0], Vnodes[1]-Vnodes[0])
    r[1] = nsifet(Vin-Vnodes[0], Vnodes[1]-Vnodes[0]) - (Vnodes[2]-Vnodes[1])/(2*Rp)
    r[2] = (Vnodes[2]-Vnodes[1])/(2*Rp) + psifet(Vin-Vnodes[3], Vnodes[2]-Vnodes[3])
    r[3] = -psifet(Vin-Vnodes[3], Vnodes[2]-Vnodes[3]) + (Vnodes[3]-Vdd)/Rp
    
    return r

def simulate_inverter_rbf(Vdd=0.4, num_points=81, Rp=1e5):
    start_time = time.time()
    
    Vin_values = np.linspace(0, Vdd, num_points)
    Vout_values = []
    
    device_rbf_model.call_count = 0 
    
    for i, Vin in enumerate(Vin_values):
        print(f"Progress: {i+1}/{num_points} (Vin = {Vin:.3f}V)", end='\r')
        
        try:
            initial_guess = np.linspace(0.05, 0.45, 4)
            Vnodes = fsolve(inverter_balance, initial_guess, 
                                     args=(Vin, Rp))
            
            Vout = (Vnodes[1] + Vnodes[2]) / 2
            Vout_values.append(Vout)
            
        except Exception as e:
            print(f"\nSolver error at Vin={Vin:.3f}V: {str(e)}")
            if len(Vout_values) > 0:
                Vout_values.append(Vout_values[-1])
            else:
                Vout_values.append(Vdd - Vin)
    
    simulation_time = time.time() - start_time
    print(f"\nSimulation completed, time elapsed: {simulation_time:.2f} seconds")
    print(f"RBF model call count: {device_rbf_model.call_count}")
    
    return Vin_values, np.array(Vout_values), simulation_time

def main():
    print("="*60)
    print("CNTFET Inverter Simulation Based on RBF Interpolation")
    print("="*60)
    
    print("\nStep 1: Initializing RBF model...")
    
    grid_size = 81
    rbf_function = 'cubic'  # 'cubic', 'linear'
    
    model = initialize_rbf_model(
        grid_size=grid_size, 
        rbf_function=rbf_function, 
        force_regenerate=True 
    )
    
    print("\nStep 2: Simulating inverter transfer characteristics...")
    Vin_values, Vout_values, sim_time = simulate_inverter_rbf(
        Vdd=0.4, 
        num_points=81,
        Rp=1e4
    )
    
    print("\nStep 3: Plotting results...")
    plt.figure(figsize=(5.4, 4.8), dpi=300)
    plt.plot(Vin_values, Vout_values, 'ro-', linewidth=2, markersize=6, 
             label=f'Simulation Result')
    
    plt.xlabel(r'$V_{in}$ [V]')
    plt.ylabel(r'$V_{out}$ [V]')
    plt.title('CNTFET Inverter Simulation')
    # plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    plt.savefig('sim_inverter.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()