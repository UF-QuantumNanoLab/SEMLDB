import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
import os

import torch
import torch.nn as nn

from .. import MODELS


class BiGRU(nn.Module):
    def __init__(self, embed_size, output_size, num_layers=2):
        super(BiGRU, self).__init__()

        self.embedding = nn.Conv1d(1, embed_size, kernel_size=1, stride=1, padding=0)
        # Define the BiGRU layer
        self.bigru = nn.GRU(input_size=embed_size,
                            hidden_size=embed_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        # Define the output layer
        self.pred_head = []
        # for i in range(output_size):
        #     self.pred_head.append(nn.Linear(2 * embed_size, 1))
        self.linear = nn.Linear(2 * embed_size, output_size)  # 2*hidden_size because it's bidirectional
        nn.init.kaiming_normal_(self.linear.weight)
        for name, param in self.bigru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        # Reshape input to have sequence length
        x = self.embedding(x.unsqueeze(1)).permute(0, 2, 1)
        # BiGRU layers
        bigru_out, _ = self.bigru(x)
        # Linear layer
        # output = []
        # for pred_head in self.pred_head:
        #     output.append(pred_head(bigru_out[:, -1, :]))
        output = self.linear(bigru_out[:, -1, :])
        return output

### physical constants:
class SimulationConfig:
    # Physical constants
    T = 300
    q0 = 1.6e-19
    kB = 1.38e-23
    kBT = (kB * T / q0)
    hbar = 1.055e-34
    vF = 9.3e5  # m/s
    eps0 = 8.854e-12

    # Normalization units
    Iunit = 1e-6  # A
    Qunit = q0    # C
    tunit = Qunit / Iunit  # s
    Cpar = 1e-18 / Qunit   # F

    epsr = 20  # CNTFET parameters
    dcnt = 1e-9

def convert_str_to_float(data):
    # If the data is a dictionary, iterate through its items
    if isinstance(data, dict):
        for key, value in data.items():
            # Recursively call the function if the value is another dictionary
            if isinstance(value, dict):
                data[key] = convert_str_to_float(value)
            # Convert to float if the value is a string and can be converted
            elif isinstance(value, str):
                try:
                    data[key] = float(value)
                except ValueError:
                    pass  # Ignore values that cannot be converted to float
    return data

def get_adjusted_simulation_data(db_helper, parameters):
    """
    Fetch simulation data with adjusted threshold voltage.
    
    Args:
        db_helper: Database helper instance
        parameters: Device parameters including threshold voltage
        
    Returns:
        Adjusted simulation data
    """
    # Extract the threshold voltage (Vth) from parameters
    vth = parameters.get('V_th', 0.258)  # Default to 0.258 if not provided
    
    if vth < 0.158 or vth > 0.408:
        return None, False, None, None, "Threshold voltage must be between 0.158 and 0.408"
    
    # Calculate the voltage shift compared to the base threshold (0.258)
    vth_shift = vth - 0.258
    
    # Create a copy of parameters without the vth for database query
    db_query_params = {k: v for k, v in parameters.items() if k != 'V_th'}

    complete_data, exact_match, distance, matched_params = db_helper.get_simulation_data(
        'CNTFET', db_query_params
    )
    
    if not complete_data:
        return None, False, None, None
    
    vg_values = complete_data.get('simulation_data', {}).get('Vg', [])
    vd_values = complete_data.get('simulation_data', {}).get('Vd', [])
    id_data = complete_data.get('simulation_data', {}).get('Id', [])
    qg_data = complete_data.get('simulation_data', {}).get('Qg', [])

    step = 0.0125
    
    # Calculate the indices for the equivalent Vg range [0-vth_shift, 0.5-vth_shift]
    index_shift = - int(round(vth_shift/step))

    start_vg_index, end_vg_index = 12, 53
    
    # Calculate indices (round to nearest index)
    start_idx = start_vg_index + index_shift
    end_idx = end_vg_index + index_shift

    total_points = len(vg_values)
    start_idx = max(0, min(start_idx, total_points - 1))
    end_idx = max(0, min(end_idx, total_points - 1))

    selected_vg = vg_values[start_idx:end_idx]
    selected_id = id_data[start_idx:end_idx]
    selected_qg = qg_data[start_idx:end_idx]
    
    # Shift the Vg values back to [0, 0.5] range
    shifted_vg = [round(vg + vth_shift, 4) for vg in selected_vg]
    
    # Construct the adjusted data
    adjusted_data = {
        'simulation_data': {
            'Vg': shifted_vg,
            'Vd': vd_values,
            'Id': selected_id,
            'Qg': selected_qg
        },
        'device_params': complete_data.get('device_params', {})
    }
    
    adjusted_data['device_params']['V_th'] = vth
    
    return adjusted_data, exact_match, distance, matched_params

def run_rnn_sim(parameters, config):
    vth = parameters.get('V_th', 0.258)  # Default to 0.258 if not provided
    scale_factor = 4.049253845214844
    parameters = convert_str_to_float(parameters)

    model = BiGRU(embed_size=32, output_size=2, num_layers=3)
    d_cnt, sca_flag = parameters.get('d_cnt', 1.02), parameters.get('sca_flag', 0)
    model.load_state_dict(torch.load(os.path.join(f'./models/CNTFET/pretrained_rnn_dcnt{d_cnt}_sca{sca_flag}.pth'), map_location=torch.device('cpu')))
    model.eval()
    # test_idx = test_dataset.valid_indices[device_indices[device_idx]]
    # test_tox = test_dataset.toxdata[test_idx].item()
    # test_Lch = test_dataset.Lchdata[test_idx].item()
    # test_eps_ox = test_dataset.epsoxdata[test_idx].item()
    
    # print(f"Plotting for device: tox={test_tox}, Lch={test_Lch}, eps_ox={test_eps_ox}")

    # max_current = test_dataset.Iddata.max().item()
    # scale_factor = max_current / 10.0

    # with torch.no_grad():
    #     output = model(torch.stack([
    #             torch.full_like(Vg_mesh.flatten(), test_tox).to(device),
    #             torch.full_like(Vg_mesh.flatten(), test_Lch).to(device),
    #             torch.full_like(Vg_mesh.flatten(), test_eps_ox).to(device),
    #             Vg_mesh.flatten().to(device),
    #             Vd_mesh.flatten().to(device)
    #         ], dim=1))
    #     Id_ratio_pred = output[:, 0]
    #     Qg_pred = output[:, 1]
    tox, Lg, eps_ox, Vd, Vg = parameters.get('tox'), parameters.get('Lg'), parameters.get('eps_ox'), parameters.get('Vd'), parameters.get('Vg')

    if not tox or not Lg or not eps_ox:
        print("Missing parameters for device.")
        return None
    
    Vdlist = torch.Tensor(np.round(np.linspace(Vd['start'], Vd['end'], Vd['step']), 4))
    Vglist = torch.Tensor(np.round(np.linspace(Vg['start'], Vg['end'], Vg['step']), 4))
    Vg_shift_list = Vglist - (vth - 0.258)

    Vd_mesh, Vg_mesh = torch.meshgrid(Vdlist, Vg_shift_list, indexing='ij')

    with torch.no_grad():
        output = model(torch.stack([
                torch.full_like(Vg_mesh.flatten(), tox),
                torch.full_like(Vg_mesh.flatten(), Lg),
                torch.full_like(Vg_mesh.flatten(), eps_ox),
                Vg_mesh.flatten(),
                Vd_mesh.flatten()
            ], dim=1))
        Id_ratio_pred = output[:, 0]
        Qg_pred = output[:, 1]
    
    Id_pred = (torch.exp(Id_ratio_pred).cpu() * torch.log(Vd_mesh.flatten()*10+1) * scale_factor).numpy().reshape(Vd_mesh.shape)*1e-6
    Qg_pred = Qg_pred.cpu().numpy().reshape(Vd_mesh.shape)*1e-18

    return_body = {
        'simulation_data': {
            'Vg': Vglist.numpy().tolist(),
            'Vd': Vdlist.numpy().tolist(),
            'Id': Id_pred.T.tolist(),
            'Qg': Qg_pred.T.tolist()
        },
        'device_params': parameters
    }
    
    return return_body

# def run_rnn_sim(parameters, config):
#     parameters = convert_str_to_float(parameters)
#     tox, Lch, Vd, Vg = parameters.get('tox'), parameters.get('Lch'), parameters.get('Vd'), parameters.get('Vg')

#     if not tox or not Lch:
#         print("Missing parameters for device.")
#         return None

#     mid_params = np.load('./models/CNTFET/mid_params.npy')
#     device_mask = (np.isclose(mid_params['tox'], tox, rtol = 1e-7)) & (np.isclose(mid_params['Lch'], Lch, rtol = 1e-7))
#     # print(f"params: {synopsys_data}")
#     device = mid_params[device_mask]
#     if len(device) == 0:
#         print(f"Given parameters not support yet.")
#         return None
    
#     alpha, Vfb = device[0]['alpha'], device[0]['Vfb']

#     Cins=2*np.pi*config.epsr*config.eps0/np.log((tox * 1e-9 + config.dcnt / 2) / (config.dcnt / 2))

#     Vdlist = np.round(np.linspace(Vd['start'], Vd['end'], Vd['step']), 4)
#     Vglist = np.round(np.linspace(Vg['start'], Vg['end'], Vg['step']), 4)

#     Vdtest, Vgtest = np.meshgrid(Vdlist, Vglist)
#     toxtest, Lchntest = np.ones_like(Vgtest)*tox, np.ones_like(Vgtest)*Lch

#     Idbal, _, _, _ = balcntsweep(Cins, config.dcnt, alpha, Vfb=Vfb, Vgv=Vglist, Vdv=Vdlist, Lch=Lch*1e-9, config=config)
#     Idbal_test = np.array(Idbal).reshape((Vg['step'], Vd['step']))
#     x_test = np.vstack((Vgtest.ravel(), Vdtest.ravel(), Lchntest.ravel(), toxtest.ravel())).T

#     modelI = BiGRU(8, output_size=1)
#     modelI.load_state_dict(torch.load(f'./models/CNTFET/pretrained.pth', map_location=torch.device('cpu')))
#     modelI.eval()
#     x_test_tensor = torch.FloatTensor(x_test)
#     Id = modelI(x_test_tensor).detach().numpy().reshape((Vg['step'], Vd['step']))
#     # print(Idbal_test * Id)

#     df = pd.DataFrame(
#         Idbal_test * Id,  # 2D array of current values
#         index=Vglist,  # Drain voltage values as row index
#         columns=Vdlist  # Gate voltage values as column headers
#     )
#     return df

def Ekcnt(dcnt, config):
    ## dcnt: in m, diameter of CNT
    k0=2/(3*dcnt)
    k=k0*np.linspace(0,5,101)
    E=config.hbar*config.vF/config.q0*(np.sqrt(k**2+k0**2)-k0)
    vel=config.vF*k/np.sqrt(k**2+k0**2)  # infs to avoid insgularity when k=0
    return k, E, vel

def cntq(Vd,U,dcnt,config):
    Ef1=0
    Ef2=-Vd
    gv=2  # valley degeneracy
    gs=2  # spin degenerarcy
    k, Ek, vel=Ekcnt(dcnt,config)
    ne_=1/(2*np.pi)*np.trapz(1/(1+np.exp((Ek+U-Ef1)/config.kBT)),k)+\
    1/(2*np.pi)*np.trapz(1/(1+np.exp((Ek+U-Ef2)/config.kBT)),k)
    Ie_=1/(2*np.pi)*np.trapz(vel/(1+np.exp((Ek+U-Ef1)/config.kBT)),k)-\
    1/(2*np.pi)*np.trapz(vel/(1+np.exp((Ek+U-Ef2)/config.kBT)),k)
    ne=gv*gs*ne_   # in /m, electron density
    Ie=config.q0*gv*gs*Ie_
    return ne, Ie

def fu(U, alphag, alphad, Vg, Vd,Ctot,dcnt,config):
    ne,_=cntq(Vd,U,dcnt,config)
    y=U+(alphag*Vg+alphad*Vd)-config.q0*ne/Ctot
    return y

def cntfetbal(Cins, dcnt=1e-9, alphag=1.0, alphad=0.0, Vd=0.1, Vg=0.1, config=None):   # CNTFET model
    Ctot=Cins/alphag  # total capacitance
    U=fsolve(fu,0.0,args=(alphag,alphad,Vg,Vd,Ctot,dcnt,config))
    return U

def balcntsweep(Cins, dcnt, alpha, Vfb,Vgv, Vdv,Lch=12e-9, draw=False, config=None):
    ### calculate CNTFET I-V characteristics
    ## Lch: in m, channel length
    Idbal=np.zeros((len(Vgv),len(Vdv)))
    Qbal=np.zeros((len(Vgv),len(Vdv)))
    for iig,Vgb in enumerate(Vgv):
        for iid, Vdb in enumerate(Vdv):
            U=cntfetbal(Cins=Cins, dcnt=dcnt, alphag=alpha, alphad=(1-alpha)/2, Vd=Vdb,Vg=Vgb-Vfb, config=config)
            Qbal[iig,iid],Idbal[iig,iid]=cntq(Vdb,U,dcnt,config)
    
    return Idbal, None, None, None


@MODELS.register()
class CNTFET:
    simulation_func = lambda parameters: run_rnn_sim(parameters, SimulationConfig)
    device_params = ['tox', 'Lg', 'eps_ox', 'd_cnt', 'V_th', 'sca_flag']
    postprocess = get_adjusted_simulation_data


if __name__ == "__main__":
    parameters = {'tox': 2, 'Lch': 10, 'Vd': {'start': 0., 'end': 0.5, 'step': 11}, 'Vg': {'start': 0., 'end': 0.5, 'step': 21}}

    Vd, Vg = parameters.get('Vd'), parameters.get('Vg')
    Vdlist = np.linspace(Vd['start'], Vd['end'], Vd['step'])
    Vglist = np.linspace(Vg['start'], Vg['end'], Vg['step'])

    config = SimulationConfig()
    Id = run_rnn_sim(parameters, config)
    # print(Id)
    # plt.figure(figsize=(5.8, 4.8), dpi=300)
    # plt.plot(Vdlist, Id.T, 'ro', label='device data')
    # plt.savefig('test.png', dpi=300, bbox_inches='tight')

