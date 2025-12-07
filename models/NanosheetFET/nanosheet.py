import os
import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import MODELS


class SimulationConfig:
    # Physical constants
    hbar = 1.0552e-34       # J*s (h/2pi)
    qe = 1.6e-19            # electron charge [C]
    me = 0.19 * 9.11e-31    # electron effective mass [Kg]
    kT = 1.38e-23 * 300 / qe  # thermal energy [eV]
    epso = 8.854e-12        # permittivity of free space
    phi = 0.026             # thermal voltage parameter
    vthm = 0.0259           # thermal voltage


def convert_str_to_float(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = convert_str_to_float(value)
            elif isinstance(value, str):
                try:
                    data[key] = float(value)
                except ValueError:
                    pass
    return data


def parse_voltage_input(v_input):
    """Parse voltage input into numpy array."""
    if v_input is None:
        raise ValueError("Voltage input cannot be None")

    if isinstance(v_input, (int, float)):
        return np.array([float(v_input)])

    if isinstance(v_input, (list, tuple)):
        return np.array(v_input, dtype=float)

    if isinstance(v_input, np.ndarray):
        return v_input.astype(float)

    if isinstance(v_input, dict):
        if {'start', 'end', 'step'}.issubset(v_input.keys()):
            return np.linspace(v_input['start'], v_input['end'], v_input['step'])
        raise ValueError("Dict format must contain 'start', 'end', 'step' keys")

    raise ValueError(f"Unsupported voltage input type: {type(v_input)}")


class BaseExp2Model(nn.Module):
    """Base class for permutation-invariant tower models with helper builders."""

    def __init__(self, embed_size: int = 32):
        super(BaseExp2Model, self).__init__()
        self.embed_size = embed_size

    def _create_type_embeddings(self, num_features: int) -> nn.Embedding:
        return nn.Embedding(num_features, self.embed_size)

    def _create_mlp(self, input_dim: int, output_dim: int, hidden_dim=None, num_layers: int = 2) -> nn.Module:
        if hidden_dim is None:
            hidden_dim = 4 * input_dim

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(nn.Linear(current_dim, output_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                current_dim = hidden_dim

        return nn.Sequential(*layers)


class TwoTowerModel(BaseExp2Model):
    """
    Two-Tower with FiLM fusion (order-free).

    Device tower tokenizes [tox, Lg, epsr] with type embeddings and shared MLP.
    Bias tower processes [Vg, Vd]. Fusion via FiLM. Output head to arbitrary size.
    """

    def __init__(self, embed_size: int = 32, output_size: int = 1, fusion_type: str = 'film', rank: int = 8):
        super(TwoTowerModel, self).__init__(embed_size)

        self.num_device_features = 3
        self.num_bias_features = 2
        self.fusion_type = fusion_type
        self.rank = rank

        # Device tower components
        self.device_type_embeddings = self._create_type_embeddings(self.num_device_features)
        self.device_shared_mlp = self._create_mlp(1 + embed_size, embed_size, hidden_dim=2 * embed_size)

        # Bias tower
        self.bias_mlp = self._create_mlp(self.num_bias_features, embed_size, hidden_dim=4 * embed_size, num_layers=3)

        # Fusion layers
        if fusion_type == 'film':
            self.film_projection = nn.Linear(embed_size, 2 * embed_size)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Output head
        self.output_head = self._create_mlp(embed_size, output_size, num_layers=2)

    def forward(self, x):
        """
        Args:
            x: [batch_size, 5] where features are [tox, Lg, epsr, Vg, Vd]
        """
        batch_size = x.shape[0]

        # Split input into device and bias parameters
        device_params = x[:, :self.num_device_features]  # [tox, Lg, epsr]
        bias_params = x[:, self.num_device_features:]     # [Vg, Vd]

        # Device tower: tokenize each device parameter with type embeddings
        device_tokens = []
        for i in range(self.num_device_features):
            scalar_val = device_params[:, i:i+1]
            type_emb = self.device_type_embeddings(torch.tensor(i, device=x.device)).unsqueeze(0).expand(batch_size, -1)
            token_input = torch.cat([scalar_val, type_emb], dim=1)
            token = self.device_shared_mlp(token_input)
            device_tokens.append(token)

        device_tokens = torch.stack(device_tokens, dim=1)
        h_p = torch.mean(device_tokens, dim=1)
        h_p = F.layer_norm(h_p, (h_p.shape[-1],))

        # Bias tower
        h_v = self.bias_mlp(bias_params)

        # FiLM fusion
        film_params = self.film_projection(h_p)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        h = F.layer_norm(h_v * gamma + beta, (gamma.shape[-1],))

        # Output head
        output = self.output_head(h)
        return output


class TwoTowerNNVSModel(nn.Module):
    """
    Two-Tower Neural Network Virtual Source Model (copied locally for nanosheet).

    Architecture:
    1. Device Tower: Processes device parameters [tox, Lg, epsr] with type embeddings
    2. Bias Tower: Processes bias parameters [Vg, Vd]
    3. Device-only head: Predicts n, v0, vdsat, beta from Device Tower output
    4. Full head: Predicts Q, Vt, vdsat from fused Device+Bias towers
    5. Virtual Source: Combines predictions using physics equations
    """

    def __init__(self, embed_size=32, charge_include=True, device_head_layers=2, full_head_layers=2):
        super(TwoTowerNNVSModel, self).__init__()

        self.embed_size = embed_size
        self.charge_include = charge_include
        self.epso = 8.854e-12
        self.phi = torch.tensor(0.026)

        # Core Two-Tower architecture with FiLM fusion
        self.two_tower = TwoTowerModel(
            embed_size=embed_size,
            output_size=3,  # Used for Q and Vt prediction
            fusion_type='film'
        )

        # Device-only prediction head (n, v0, vdsat, beta from Device Tower)
        self.device_head = self._create_mlp(
            input_size=embed_size,
            output_size=4,  # n, v0, vdsat, beta
            num_layers=device_head_layers,
            hidden_size=2 * embed_size
        )

        # Override the Two-Tower output head for Q, Vt prediction
        self.two_tower.output_head = self._create_mlp(
            input_size=embed_size,  # FiLM fusion output size
            output_size=3,  # Q, Vt, vdsat
            num_layers=full_head_layers,
            hidden_size=2 * embed_size
        )

        self.softplus = nn.Softplus()

    def _create_mlp(self, input_size, output_size, num_layers=2, hidden_size=None):
        """Create MLP with LayerNorm and GELU activation."""
        if hidden_size is None:
            hidden_size = 2 * input_size

        layers = []
        current_size = input_size

        for i in range(num_layers):
            if i == num_layers - 1:  # Last layer
                layers.append(nn.Linear(current_size, output_size))
            else:
                layers.append(nn.Linear(current_size, hidden_size))
                layers.append(nn.LayerNorm(hidden_size))
                layers.append(nn.GELU())
                current_size = hidden_size

        return nn.Sequential(*layers)

    def forward(self, tox, Lg, epsr, Vg, Vd, compute_gradients=False):
        """
        Forward pass through Two-Tower NNVS model.

        Args:
            tox, Lg, epsr: Device parameters [batch_size]
            Vg, Vd: Bias parameters [batch_size]
        """
        batch_size = tox.shape[0]

        if compute_gradients:
            Vg = Vg.requires_grad_(True) if not Vg.requires_grad else Vg
            Vd = Vd.requires_grad_(True) if not Vd.requires_grad else Vd

        # Prepare input for Two-Tower model: [tox, Lg, epsr, Vg, Vd]
        x = torch.stack([
            tox / 10,      # Scale to ~0.1-1 range
            Lg / 100,      # Scale to ~0.01-1 range
            epsr / 100,    # Scale to ~0.1-1 range
            Vg,            # Already in 0-1 range
            Vd             # Already in 0-1 range
        ], dim=1)  # [batch_size, 5]

        # Get intermediate outputs from Two-Tower model
        device_params = x[:, :3]  # [tox, Lg, epsr]
        bias_params = x[:, 3:]    # [Vg, Vd]

        # Device Tower processing (with type embeddings)
        device_tokens = []
        for i in range(3):  # 3 device features
            scalar_val = device_params[:, i:i+1]
            type_emb = self.two_tower.device_type_embeddings(torch.tensor(i, device=x.device)).unsqueeze(0).expand(batch_size, -1)
            token_input = torch.cat([scalar_val, type_emb], dim=1)
            token = self.two_tower.device_shared_mlp(token_input)
            device_tokens.append(token)

        device_tokens_stacked = torch.stack(device_tokens, dim=1)
        h_p = torch.mean(device_tokens_stacked, dim=1)  # Device tower output
        h_p = torch.nn.functional.layer_norm(h_p, (h_p.shape[-1],))

        # Bias Tower processing
        h_v = self.two_tower.bias_mlp(bias_params)

        # Device-only predictions (n, v0, vdsat, beta) from Device Tower
        device_only_output = self.device_head(h_p)
        n_raw = device_only_output[:, 0]
        v0_raw = device_only_output[:, 1]
        beta_raw = device_only_output[:, 3]

        # Apply constraints and transformations
        n = torch.exp(n_raw)  # n should be positive
        v0 = torch.exp(v0_raw)  # v0 should be positive
        _B = 1.4 + 1.1 * torch.sigmoid(beta_raw)

        # Full model predictions (Q, Vt, vdsat) using FiLM fusion
        film_params = self.two_tower.film_projection(h_p)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        h_fused = torch.nn.functional.layer_norm(h_v * gamma + beta, (gamma.shape[-1],))

        full_output = self.two_tower.output_head(h_fused)
        Q = full_output[:, 0] if self.charge_include else torch.zeros_like(Vg)
        Vt = full_output[:, 1]  # Threshold voltage
        vdsat_raw = full_output[:, 2]
        vdsat = 0.01 + 0.39 * torch.sigmoid(vdsat_raw)

        # Inversion capacitance
        Cinv = 1e12 * epsr * self.epso / tox  # [F/m]

        F_softplus = self.softplus((Vg - Vt) / (n * self.phi))

        # Final current calculation (log scale)
        log_Id = (torch.log(Cinv * n * self.phi) +
                  torch.log(F_softplus) +
                  torch.log(v0) +
                  torch.log(Vd) -
                  1/_B * torch.logsumexp(torch.stack([_B*torch.log(vdsat), _B*torch.log(Vd)]), dim=0))

        # Package predicted parameters
        param_dict = {
            'n': n,
            'v_0': v0,
            'v_dsat': vdsat,
            'beta': _B,
            'F': Vt,  # Threshold voltage
            'Q': Q if self.charge_include else torch.zeros_like(Vg)
        }

        Id_pred = torch.exp(log_Id)
        if compute_gradients:
            dI_dVg, dI_dVd = torch.autograd.grad(
                    outputs=Id_pred,
                    inputs=(Vg, Vd),
                    grad_outputs=torch.ones_like(Id_pred),
                    create_graph=True,
                    retain_graph=True
                )
            dI_dVg, dI_dVd = None, None
        else:
            dI_dVg, dI_dVd = None, None

        return log_Id, param_dict, {'dI_dVg': dI_dVg, 'dI_dVd': dI_dVd}


def get_adjusted_simulation_data(db_helper, parameters):
    """Mirror CNTFET Vth shifting logic for nanosheet."""
    vth = parameters.get('V_th', 0.258)

    if vth < 0.158 or vth > 0.408:
        return None, False, None, None, "Threshold voltage must be between 0.158 and 0.408"

    vth_shift = vth - 0.258

    db_query_params = {k: v for k, v in parameters.items() if k != 'V_th'}

    complete_data, exact_match, distance, matched_params = db_helper.get_simulation_data(
        'NanosheetFET', db_query_params
    )

    if not complete_data:
        return None, False, None, None

    vg_values = complete_data.get('simulation_data', {}).get('Vg', [])
    vd_values = complete_data.get('simulation_data', {}).get('Vd', [])
    id_data = complete_data.get('simulation_data', {}).get('Id', [])
    qg_data = complete_data.get('simulation_data', {}).get('Qg', [])

    step = 0.0125
    index_shift = -int(round(vth_shift / step))

    start_vg_index, end_vg_index = 12, 53

    start_idx = start_vg_index + index_shift
    end_idx = end_vg_index + index_shift

    total_points = len(vg_values)
    start_idx = max(0, min(start_idx, total_points - 1))
    end_idx = max(0, min(end_idx, total_points - 1))

    selected_vg = vg_values[start_idx:end_idx]
    selected_id = id_data[start_idx:end_idx]
    selected_qg = qg_data[start_idx:end_idx]

    shifted_vg = [round(vg + vth_shift, 4) for vg in selected_vg]

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


def _load_twotower_model(embed_size=16):
    """Load the pretrained Two-Tower model for nanosheet inference."""
    model = TwoTowerNNVSModel(embed_size=embed_size, charge_include=True)
    model_dir = os.path.dirname(__file__)
    model_path = os.path.join(model_dir, "NanosheetFET_idvd_idvg.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pretrained weights not found at {model_path}. "
            "Place NanosheetFET_idvd_idvg.pth alongside nanosheet.py."
        )

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def run_twotower_sim(parameters, config=None):
    """Run nanosheet simulation using the Two-Tower Hybrid DGFET model."""
    parameters = convert_str_to_float(parameters)
    tox = parameters.get('tox')
    Lg = parameters.get('Lg')
    eps_ox = parameters.get('eps_ox')
    Vd_input = parameters.get('Vd')
    Vg_input = parameters.get('Vg')

    if tox is None or Lg is None or eps_ox is None:
        raise ValueError("Missing device parameters: require tox, Lg, eps_ox.")

    Vd_array = parse_voltage_input(Vd_input)
    Vg_array = parse_voltage_input(Vg_input)

    Vd_tensor = torch.tensor(Vd_array, dtype=torch.float32)
    Vg_tensor = torch.tensor(Vg_array, dtype=torch.float32)
    Vd_mesh, Vg_mesh = torch.meshgrid(Vd_tensor, Vg_tensor, indexing='ij')

    model = _load_twotower_model()

    with torch.no_grad():
        log_Id_pred, param_dict, _ = model(
            torch.full_like(Vg_mesh.flatten(), float(tox)),
            torch.full_like(Vg_mesh.flatten(), float(Lg)),
            torch.full_like(Vg_mesh.flatten(), float(eps_ox)),
            Vg_mesh.flatten(),
            Vd_mesh.flatten()
        )

    Id_pred = torch.exp(log_Id_pred).cpu().numpy().reshape(Vd_mesh.shape)
    Qg_pred = param_dict.get('Q', torch.zeros_like(Vg_mesh.flatten()))
    Qg_pred = Qg_pred.cpu().numpy().reshape(Vd_mesh.shape)

    return_body = {
        'simulation_data': {
            'Vg': Vg_array.tolist(),
            'Vd': Vd_array.tolist(),
            'Id': Id_pred.T.tolist(),
            'Qg': Qg_pred.T.tolist()
        },
        'device_params': parameters
    }

    return return_body


@MODELS.register()
class NanosheetFET:
    simulation_func = staticmethod(lambda parameters: run_twotower_sim(parameters, SimulationConfig))
    device_params = ['tox', 'Lg', 'eps_ox', 'V_th']
    voltage_params = ['Vg', 'Vd']
    postprocess = get_adjusted_simulation_data

if __name__ == "__main__":
    parameters = {
        'tox': 2.0,
        'Lg': 10.0,
        'eps_ox': 4.0,
        'V_th': 0.25,  # optional; remove or change as needed
        'Vd': {'start': 0.001, 'end': 0.501, 'step': 41},
        'Vg': {'start': 0.0, 'end': 0.50, 'step': 41},
    }

    result = run_twotower_sim(parameters)
    # Uncomment to inspect output shapes when running this module directly
#print("Id shape:", np.array(result['simulation_data']['Id']).shape)
#print("Qg shape:", np.array(result['simulation_data']['Qg']).shape)