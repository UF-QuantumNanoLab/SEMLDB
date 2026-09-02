"""
2DFET backend model for SEMLDB.

NumPy-only inference core for the 2DFET Two-Tower FiLM surrogate
(ballistic NEGF-Poisson training data). Reproduces the PyTorch model's
forward pass exactly (validated to ~1e-6), so no torch runtime is needed
for this device.

Device parameters (as exposed to the API / frontend):
    tox        gate oxide thickness [nm]   (trained domain: 1 - 3 nm)
    Lg         gate length [nm]            (trained domain: 6 - 30 nm)
    eps_ox     oxide dielectric constant   (trained domain: 4 - 25)
    material   MoS2 / MoSe2                -> meff 0.5 / 0.6 [m0]
    transport  fixed to ballistic          -> D = 0 [eV^2]
    V_th       threshold voltage [V]       (reference 0.20, range 0.10 - 0.35)

Voltage sweeps:
    Vg [V]  (trained domain: -0.15 - 0.6)
    Vd [V]  (trained domain: 0.001 - 0.501)

Outputs (dataset units):
    Id [A/m] (numerically equal to uA/um), shape [len(Vg), len(Vd)]
    Qg [C/m], shape [len(Vg), len(Vd)]
"""
import math
import os
import pickle

import numpy as np
import torch

from .. import MODELS

_MODEL_DIR = os.path.dirname(__file__)
_PTH_PATH = os.path.join(_MODEL_DIR, "TwoDFET.pth")
_SCALERS_PATH = os.path.join(_MODEL_DIR, "TwoDFET.pkl")

_LAYERNORM_EPS = 1e-5


class _PickleState:
    """State-only replacement for training classes referenced by TwoDFET.pkl."""


class _ScalerUnpickler(pickle.Unpickler):
    """Load only the known scaler artifact types without training-code imports."""

    _STATE_CLASSES = {
        ("sklearn.preprocessing._data", "StandardScaler"),
        ("two_tower_dataset", "TargetTransformer"),
        ("two_tower_dataset", "TargetTransformConfig"),
    }
    _NUMPY_GLOBALS = {
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
    }

    def find_class(self, module, name):
        if (module, name) in self._STATE_CLASSES:
            return _PickleState
        if (module, name) in self._NUMPY_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            "Unsupported global in 2DFET scaler artifact: %s.%s" % (module, name)
        )


def _load_scaler_bundle(path):
    with open(path, "rb") as handle:
        bundle = _ScalerUnpickler(handle).load()

    required = {
        "device_scaler", "bias_scaler", "target_scaler",
        "device_feature_fields", "bias_feature_fields", "target_fields",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError("2DFET scaler artifact is missing: %s" % sorted(missing))
    return bundle

# Vectorised exact error function (stdlib math.erf) -> matches torch's exact GELU.
_erf = np.vectorize(math.erf, otypes=[np.float64])


def _linear(x, w, b):
    # torch Linear stores weight as (out, in); y = x @ w.T + b
    return x @ w.T + b


def _gelu(x):
    # Exact GELU (nn.GELU default, approximate='none').
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _layer_norm(x, eps, w=None, b=None):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # biased (population) variance, like torch
    y = (x - mean) / np.sqrt(var + eps)
    if w is not None:
        y = y * w + b
    return y


class Surrogate:
    """Two-Tower FiLM surrogate loaded from the trained PyTorch checkpoint."""

    def __init__(self, pth_path=_PTH_PATH, scalers_path=_SCALERS_PATH):
        checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)

        self.w = {}
        for key, tensor in state.items():
            prefix = "backbone."
            clean_key = key[len(prefix):] if key.startswith(prefix) else key
            self.w[clean_key] = tensor.detach().cpu().numpy().astype(np.float64)

        config = checkpoint.get("config", {})
        fusion_type = config.get("fusion_type", "film")
        if fusion_type != "film":
            raise ValueError("2DFET checkpoint must use FiLM fusion, got %r" % fusion_type)

        scaler_bundle = _load_scaler_bundle(scalers_path)
        self.scaler = {
            name: (
                np.asarray(scaler_bundle[name].mean_, dtype=np.float64),
                np.asarray(scaler_bundle[name].scale_, dtype=np.float64),
            )
            for name in ("device_scaler", "bias_scaler", "target_scaler")
        }
        self.eps = _LAYERNORM_EPS
        self.embed = int(config.get("embed_size", 16))
        self.device_fields = list(scaler_bundle["device_feature_fields"])
        self.bias_fields = list(scaler_bundle["bias_feature_fields"])
        self.target_fields = list(scaler_bundle["target_fields"])

        checkpoint_targets = list(checkpoint.get("target_fields", self.target_fields))
        if checkpoint_targets != self.target_fields:
            raise ValueError(
                "2DFET checkpoint/scaler target mismatch: %r != %r"
                % (checkpoint_targets, self.target_fields)
            )

    def _std_transform(self, x, name):
        mean, scale = self.scaler[name]
        return (x - mean) / scale

    def _std_inverse(self, x, name):
        mean, scale = self.scaler[name]
        return x * scale + mean

    def _device_tower(self, xd):
        # xd: (N, 5) scaled device features. Per-feature tokenisation + mean pool.
        w = self.w
        emb = w["device_type_embeddings.weight"]  # (5, embed)
        n = xd.shape[0]
        tokens = np.empty((n, xd.shape[1], self.embed), dtype=np.float64)
        for i in range(xd.shape[1]):
            scalar = xd[:, i:i + 1]
            type_emb = np.broadcast_to(emb[i], (n, self.embed))
            t = np.concatenate([scalar, type_emb], axis=1)
            t = _linear(t, w["device_shared_mlp.0.weight"], w["device_shared_mlp.0.bias"])
            t = _layer_norm(t, self.eps, w["device_shared_mlp.1.weight"], w["device_shared_mlp.1.bias"])
            t = _gelu(t)
            t = _linear(t, w["device_shared_mlp.3.weight"], w["device_shared_mlp.3.bias"])
            tokens[:, i, :] = t
        h_p = tokens.mean(axis=1)
        h_p = _layer_norm(h_p, self.eps)  # F.layer_norm, no affine
        return h_p

    def _bias_tower(self, xb):
        w = self.w
        h = _linear(xb, w["bias_mlp.0.weight"], w["bias_mlp.0.bias"])
        h = _layer_norm(h, self.eps, w["bias_mlp.1.weight"], w["bias_mlp.1.bias"])
        h = _gelu(h)
        h = _linear(h, w["bias_mlp.3.weight"], w["bias_mlp.3.bias"])
        h = _layer_norm(h, self.eps, w["bias_mlp.4.weight"], w["bias_mlp.4.bias"])
        h = _gelu(h)
        h = _linear(h, w["bias_mlp.6.weight"], w["bias_mlp.6.bias"])
        return h

    def _forward_scaled(self, xd, xb):
        w = self.w
        h_p = self._device_tower(xd)
        h_v = self._bias_tower(xb)
        # FiLM fusion
        film = _linear(h_p, w["film_projection.weight"], w["film_projection.bias"])
        gamma, beta = film[:, :self.embed], film[:, self.embed:]
        h = _layer_norm(h_v * gamma + beta, self.eps)
        # Output head
        h = _linear(h, w["output_head.0.weight"], w["output_head.0.bias"])
        h = _layer_norm(h, self.eps, w["output_head.1.weight"], w["output_head.1.bias"])
        h = _gelu(h)
        out = _linear(h, w["output_head.3.weight"], w["output_head.3.bias"])
        return out

    def predict_points(self, x_device_raw, x_bias_raw):
        """Physical [Id, Q] for raw (tox,Lg,eps_ox,meff,D) [SI] & (Vg,Vd) rows."""
        x_device_raw = np.asarray(x_device_raw, dtype=np.float64).reshape(-1, len(self.device_fields))
        x_bias_raw = np.asarray(x_bias_raw, dtype=np.float64).reshape(-1, len(self.bias_fields))
        xd = self._std_transform(x_device_raw, "device_scaler")
        xb = self._std_transform(x_bias_raw, "bias_scaler")
        scaled = self._forward_scaled(xd, xb)
        y_trans = self._std_inverse(scaled, "target_scaler")  # log10(Id), Q
        y = y_trans.copy()
        id_idx = self.target_fields.index("Id")
        y[:, id_idx] = np.power(10.0, y_trans[:, id_idx])  # undo log10
        return y

    def predict_grid(self, tox, Lg, eps_ox, meff, D, Vg, Vd):
        """I-V over a Vg x Vd grid for one device. Returns (Id, Q), each (len(Vg), len(Vd))."""
        Vg = np.asarray(Vg, dtype=np.float64).ravel()
        Vd = np.asarray(Vd, dtype=np.float64).ravel()
        VG, VD = np.meshgrid(Vg, Vd, indexing="ij")
        n = VG.size
        device_row = np.array([tox, Lg, eps_ox, meff, D], dtype=np.float64)
        x_device = np.broadcast_to(device_row, (n, 5))
        x_bias = np.column_stack([VG.ravel(), VD.ravel()])
        y = self.predict_points(x_device, x_bias)
        Id = y[:, self.target_fields.index("Id")].reshape(VG.shape)
        Q = y[:, self.target_fields.index("Q")].reshape(VG.shape)
        return Id, Q


_SURROGATE = None


def _get_surrogate():
    global _SURROGATE
    if _SURROGATE is None:
        _SURROGATE = Surrogate()
    return _SURROGATE


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
            return np.linspace(v_input['start'], v_input['end'], int(v_input['step']))
        raise ValueError("Dict format must contain 'start', 'end', 'step' keys")

    raise ValueError("Unsupported voltage input type: %s" % type(v_input))


# Channel material dropdown: the frontend sends the option index.
MATERIALS = ['MoS2', 'MoSe2']
MATERIAL_MEFF = [0.5, 0.6]  # [m0], aligned with MATERIALS

def _resolve_option(value, options, values, name):
    """Map a dropdown selection (index or option name) to its physical value."""
    if isinstance(value, str) and value in options:
        idx = options.index(value)
    else:
        idx = int(float(value))
    if not 0 <= idx < len(options):
        raise ValueError("Unknown %s selection: %s" % (name, value))
    return values[idx]


def _resolve_meff(parameters):
    """Map the 'material' selection (index or name) to meff; accept raw meff too."""
    if parameters.get('meff') is not None:
        return float(parameters['meff'])
    material = parameters.get('material')
    if material is None:
        raise ValueError("Missing device parameter: require 'material' (or 'meff').")
    return _resolve_option(material, MATERIALS, MATERIAL_MEFF, 'material')


def _resolve_D(parameters):
    """The current database and surrogate contain ballistic data only."""
    return 0.0


# Absolute threshold-voltage knob, SiFET-style: the stored/trained sweep is
# -0.15..0.6 V but the UI window is 0..0.5 V, leaving 0.15 V of lower margin
# and 0.1 V of upper margin for the rigid work-function shift (exact physics:
# Vg and Vfb enter the NEGF-Poisson equations only as Vg - Vfb).
VTH_REF = 0.20
VTH_MIN, VTH_MAX = 0.10, 0.35
VG_STEP = 0.0125
# Display window rows in the stored 61-point Vg grid (-0.15..0.6 V)
WIN_START, WIN_END = 12, 53  # rows 12..52 -> Vg 0.0..0.5 V (41 points)


def run_simulation(parameters):
    """Run a 2DFET simulation with the NumPy Two-Tower FiLM surrogate.

    Expects tox and Lg in nm (converted to m internally for the surrogate).
    'material' (MoS2 or MoSe2) selects the channel effective mass.
    Transport is fixed to the ballistic limit (D = 0).
    V_th [V] (reference 0.20) rigidly shifts the transfer characteristics
    along Vg: the model is evaluated at Vg - (V_th - 0.20).
    """
    parameters = convert_str_to_float(parameters)
    tox = parameters.get('tox')
    Lg = parameters.get('Lg')
    eps_ox = parameters.get('eps_ox')
    meff = _resolve_meff(parameters)
    D = _resolve_D(parameters)
    vth = float(parameters.get('V_th', VTH_REF) or VTH_REF)
    dvth = vth - VTH_REF

    if tox is None or Lg is None or eps_ox is None:
        raise ValueError("Missing device parameters: require tox, Lg, eps_ox, material.")

    Vg_array = parse_voltage_input(parameters.get('Vg'))
    Vd_array = parse_voltage_input(parameters.get('Vd'))

    model = _get_surrogate()
    Id, Q = model.predict_grid(
        tox=float(tox) * 1e-9,   # nm -> m
        Lg=float(Lg) * 1e-9,     # nm -> m
        eps_ox=float(eps_ox),
        meff=float(meff),
        D=float(D),
        Vg=Vg_array - dvth,
        Vd=Vd_array,
    )

    device_params = dict(parameters)
    device_params['meff'] = meff  # resolved from 'material'
    device_params['D'] = D        # fixed ballistic value

    return {
        'simulation_data': {
            'Vg': Vg_array.tolist(),
            'Vd': Vd_array.tolist(),
            'Id': Id.tolist(),   # [len(Vg), len(Vd)], A/m (= uA/um)
            'Qg': Q.tolist(),    # [len(Vg), len(Vd)], C/m
        },
        'device_params': device_params,
    }


def get_simulation_data(db_helper, parameters):
    """Fetch pre-computed 2DFET simulation data from the database (SiFET-style).

    'material' is translated to meff, and D is fixed to zero for ballistic
    transport, matching the current database.
    V_th is not a database axis: the stored grid is wider (-0.15..0.6 V) than
    the fixed 0..0.5 V display window; the window is slid by the Vth shift
    over the stored rows, then relabeled back to 0..0.5 V. The margin rows
    absorb shifts of V_th in [0.10, 0.35] around the 0.20 reference.
    """
    parameters = convert_str_to_float(parameters)
    vth = float(parameters.get('V_th', VTH_REF) or VTH_REF)
    vth = min(max(vth, VTH_MIN), VTH_MAX)
    vth_shift = vth - VTH_REF
    meff = _resolve_meff(parameters)
    D = _resolve_D(parameters)

    db_query_params = {k: v for k, v in parameters.items()
                       if k not in ('V_th', 'material', 'meff', 'transport', 'D')}
    db_query_params['meff'] = meff
    db_query_params['D'] = D

    complete_data, exact_match, distance, matched_params = \
        db_helper.get_simulation_data('2DFET', db_query_params)

    if not complete_data:
        return None, False, None, None

    sd = complete_data.get('simulation_data', {})
    vg_values = sd.get('Vg', [])
    vd_values = sd.get('Vd', [])
    id_data = sd.get('Id', [])
    qg_data = sd.get('Qg', [])

    # Slide the fixed display window over the stored grid
    index_shift = -int(round(vth_shift / VG_STEP))
    total_points = len(vg_values)
    start_idx = max(0, min(WIN_START + index_shift, total_points))
    end_idx = max(0, min(WIN_END + index_shift, total_points))

    selected_vg = vg_values[start_idx:end_idx]
    selected_id = id_data[start_idx:end_idx]
    selected_qg = qg_data[start_idx:end_idx]

    # Relabel the window back to the fixed 0..0.5 V axis
    shifted_vg = [round(vg + vth_shift, 4) + 0.0 for vg in selected_vg]  # +0.0 normalizes -0.0

    simulation_data = {
        'Vg': shifted_vg,
        'Vd': vd_values,
        'Id': selected_id,
        'Qg': selected_qg,
        'nVg': len(shifted_vg),
        'nVd': len(vd_values),
    }

    device_params = dict(complete_data.get('device_params', {}))
    device_params['V_th'] = vth
    if 'meff' in device_params:
        try:
            device_params['material'] = MATERIALS[MATERIAL_MEFF.index(device_params['meff'])]
        except ValueError:
            pass
    adjusted_data = {
        'simulation_data': simulation_data,
        'device_params': device_params,
    }

    return adjusted_data, exact_match, distance, matched_params


@MODELS.register("2DFET")
class TwoDFET:
    simulation_func = staticmethod(run_simulation)
    device_params = ['tox', 'Lg', 'eps_ox', 'material', 'V_th']
    voltage_params = ['Vg', 'Vd']
    postprocess = staticmethod(get_simulation_data)


if __name__ == "__main__":
    parameters = {
        'tox': 2.0,
        'Lg': 10.0,
        'eps_ox': 20.0,
        'meff': 0.5,
        'D': 0.0,
        'Vg': {'start': -0.15, 'end': 0.6, 'step': 61},
        'Vd': {'start': 0.001, 'end': 0.501, 'step': 41},
    }
    result = run_simulation(parameters)
    Id = np.array(result['simulation_data']['Id'])
    print("Id grid", Id.shape, "min", Id.min(), "max", Id.max())
