"""
cryogenic.py
------------
Cryogenic MOSFET device.

Live simulation chains two BYOM adapters trained by the TCAD-automation-pipeline
project (see IVmodel/, CVmodel/, CVgdmodel/ there):

    [Temp, Lg, Tox] --PolyAdapter.predict--> Z --AEDecoderOnlyAdapter.predict--> curve

Each of the three curves (Id-Vg, Cgg-Vg, Cgd-Vg) has its own poly/decoder
weight pair under IV/, CVgg/, CVgd/ respectively. byom_adapter_poly.py and
byom_adapter_autoencoder.py are verbatim copies of the adapters used to train
those weights (only the inference-only classes -- PolyAdapter.predict and
AEDecoderOnlyAdapter.predict -- are exercised here).
"""
import os

import numpy as np

from .. import MODELS
from .byom_adapter_poly import PolyAdapter
from .byom_adapter_autoencoder import AEDecoderOnlyAdapter

_MODEL_DIR = os.path.dirname(__file__)
_FEATURE_SET_Current = ['Temp', 'Lg', 'Tox']
_FEATURE_SET_Capacitance = ['Temp', 'Tox', 'Lg']

# Latent dimensionality of each sub-model's poly->decoder bridge
# (from ae_ext_params.json / poly's z0..zN label_set at training time).
_LATENT_DIMS = {'IV': 6, 'CVgg': 6, 'CVgd': 8}

# Vg sweep grids reconstructed from the training CSVs' "V = ..." column
# headers (uniform step in both cases).
_VG_IV = np.linspace(0.005, 0.995, 199)
_VG_CV = np.linspace(-0.995, 0.980, 80)

_adapters = {}


def _get_adapters(curve_name):
    """Lazily construct and cache the (poly, decoder) adapter pair for one curve."""
    if curve_name not in _adapters:
        base = os.path.join(_MODEL_DIR, curve_name)
        _FEATURE_SET = _FEATURE_SET_Current if curve_name == 'IV' else _FEATURE_SET_Capacitance
        z_labels = [f'z{i}' for i in range(_LATENT_DIMS[curve_name])]
        vg_grid = _VG_IV if curve_name == 'IV' else _VG_CV

        poly = PolyAdapter(
            pretrained_model_path=os.path.join(base, 'poly'),
            context={'feature_set': _FEATURE_SET, 'label_set': z_labels},
        )
        decoder = AEDecoderOnlyAdapter(
            pretrained_model_path=os.path.join(base, 'decoder'),
            context={
                'feature_set': z_labels,
                'label_set': [f'V={v:.3f}' for v in vg_grid],
            },
        )
        _adapters[curve_name] = (poly, decoder)
    return _adapters[curve_name]


# The autoencoders for Cgg/Cgd were trained on log-transformed targets
# (full_pipeline.py's feature_transform_config: 'log' for the all-positive
# Cgg curves, 'neg_log_abs' for the all-negative Cgd curves -- see
# _transform_y/_invert_y). Decoder output is therefore ln(C) or ln(|C|), and
# must be exponentiated (with the right sign) to recover physical Farads.
# Verified by round-tripping real training curves through the actual
# encoder_only/decoder_only weights: ln(curve) -> encode -> decode -> exp(.)
# reconstructs the original curve to within the models' reported R^2.
_INVERT_LOG = {
    'IV': lambda y: y,
    'CVgg': lambda y: np.exp(y),
    'CVgd': lambda y: -np.exp(y),
}


def _predict_curve(curve_name, parameters):
    poly, decoder = _get_adapters(curve_name)
    x = np.array([[float(parameters[p]) for p in poly.feature_set]])
    z = poly.predict(x)
    y = decoder.predict(z)
    return _INVERT_LOG[curve_name](y[0])


def run_simulation(parameters):
    """
    Main simulation function using the poly->decoder ML models.

    Args:
        parameters (dict): must contain Temp, Lg, Tox.

    Returns:
        dict: {'simulation_data': {...}, 'device_params': parameters}
    """
    missing = [p for p in _FEATURE_SET_Current if p not in parameters]
    if missing:
        raise ValueError(f"Missing required parameter(s): {missing}")

    id_vg = _predict_curve('IV', parameters)
    c_gate = _predict_curve('CVgg', parameters)
    c_drain = _predict_curve('CVgd', parameters)

    return {
        'simulation_data': {
            'Id_Vg': {
                'Vg': _VG_IV.tolist(),
                'Id': id_vg.tolist(),
            },
            'C_Vg': {
                'Vg': _VG_CV.tolist(),
                'C_gate': c_gate.tolist(),
                'C_drain': c_drain.tolist(),
            },
        },
        'device_params': parameters,
    }


def get_simulation_data(db_helper, parameters):
    """
    Fetch pre-computed simulation data from the database (nearest-match
    fallback handled by db_helper.get_simulation_data).
    """
    complete_data, exact_match, distance, matched_params = db_helper.get_simulation_data(
        'Cryogenic', parameters
    )
    if not complete_data:
        return None, False, None, None

    adjusted_data = {
        'simulation_data': complete_data.get('simulation_data', {}),
        'device_params': complete_data.get('device_params', {}),
    }
    return adjusted_data, exact_match, distance, matched_params


@MODELS.register()
class Cryogenic:
    simulation_func = run_simulation
    device_params = ['Temp', 'Lg', 'Tox']
    voltage_params = None
    postprocess = get_simulation_data
