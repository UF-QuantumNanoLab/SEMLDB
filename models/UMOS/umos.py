import torch
import os
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin

from .. import MODELS


IdVd_config = {
    'size_0'              : 1998,
    'size_1'              : 999,
    'size_2'              : 500,
    'size_3'              : 125,
    'latent_size'         : 7,
    'LS_input_f_size'     : 334,
    'degree'              : 8,
    'cross_degree'        : 7,
    }

IdVg_config = {
    'size_0'              : 998,
    'size_1'              : 499,
    'size_2'              : 250,
    'size_3'              : 62,
    'latent_size'         : 8,
    'LS_input_f_size'     : 214,
    'degree'              : 7,
    'cross_degree'        : 6,
    }

BV_config = {
    'size_0'              : 1002,
    'size_1'              : 501,
    'size_2'              : 250,
    'size_3'              : 62,
    'latent_size'         : 6,
    'LS_input_f_size'     : 210,
    'degree'              : 6,
    'cross_degree'        : 6,
    }

CV_config = {
    'size_0'              : 1984,
    'size_1'              : 992,
    'size_2'              : 496,
    'size_3'              : 124,
    'latent_size'         : 7,
    'LS_input_f_size'     : 715,
    'degree'              : 9,
    'cross_degree'        : 9,
    }

class Autoencoder(nn.Module):
    def __init__(self, size_0, size_1, size_2, size_3, latent_size, **kwargs):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(size_0, size_1),
            nn.ReLU(),
            nn.Linear(size_1, size_2),
            nn.ReLU(),
            nn.Linear(size_2, size_3),
            nn.ReLU(),
            nn.Linear(size_3, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, size_3),
            nn.ReLU(),
            nn.Linear(size_3, size_2),
            nn.ReLU(),
            nn.Linear(size_2, size_1),
            nn.ReLU(),
            nn.Linear(size_1, size_0),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_latent_space(self, x):
        return self.encoder(x)

def polynomial_features(x, d, p):
    n_samples, n_features = x.shape

    # Base polynomial features up to degree d
    poly = PolynomialFeatures(d, include_bias=True)
    x_poly = poly.fit_transform(x)

    # Filter out terms with order higher than p
    def filter_terms(terms, degree):
        return [term for term in terms if sum(term) <= degree or max(term) == sum(term)]

    feature_indices = poly.powers_  # Array of powers for each feature
    # print(feature_indices)
    filtered_indices = filter_terms(feature_indices, p)
    
    # Create new feature matrix with filtered terms
    x_filtered_poly = np.empty((n_samples, len(filtered_indices)))

    for i, index in enumerate(filtered_indices):
        # print(index)
        term = np.prod([x[:, j]**exp for j, exp in enumerate(index)], axis=0)
        x_filtered_poly[:, i] = term

    return x_filtered_poly

class PolynomialFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, degree, cross_degree):
        self.degree = degree
        self.cross_degree = cross_degree

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        n_samples, n_features = x.shape

        # Base polynomial features up to degree d
        x_filtered_poly = polynomial_features(x, self.degree, self.cross_degree)

        return x_filtered_poly

class LatentSpacePolyNN(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(LatentSpacePolyNN, self).__init__()
        self.linear = nn.Linear(dim_input, dim_output)
    
    def forward(self, x):
        out = self.linear(x)
        return out

def get_simulation_data(db_helper, parameters):
    """
    Fetch simulation data with preset condition
    """
    complete_data, exact_match, distance, matched_params = db_helper.get_simulation_data(
        'UMOS', parameters
    )
    
    if not complete_data:
        return None, False, None, None
    
    adjusted_data = {
        'simulation_data': complete_data.get('simulation_data', {}),
        'device_params': complete_data.get('device_params', {})
    }
    
    return adjusted_data, exact_match, distance, matched_params

def run_AE_sim(parameters):
    wpil, wsp, Nbuf, Npil = parameters.get('wpil'), parameters.get('wsp'), parameters.get('Nbuf'), parameters.get('Npil')

    feature_names   = ['wpil', 'wsp', 'Nbuf', 'Npil']
    x = [wpil, wsp, Nbuf, Npil]
    x_df = pd.DataFrame([x], columns=feature_names)

    # IdVd inference section
    model_dir = os.path.dirname(__file__)
    parent_path = os.path.join(model_dir, 'idvd_umos_1_curves_log_linear/')
    IdVd_AE = Autoencoder(**IdVd_config)
    IdVd_AE.load_state_dict(torch.load(os.path.join(parent_path, 'idvd_umos_curves_log_linear_scale.pth'), map_location=torch.device('cpu')))
    IdVd_AE.eval()

    IdVd_LS_poly = LatentSpacePolyNN(dim_input=IdVd_config['LS_input_f_size'], dim_output=IdVd_config['latent_size'])
    IdVd_LS_poly.load_state_dict(torch.load(os.path.join(parent_path, 'idvd_umos_poly_regression_model.pth'), map_location=torch.device('cpu')))
    IdVd_LS_poly.eval()

    IdVd_scaler_x = joblib.load(os.path.join(parent_path, 'idvd_scaler_x.pkl'))
    IdVd_scaler_y_IV = joblib.load(os.path.join(parent_path, 'idvd_scaler_iv_linear.pkl'))
    IdVd_scaler_y_IV_log = joblib.load(os.path.join(parent_path, 'idvd_scaler_iv_log.pkl'))
    IdVdscaler_ls = joblib.load(os.path.join(parent_path, 'idvd_scaler_ls.pkl'))

    x_scaled = IdVd_scaler_x.transform(x_df)
    x_features = polynomial_features(x_scaled, IdVd_config['degree'], IdVd_config['cross_degree'])
    x_scaled_tensor = torch.tensor(x_features, dtype=torch.float32)

    y_ls = IdVd_LS_poly(x_scaled_tensor)

    decoder_input = IdVdscaler_ls.inverse_transform(y_ls.detach().numpy())
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float32)
    decoder_output = IdVd_AE.decoder(decoder_input_tensor).detach().numpy()
    decoder_output = decoder_output.flatten().reshape((2, -1))

    IdVd_linear_scaled, IdVd_log_scaled = decoder_output[0], decoder_output[1]
    IdVd, IdVd_log = IdVd_scaler_y_IV.inverse_transform(IdVd_linear_scaled.reshape(1,-1)).flatten(), IdVd_scaler_y_IV_log.inverse_transform(IdVd_log_scaled.reshape(1,-1)).flatten()

    # IdVg inference section
    model_dir = os.path.dirname(__file__)
    parent_path = os.path.join(model_dir, 'idvg_umos_1_curves_log_linear/')
    IdVg_AE = Autoencoder(**IdVg_config)
    IdVg_AE.load_state_dict(torch.load(os.path.join(parent_path, 'idvg_umos_curves_log_linear_scale.pth'), map_location=torch.device('cpu')))
    IdVg_AE.eval()

    IdVg_LS_poly = LatentSpacePolyNN(dim_input=IdVg_config['LS_input_f_size'], dim_output=IdVg_config['latent_size'])
    IdVg_LS_poly.load_state_dict(torch.load(os.path.join(parent_path, 'idvg_umos_poly_regression_model.pth'), map_location=torch.device('cpu')))
    IdVg_LS_poly.eval()

    IdVg_scaler_x = joblib.load(os.path.join(parent_path, 'idvg_scaler_x.pkl'))
    IdVg_scaler_y_IV = joblib.load(os.path.join(parent_path, 'idvg_scaler_iv_linear.pkl'))
    IdVg_scaler_y_IV_log = joblib.load(os.path.join(parent_path, 'idvg_scaler_iv_log.pkl'))
    IdVgscaler_ls = joblib.load(os.path.join(parent_path, 'idvg_scaler_ls.pkl'))

    x_scaled = IdVg_scaler_x.transform(x_df)
    x_features = polynomial_features(x_scaled, IdVg_config['degree'], IdVg_config['cross_degree'])
    x_scaled_tensor = torch.tensor(x_features, dtype=torch.float32)

    y_ls = IdVg_LS_poly(x_scaled_tensor)

    decoder_input = IdVgscaler_ls.inverse_transform(y_ls.detach().numpy())
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float32)
    decoder_output = IdVg_AE.decoder(decoder_input_tensor).detach().numpy()
    decoder_output = decoder_output.flatten().reshape((2, -1))
    
    IdVg_linear_scaled, IdVg_log_scaled = decoder_output[0], decoder_output[1]
    IdVg, IdVg_log = IdVg_scaler_y_IV.inverse_transform(IdVg_linear_scaled.reshape(1,-1)).flatten(), IdVg_scaler_y_IV_log.inverse_transform(IdVg_log_scaled.reshape(1,-1)).flatten()

    # BV inference section
    model_dir = os.path.dirname(__file__)
    parent_path = os.path.join(model_dir, 'bv_umos_1_curves_log_linear/')
    BV_AE = Autoencoder(**BV_config)
    BV_AE.load_state_dict(torch.load(os.path.join(parent_path, 'bv_umos_curves_log_linear_scale.pth'), map_location=torch.device('cpu')))
    BV_AE.eval()

    BV_LS_poly = LatentSpacePolyNN(dim_input=BV_config['LS_input_f_size'], dim_output=BV_config['latent_size'])
    BV_LS_poly.load_state_dict(torch.load(os.path.join(parent_path, 'bv_umos_poly_regression_model.pth'), map_location=torch.device('cpu')))
    BV_LS_poly.eval()

    BV_scaler_x = joblib.load(os.path.join(parent_path, 'bv_scaler_x.pkl'))
    BV_scaler_y_IV = joblib.load(os.path.join(parent_path, 'bv_scaler_iv_linear.pkl'))
    BV_scaler_y_IV_log = joblib.load(os.path.join(parent_path, 'bv_scaler_iv_log.pkl'))
    BVscaler_ls = joblib.load(os.path.join(parent_path, 'bv_scaler_ls.pkl'))

    x_scaled = BV_scaler_x.transform(x_df)
    x_features = polynomial_features(x_scaled, BV_config['degree'], BV_config['cross_degree'])
    x_scaled_tensor = torch.tensor(x_features, dtype=torch.float32)

    y_ls = BV_LS_poly(x_scaled_tensor)

    decoder_input = BVscaler_ls.inverse_transform(y_ls.detach().numpy())
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float32)
    decoder_output = BV_AE.decoder(decoder_input_tensor).detach().numpy()
    decoder_output = decoder_output.flatten().reshape((2, -1))

    BV_linear_scaled, BV_log_scaled = decoder_output[0], decoder_output[1]
    BV_linear, BV_log = BV_scaler_y_IV.inverse_transform(BV_linear_scaled.reshape(1,-1)).flatten(), BV_scaler_y_IV_log.inverse_transform(BV_log_scaled.reshape(1,-1)).flatten()

    # CV inference section
    model_dir = os.path.dirname(__file__)
    parent_path = os.path.join(model_dir, 'cv_umos_1_curves_log_linear/')
    CV_AE = Autoencoder(**CV_config)
    CV_AE.load_state_dict(torch.load(os.path.join(parent_path, 'cv_umos_curves_log_linear_scale.pth'), map_location=torch.device('cpu')))
    CV_AE.eval()

    CV_LS_poly = LatentSpacePolyNN(dim_input=CV_config['LS_input_f_size'], dim_output=CV_config['latent_size'])
    CV_LS_poly.load_state_dict(torch.load(os.path.join(parent_path, 'cv_umos_poly_regression_model.pth'), map_location=torch.device('cpu')))
    CV_LS_poly.eval()

    CV_scaler_x = joblib.load(os.path.join(parent_path, 'cv_scaler_x.pkl'))
    CV_scaler_y_IV = joblib.load(os.path.join(parent_path, 'cv_scaler_cv_linear.pkl'))
    CV_scaler_y_IV_log = joblib.load(os.path.join(parent_path, 'cv_scaler_cv_log.pkl'))
    CVscaler_ls = joblib.load(os.path.join(parent_path, 'cv_scaler_ls.pkl'))

    x_scaled = CV_scaler_x.transform(x_df)
    x_features = polynomial_features(x_scaled, CV_config['degree'], CV_config['cross_degree'])
    x_scaled_tensor = torch.tensor(x_features, dtype=torch.float32)

    y_ls = CV_LS_poly(x_scaled_tensor)

    decoder_input = CVscaler_ls.inverse_transform(y_ls.detach().numpy())
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float32)
    decoder_output = CV_AE.decoder(decoder_input_tensor).detach().numpy()
    decoder_output = decoder_output.flatten().reshape((2, -1))

    CV_linear_scaled, CV_log_scaled = decoder_output[0], decoder_output[1]
    CV_linear, CV_log = CV_scaler_y_IV.inverse_transform(CV_linear_scaled.reshape(1,-1)).reshape(2, -1), CV_scaler_y_IV_log.inverse_transform(CV_log_scaled.reshape(1,-1)).reshape(2, -1)
    


    return_body = {
        'simulation_data': {
            'Id_Vd': {
                    'Vg': [15],
                    'Vd': np.linspace(0, 5, IdVd.shape[0]).tolist(),
                    'Id': [IdVd.tolist()],
                },
            'Id_Vg': {
                    'Vg': np.linspace(0, 20, IdVg.shape[0]).tolist(),
                    'Vd': [10],
                    'Id': [IdVg.tolist()],
                    'Id_log': [IdVg_log.tolist()],
                },
            'BV': {
                    'Vd': np.linspace(0, 2000, BV_linear.shape[0]).tolist(),
                    'Id': BV_linear.tolist(),
                },
            'C_Vd': {
                    'Vd': np.linspace(0, 100, CV_linear.shape[0]).tolist(),
                    'C_gate': CV_linear[0].tolist(),
                    'C_drain': CV_linear[1].tolist(),
                }
        },
        'device_params': parameters
    }
    
    return return_body


@MODELS.register()
class UMOS:
    simulation_func = run_AE_sim
    device_params = ['wpil', 'wsp', 'Nbuf', 'Npil']
    voltage_params = None
    postprocess = get_simulation_data
