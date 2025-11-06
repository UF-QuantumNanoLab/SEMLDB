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
    'size_0'              : 1080,
    'size_1'              : 540,
    'size_2'              : 270,
    'size_3'              : 68,
    'latent_size'         : 5,
    'LS_input_f_size'     : 723,
    'degree'              : 11,
    'cross_degree'        : 9,
    }

IdVg_config = {
    'size_0'              : 2412,
    'size_1'              : 1206,
    'size_2'              : 603,
    'size_3'              : 151,
    'latent_size'         : 7,
    'LS_input_f_size'     : 495,
    'degree'              : 8,
    'cross_degree'        : 8,
    }

ft_config = {
    'size_0'              : 161,
    'size_1'              : 80,
    'size_2'              : 40,
    'size_3'              : 10,
    'latent_size'         : 5,
    'LS_input_f_size'     : 1009,
    'degree'              : 12,
    'cross_degree'        : 10,
    }

CV_config = {
    'size_0'              : 802,
    'size_1'              : 401,
    'size_2'              : 200,
    'size_3'              : 50,
    'latent_size'         : 4,
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
        'DiamondFET', parameters
    )
    
    if not complete_data:
        return None, False, None, None
    
    adjusted_data = {
        'simulation_data': complete_data.get('simulation_data', {}),
        'device_params': complete_data.get('device_params', {})
    }
    
    return adjusted_data, exact_match, distance, matched_params

def run_AE_sim(parameters):
    Lg, Lgs, Tox, Rc = 1e-3*parameters.get('Lg'), 1e-3*parameters.get('Lgs'), 1e-3*parameters.get('Tox'), parameters.get('Rc')

    feature_names   = ['Lg', 'Lgs', 'Tox', 'Rc']
    x = [Lg, Lgs, Tox, Rc]
    x_df = pd.DataFrame([x], columns=feature_names)

    # IdVd inference section
    model_dir = os.path.dirname(__file__)
    parent_path = os.path.join(model_dir, 'idvd_diamond_9_curves_linear/')
    IdVd_AE = Autoencoder(**IdVd_config)
    IdVd_AE.load_state_dict(torch.load(os.path.join(parent_path, 'idvd_diamond_9_curves_linear_scale.pth'), map_location=torch.device('cpu')))
    IdVd_AE.eval()

    IdVd_LS_poly = LatentSpacePolyNN(dim_input=IdVd_config['LS_input_f_size'], dim_output=IdVd_config['latent_size'])
    IdVd_LS_poly.load_state_dict(torch.load(os.path.join(parent_path, 'idvd_diamond_poly_regression_model.pth'), map_location=torch.device('cpu')))
    IdVd_LS_poly.eval()

    IdVd_scaler_x = joblib.load(os.path.join(parent_path, 'idvd_diamond_scaler_x.pkl'))
    IdVd_scaler_y_IV = joblib.load(os.path.join(parent_path, 'idvd_diamond_scaler_iv.pkl'))
    IdVdscaler_ls = joblib.load(os.path.join(parent_path, 'idvd_diamond_scaler_ls.pkl'))

    x_scaled = IdVd_scaler_x.transform(x_df)
    x_features = polynomial_features(x_scaled, IdVd_config['degree'], IdVd_config['cross_degree'])
    x_scaled_tensor = torch.tensor(x_features, dtype=torch.float32)

    y_ls = IdVd_LS_poly(x_scaled_tensor)

    decoder_input   = IdVdscaler_ls.inverse_transform(y_ls.detach().numpy())
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float32)
    decoder_output  = IdVd_AE.decoder(decoder_input_tensor).detach().numpy()

    IdVd = -1 * IdVd_scaler_y_IV.inverse_transform(decoder_output.reshape(1,-1)).reshape((9, -1))

    # IdVg inference section
    model_dir = os.path.dirname(__file__)
    parent_path = os.path.join(model_dir, 'idvg_diamond_6_curves_linear_log/')
    IdVg_AE = Autoencoder(**IdVg_config)
    IdVg_AE.load_state_dict(torch.load(os.path.join(parent_path, 'idvg_diamond_6_curves_linear_log_scale.pth'), map_location=torch.device('cpu')))
    IdVg_AE.eval()

    IdVg_LS_poly = LatentSpacePolyNN(dim_input=IdVg_config['LS_input_f_size'], dim_output=IdVg_config['latent_size'])
    IdVg_LS_poly.load_state_dict(torch.load(os.path.join(parent_path, 'idvg_diamond_poly_regression_model.pth'), map_location=torch.device('cpu')))
    IdVg_LS_poly.eval()

    IdVg_scaler_x = joblib.load(os.path.join(parent_path, 'idvg_diamond_scaler_x.pkl'))
    IdVg_scaler_y_IV = joblib.load(os.path.join(parent_path, 'idvg_diamond_scaler_iv_linear.pkl'))
    IdVg_scaler_y_IV_log = joblib.load(os.path.join(parent_path, 'idvg_diamond_scaler_iv_log.pkl'))
    IdVgscaler_ls = joblib.load(os.path.join(parent_path, 'idvg_diamond_scaler_ls.pkl'))

    x_scaled = IdVg_scaler_x.transform(x_df)
    x_features = polynomial_features(x_scaled, IdVg_config['degree'], IdVg_config['cross_degree'])
    x_scaled_tensor = torch.tensor(x_features, dtype=torch.float32)

    y_ls = IdVg_LS_poly(x_scaled_tensor)

    decoder_input = IdVgscaler_ls.inverse_transform(y_ls.detach().numpy())
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float32)
    decoder_output = IdVg_AE.decoder(decoder_input_tensor).detach().numpy()
    decoder_output = decoder_output.flatten().reshape((2, -1))

    IdVg_linear_scaled, IdVg_log_scaled = decoder_output[0], decoder_output[1]
    IdVg, IdVg_log = -1 * IdVg_scaler_y_IV.inverse_transform(IdVg_linear_scaled.reshape(1,-1)).reshape((6, -1)), IdVg_scaler_y_IV_log.inverse_transform(IdVg_log_scaled.reshape(1,-1)).reshape((6, -1))

    # ft inference section
    model_dir = os.path.dirname(__file__)
    parent_path = os.path.join(model_dir, 'ft_diamond_1_curve_linear/')
    ft_AE = Autoencoder(**ft_config)
    ft_AE.load_state_dict(torch.load(os.path.join(parent_path, 'ft_diamond_1_curve_linear_scale.pth'), map_location=torch.device('cpu')))
    ft_AE.eval()

    ft_LS_poly = LatentSpacePolyNN(dim_input=ft_config['LS_input_f_size'], dim_output=ft_config['latent_size'])
    ft_LS_poly.load_state_dict(torch.load(os.path.join(parent_path, 'ft_diamond_poly_regression_model.pth'), map_location=torch.device('cpu')))
    ft_LS_poly.eval()

    ft_scaler_x = joblib.load(os.path.join(parent_path, 'ft_diamond_scaler_x.pkl'))
    ft_scaler_y_IV = joblib.load(os.path.join(parent_path, 'ft_diamond_scaler_iv.pkl'))
    ftscaler_ls = joblib.load(os.path.join(parent_path, 'ft_diamond_scaler_ls.pkl'))

    x_scaled = ft_scaler_x.transform(x_df)
    x_features = polynomial_features(x_scaled, ft_config['degree'], ft_config['cross_degree'])
    x_scaled_tensor = torch.tensor(x_features, dtype=torch.float32)

    y_ls = ft_LS_poly(x_scaled_tensor)

    decoder_input   = ftscaler_ls.inverse_transform(y_ls.detach().numpy())
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float32)
    decoder_output  = ft_AE.decoder(decoder_input_tensor).detach().numpy()

    ft = ft_scaler_y_IV.inverse_transform(decoder_output.reshape(1,-1)).flatten()

    # CV inference section
    model_dir = os.path.dirname(__file__)
    parent_path = os.path.join(model_dir, 'cv_diamond_2_curves_linear/')
    CV_AE = Autoencoder(**CV_config)
    CV_AE.load_state_dict(torch.load(os.path.join(parent_path, 'cv_diamond_2_curves_linear_scale.pth'), map_location=torch.device('cpu')))
    CV_AE.eval()

    CV_LS_poly = LatentSpacePolyNN(dim_input=CV_config['LS_input_f_size'], dim_output=CV_config['latent_size'])
    CV_LS_poly.load_state_dict(torch.load(os.path.join(parent_path, 'cv_diamond_poly_regression_model.pth'), map_location=torch.device('cpu')))
    CV_LS_poly.eval()

    CV_scaler_x = joblib.load(os.path.join(parent_path, 'cv_diamond_scaler_x.pkl'))
    CV_scaler_y_IV = joblib.load(os.path.join(parent_path, 'cv_diamond_scaler_iv.pkl'))
    CVscaler_ls = joblib.load(os.path.join(parent_path, 'cv_diamond_scaler_ls.pkl'))

    x_scaled = CV_scaler_x.transform(x_df)
    x_features = polynomial_features(x_scaled, CV_config['degree'], CV_config['cross_degree'])
    x_scaled_tensor = torch.tensor(x_features, dtype=torch.float32)

    y_ls = CV_LS_poly(x_scaled_tensor)

    decoder_input   = CVscaler_ls.inverse_transform(y_ls.detach().numpy())
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float32)
    decoder_output  = CV_AE.decoder(decoder_input_tensor).detach().numpy()

    CV = CV_scaler_y_IV.inverse_transform(decoder_output.reshape(1,-1)).reshape((2, -1))

    return_body = {
        'simulation_data': {
            'Id_Vd': {
                    'Vg': [-5, -3, -1, 1, 3, 5, 7, 9, 11],
                    'Vd': np.linspace(-12, -0.02, IdVd.shape[1]).tolist() + [0],
                    'Id': IdVd.tolist(),
                },
            'Id_Vg': {
                    'Vg': np.linspace(-5, 12, IdVg.shape[1]).tolist(),
                    'Vd': [-11, -9, -7, -5, -3, -1],
                    'Id': IdVg.tolist(),
                    'Id_log': IdVg_log.tolist(),
                },
            'ft': {
                    'Vg': np.linspace(-4, 12, ft.shape[0]).tolist(),
                    'ft': ft.tolist(),
                },
            'C_Vg': {
                    'Vg': np.linspace(-4, 12, CV.shape[1]).tolist(),
                    'C_gate': CV[0].tolist(),
                    'C_drain': CV[1].tolist(),
                }
        },
        'device_params': parameters
    }
    
    return return_body


@MODELS.register()
class DiamondFET:
    simulation_func = run_AE_sim
    device_params = ['Lg', 'Lgs', 'Tox', 'Rc']
    voltage_params = None
    postprocess = get_simulation_data
