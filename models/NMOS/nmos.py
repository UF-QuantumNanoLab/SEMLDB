import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin

from .. import MODELS


IdVg_config = {
    'size_0'              : 800,
    'size_1'              : 400,
    'size_2'              : 200,
    'size_3'              : 50,
    'latent_size'         : 5,
    'LS_input_f_size'     : 120,
    'degree'              : 7,
    'cross_degree'        : 7,
    }

IdVd_config = {
    'size_0'              : 800,
    'size_1'              : 400,
    'size_2'              : 200,
    'size_3'              : 50,
    'latent_size'         : 3,
    'LS_input_f_size'     : 235,
    'degree'              : 14,
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

def convert_map_tensor_to_list(data_loader):
    input_list = []
    latent_list = []
    for batch in data_loader:
        input_tensor, latent_tensor = batch
        input_list.append(tensor_to_numpy(input_tensor))
        latent_list.append(tensor_to_numpy(latent_tensor))

    input_np = np.vstack(input_list)
    latent_np = np.vstack(latent_list)

    return input_np, latent_np

def get_simulation_data(db_helper, parameters):
    """
    Fetch simulation data with preset condition
    """
    complete_data, exact_match, distance, matched_params = db_helper.get_simulation_data(
        'NMOS', parameters
    )
    
    if not complete_data:
        return None, False, None, None
    
    adjusted_data = {
        'simulation_data': complete_data.get('simulation_data', {}),
        'device_params': complete_data.get('device_params', {})
    }
    
    return adjusted_data, exact_match, distance, matched_params

def run_AE_sim(parameters):
    Lg, THF, XjSD = 1e-3*parameters.get('Lg'), 1e-3*parameters.get('THF'), 1e-3*parameters.get('XjSD')

    feature_names   = ['Lg', 'THF', 'XjSD']
    x = [Lg, THF, XjSD]
    x_df = pd.DataFrame([x], columns=feature_names)

    # IdVd inference section
    parent_path = './models/NMOS/idvd_nmos_3_par_4_curves_linear/'
    IdVd_AE = Autoencoder(**IdVd_config)
    IdVd_AE.load_state_dict(torch.load(parent_path + 'idvd_nmos_4_curves_linear_scale.pth', map_location=torch.device('cpu')))
    IdVd_AE.eval()

    IdVd_LS_poly = LatentSpacePolyNN(dim_input=IdVd_config['LS_input_f_size'], dim_output=IdVd_config['latent_size'])
    IdVd_LS_poly.load_state_dict(torch.load(parent_path + 'idvd_nmos_poly_regression_model.pth', map_location=torch.device('cpu')))
    IdVd_LS_poly.eval()

    IdVd_scaler_x = joblib.load(parent_path + 'idvd_nmos_scaler_x.pkl')
    IdVd_scaler_y_IV = joblib.load(parent_path + 'idvd_nmos_scaler_iv.pkl')
    IdVdscaler_ls = joblib.load(parent_path + 'idvd_nmos_scaler_ls.pkl')

    x_scaled = IdVd_scaler_x.transform(x_df)
    x_features = polynomial_features(x_scaled, IdVd_config['degree'], IdVd_config['cross_degree'])
    x_scaled_tensor = torch.tensor(x_features, dtype=torch.float32)

    y_ls = IdVd_LS_poly(x_scaled_tensor)

    decoder_input   = IdVdscaler_ls.inverse_transform(y_ls.detach().numpy())
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float32)
    decoder_output  = IdVd_AE.decoder(decoder_input_tensor).detach().numpy()

    IdVd = IdVd_scaler_y_IV.inverse_transform(decoder_output.reshape(1,-1)).reshape((4, -1))

    # IdVg inference section
    parent_path = './models/NMOS/idvg_nmos_3_par_2_curves_log_linear/'
    IdVg_AE = Autoencoder(**IdVg_config)
    IdVg_AE.load_state_dict(torch.load(parent_path + 'idvg_nmos_2_curves_log_linear_scale.pth', map_location=torch.device('cpu')))
    IdVg_AE.eval()

    IdVg_LS_poly = LatentSpacePolyNN(dim_input=IdVg_config['LS_input_f_size'], dim_output=IdVg_config['latent_size'])
    IdVg_LS_poly.load_state_dict(torch.load(parent_path + 'idvg_nmos_poly_regression_model.pth', map_location=torch.device('cpu')))
    IdVg_LS_poly.eval()

    IdVg_scaler_x = joblib.load(parent_path + 'idvg_scaler_x.pkl')
    IdVg_scaler_y_IV = joblib.load(parent_path + 'idvg_scaler_iv_linear.pkl')
    IdVg_scaler_y_IV_log = joblib.load(parent_path + 'idvg_scaler_iv_log.pkl')
    IdVgscaler_ls = joblib.load(parent_path + 'idvg_scaler_ls.pkl')

    x_scaled = IdVg_scaler_x.transform(x_df)
    x_features = polynomial_features(x_scaled, IdVg_config['degree'], IdVg_config['cross_degree'])
    x_scaled_tensor = torch.tensor(x_features, dtype=torch.float32)

    y_ls = IdVg_LS_poly(x_scaled_tensor)

    decoder_input = IdVgscaler_ls.inverse_transform(y_ls.detach().numpy())
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float32)
    decoder_output = IdVg_AE.decoder(decoder_input_tensor).detach().numpy()
    decoder_output = decoder_output.flatten().reshape((2, -1))

    IdVg_linear_scaled, IdVg_log_scaled = decoder_output[0], decoder_output[1]
    IdVg, IdVg_log = IdVg_scaler_y_IV.inverse_transform(IdVg_linear_scaled.reshape(1,-1)).reshape((2, -1)), IdVg_scaler_y_IV_log.inverse_transform(IdVg_log_scaled.reshape(1,-1)).reshape((2, -1))

    return_body = {
        'simulation_data': {
            'Id_Vd': {
                    'Vg': [0.25, 0.5, 0.75, 1.0],
                    'Vd': np.linspace(0.0025, 1, IdVd.shape[1]).tolist(),
                    'Id': IdVd.tolist(),
                },
            'Id_Vg': {
                    'Vg': np.linspace(0.0025, 1, IdVg.shape[1]).tolist(),
                    'Vd': [0.05, 1.0],
                    'Id': IdVg.tolist(),
                    'Id_log': IdVg_log.tolist(),
                }
        },
        'device_params': parameters
    }
    
    return return_body


@MODELS.register()
class NMOS:
    simulation_func = run_AE_sim
    device_params = ['Lg', 'THF', 'XjSD']
    postprocess = get_simulation_data
