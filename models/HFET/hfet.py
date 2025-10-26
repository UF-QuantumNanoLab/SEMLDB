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
    'size_0'              : 900,
    'size_1'              : 450,
    'size_2'              : 225,
    'size_3'              : 56,
    'latent_size'         : 18,
    'LS_input_f_size'     : 6463,
    'degree'              : 12,
    'cross_degree'        : 8,
    }

IdVg_config = {
    'size_0'              : 1784,
    'size_1'              : 892,
    'size_2'              : 446,
    'size_3'              : 112,
    'latent_size'         : 20,
    'LS_input_f_size'     : 841,
    'degree'              : 12,
    'cross_degree'        : 5,
    }

IgVg_config = {
    'size_0'              : 1784,
    'size_1'              : 892,
    'size_2'              : 446,
    'size_3'              : 112,
    'latent_size'         : 18,
    'LS_input_f_size'     : 834,
    'degree'              : 11,
    'cross_degree'        : 5,
}

BV_config = {
    'size_0'              : 1000,
    'size_1'              : 500,
    'size_2'              : 250,
    'size_3'              : 62,
    'latent_size'         : 12,
    'LS_input_f_size'     : 841,
    'degree'              : 12,
    'cross_degree'        : 5,
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
        'HFET', parameters
    )
    
    if not complete_data:
        return None, False, None, None
    
    adjusted_data = {
        'simulation_data': complete_data.get('simulation_data', {}),
        'device_params': complete_data.get('device_params', {})
    }
    
    return adjusted_data, exact_match, distance, matched_params

def run_AE_sim(parameters):
    Lsg, Lgd, Lg, hpas, hAlGaN, hch, hg = parameters.get('Lsg'), parameters.get('Lgd'), parameters.get('Lg'), parameters.get('hpas'), parameters.get('hAlGaN'), parameters.get('hch'), parameters.get('hg')

    feature_names   = ['Lsg', 'Lgd', 'Lg', 'hpas', 'hAlGaN', 'hch', 'hg']
    x = [Lsg, Lgd, Lg, hpas, hAlGaN, hch, hg]
    x_df = pd.DataFrame([x], columns=feature_names)

    if (hch > 0.019 * hch / hAlGaN - 0.002):
        # IdVd inference section
        model_dir = os.path.dirname(__file__)
        parent_path = os.path.join(model_dir, 'IdVd/')
        IdVd_AE = Autoencoder(**IdVd_config)
        IdVd_AE.load_state_dict(torch.load(os.path.join(parent_path, 'idvd_3_curves_linear_scale_hch_hAlGaN_filter.pth'), map_location=torch.device('cpu')))
        IdVd_AE.eval()

        IdVd_LS_poly = LatentSpacePolyNN(dim_input=IdVd_config['LS_input_f_size'], dim_output=IdVd_config['latent_size'])
        IdVd_LS_poly.load_state_dict(torch.load(os.path.join(parent_path, 'idvd_3_curves_linear_scale_poly_hch_hAlGaN_filter.pth'), map_location=torch.device('cpu')))
        IdVd_LS_poly.eval()

        IdVd_scaler_x = joblib.load(os.path.join(parent_path, 'scaler_x.pkl'))
        IdVd_scaler_y_IV = joblib.load(os.path.join(parent_path, 'scaler_iv.pkl'))
        IdVdscaler_ls = joblib.load(os.path.join(parent_path, 'scaler_ls.pkl'))

        x_scaled = IdVd_scaler_x.transform(x_df)
        x_features = polynomial_features(x_scaled, IdVd_config['degree'], IdVd_config['cross_degree'])
        x_scaled_tensor = torch.tensor(x_features, dtype=torch.float32)

        y_ls = IdVd_LS_poly(x_scaled_tensor)

        decoder_input   = IdVdscaler_ls.inverse_transform(y_ls.detach().numpy())
        decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float32)
        decoder_output  = IdVd_AE.decoder(decoder_input_tensor).detach().numpy()

        IdVd = IdVd_scaler_y_IV.inverse_transform(decoder_output.reshape(1,-1)).reshape((3, -1))

    else:
        IdVd = np.zeros((3, 300))

    # IdVg inference section
    model_dir = os.path.dirname(__file__)
    parent_path = os.path.join(model_dir, 'IdVg/')
    IdVg_AE = Autoencoder(**IdVg_config)
    IdVg_AE.load_state_dict(torch.load(os.path.join(parent_path, 'idvg_2_curves_log_linear_scale.pth'), map_location=torch.device('cpu')))
    IdVg_AE.eval()

    IdVg_LS_poly = LatentSpacePolyNN(dim_input=IdVg_config['LS_input_f_size'], dim_output=IdVg_config['latent_size'])
    IdVg_LS_poly.load_state_dict(torch.load(os.path.join(parent_path, 'poly_regression_model.pth'), map_location=torch.device('cpu')))
    IdVg_LS_poly.eval()

    IdVg_scaler_x = joblib.load(os.path.join(parent_path, 'scaler_x.pkl'))
    IdVg_scaler_y_IV = joblib.load(os.path.join(parent_path, 'scaler_iv_linear.pkl'))
    IdVg_scaler_y_IV_log = joblib.load(os.path.join(parent_path, 'scaler_iv_log.pkl'))
    IdVgscaler_ls = joblib.load(os.path.join(parent_path, 'scaler_ls.pkl'))

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

    # IgVg inference section
    model_dir = os.path.dirname(__file__)
    parent_path = os.path.join(model_dir, 'IgVg/')
    IgVg_AE = Autoencoder(**IgVg_config)
    IgVg_AE.load_state_dict(torch.load(os.path.join(parent_path, 'igvg_2_curves_log_linear_scale.pth'), map_location=torch.device('cpu')))
    IgVg_AE.eval()

    IgVg_LS_poly = LatentSpacePolyNN(dim_input=IgVg_config['LS_input_f_size'], dim_output=IgVg_config['latent_size'])
    IgVg_LS_poly.load_state_dict(torch.load(os.path.join(parent_path, 'poly_regression_model.pth'), map_location=torch.device('cpu')))
    IgVg_LS_poly.eval()

    IgVg_scaler_x = joblib.load(os.path.join(parent_path, 'scaler_x.pkl'))
    IgVg_scaler_y_IV = joblib.load(os.path.join(parent_path, 'scaler_iv_linear.pkl'))
    IgVg_scaler_y_IV_log = joblib.load(os.path.join(parent_path, 'scaler_iv_log.pkl'))
    IgVgscaler_ls = joblib.load(os.path.join(parent_path, 'scaler_ls.pkl'))

    x_scaled = IgVg_scaler_x.transform(x_df)
    x_features = polynomial_features(x_scaled, IgVg_config['degree'], IgVg_config['cross_degree'])
    x_scaled_tensor = torch.tensor(x_features, dtype=torch.float32)

    y_ls = IgVg_LS_poly(x_scaled_tensor)

    decoder_input = IgVgscaler_ls.inverse_transform(y_ls.detach().numpy())
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float32)
    decoder_output = IgVg_AE.decoder(decoder_input_tensor).detach().numpy()
    decoder_output = decoder_output.flatten().reshape((2, -1))

    IgVg_linear_scaled, IgVg_log_scaled = decoder_output[0], decoder_output[1]
    IgVg, IgVg_log = IgVg_scaler_y_IV.inverse_transform(IgVg_linear_scaled.reshape(1,-1)).reshape((2, -1)), IgVg_scaler_y_IV_log.inverse_transform(IgVg_log_scaled.reshape(1,-1)).reshape((2, -1))

    # BV curve section
    model_dir = os.path.dirname(__file__)
    parent_path = os.path.join(model_dir, 'BV/')
    BV_AE = Autoencoder(**BV_config)
    BV_AE.load_state_dict(torch.load(os.path.join(parent_path, 'bv_curve_log_linear_scale.pth'), map_location=torch.device('cpu')))
    BV_AE.eval()

    BV_LS_poly = LatentSpacePolyNN(dim_input=BV_config['LS_input_f_size'], dim_output=BV_config['latent_size'])
    BV_LS_poly.load_state_dict(torch.load(os.path.join(parent_path, 'bv_poly_regression_model.pth'), map_location=torch.device('cpu')))
    BV_LS_poly.eval()

    BV_scaler_x = joblib.load(os.path.join(parent_path, 'scaler_bv_x.pkl'))
    BV_scaler_y_IV_log = joblib.load(os.path.join(parent_path, 'scaler_bv_log.pkl'))
    BVscaler_ls = joblib.load(os.path.join(parent_path, 'scaler_bv_ls.pkl'))

    x_scaled = BV_scaler_x.transform(x_df)
    x_features = polynomial_features(x_scaled, BV_config['degree'], BV_config['cross_degree'])
    x_scaled_tensor = torch.tensor(x_features, dtype=torch.float32)

    y_ls = BV_LS_poly(x_scaled_tensor)

    decoder_input = BVscaler_ls.inverse_transform(y_ls.detach().numpy())
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.float32)
    decoder_output = BV_AE.decoder(decoder_input_tensor).detach().numpy()
    decoder_output = decoder_output.flatten().reshape((2, -1))

    BV_linear_scaled, BV_log_scaled = decoder_output[0], decoder_output[1]
    IdVg_BV_log = BV_scaler_y_IV_log.inverse_transform(BV_log_scaled.reshape(1,-1)).flatten()

    return_body = {
        'simulation_data': {
            'Id_Vd': {
                    'Vg': [1.0, 3.0, 5.0],
                    'Vd': np.linspace(0.02, 18, IdVd.shape[1]).tolist(),
                    'Id': IdVd.tolist(),
                },
            'Id_Vg': {
                    'Vg': np.linspace(-1.9, 7, IdVg.shape[1]).tolist(),
                    'Vd': [1.0, 15.0],
                    'Id': IdVg.tolist(),
                    'Id_log': IdVg_log.tolist(),
                },
            'Ig_Vg': {
                    'Vg': np.linspace(-1.9, 7, IdVg.shape[1]).tolist(),
                    'Vd': [1.0, 15.0],
                    'Ig': IgVg.tolist(),
                    'Ig_log': IgVg_log.tolist(),
                },
            'Id_Vd_BVOutput': {
                    'Vd': np.linspace(2, 1000, IdVg_BV_log.shape[0]).tolist(),
                    'Id_log': IdVg_BV_log.tolist(),
            }
        },
        'device_params': parameters
    }
    
    return return_body


@MODELS.register()
class HFET:
    simulation_func = run_AE_sim
    device_params = ['Lsg', 'Lgd', 'Lg', 'hpas', 'hAlGaN', 'hch', 'hg']
    voltage_params = None
    postprocess = get_simulation_data
