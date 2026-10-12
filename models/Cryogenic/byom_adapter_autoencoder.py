#!/usr/bin/env python3
"""
byom_adapter_autoencoder.py
---------------------------
Bring-Your-Own-Model (BYOM) adapter module containing two adapters:

1. AutoencoderAdapter
   Autoencoder (on y) + EncoderMLP that maps raw X → latent space.
   Architecture:
     fit:     y ──AE──► reconstruct y   |   X ──MLP──► Z_scaled ──decoder──► y
     predict: X ──MLP──► Z_scaled ──inv_ls──► Z ──decoder──► y_scaled ──inv_y──► y
   Artifacts: autoencoder_model.pth, encoder_mlp_model.pth,
              scaler_ls.pkl, scaler_y.pkl, ae_ext_params.json

2. DecoderAdapter
   Autoencoder (on y, only decoder used at inference) + Ridge regression
   that maps raw X → latent space. Lighter-weight alternative with no
   MLP training for the X→Z step.
   Architecture:
     fit:     y ──AE──► reconstruct y   |   X ──Ridge──► Z_scaled ──decoder──► y
     predict: X ──Ridge──► Z_scaled ──inv_ls──► Z ──decoder──► y_scaled ──inv_y──► y
   Artifacts: autoencoder_model.pth, ridge_model.pkl,
              scaler_ls.pkl, scaler_y.pkl, decoder_ext_params.json,
              decoder_adapter.joblib (bundle)

Both adapters:
  • Search over latent_dims listed in model_config.
  • Select the best model by highest mean R² across outputs
    (tie-broken by lowest RMSE, then lowest MSE) on the full X→y chain.

model_config keys (all optional, with defaults shown):
  latent_dims  (str | list) : latent sizes to search  [default: "2,3,4"]
  alpha        (float)      : Ridge alpha (DecoderAdapter only) [default: 1.0]
  epochs       (int)        : AE training epochs       [default: 50]
  lr           (float)      : Adam learning rate       [default: 1e-3]
  batch_size   (int)        : DataLoader batch size    [default: 32]

Pipeline entry point:
  get_model(..., adapter_type="autoencoder")  → AutoencoderAdapter (default)
  get_model(..., adapter_type="decoder")      → DecoderAdapter
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None  # type: ignore

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Neural network modules
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """Symmetric autoencoder on the output space y."""

    def __init__(self, size_0: int, size_1: int, size_2: int, size_3: int, latent_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(size_0, size_1), nn.ReLU(),
            nn.Linear(size_1, size_2), nn.ReLU(),
            nn.Linear(size_2, size_3), nn.ReLU(),
            nn.Linear(size_3, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, size_3), nn.ReLU(),
            nn.Linear(size_3, size_2), nn.ReLU(),
            nn.Linear(size_2, size_1), nn.ReLU(),
            nn.Linear(size_1, size_0), nn.Sigmoid(),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.decoder(self.encoder(x))

    def get_latent(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.encoder(x)


class EncoderMLP(nn.Module):
    """Small MLP: raw X features → latent space (replaces poly mapping)."""

    def __init__(self, n_input: int, n_hidden: int, n_latent: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_latent),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class AutoencoderAdapter:
    """
    BYOM adapter: Autoencoder (on y) + EncoderMLP (X → Z_scaled).
    Best latent dimension selected by mean R² on the full X → y chain.
    """

    def __init__(
        self,
        artifacts_dir: Optional[str] = None,
        model_config_path: Optional[str] = None,
        pretrained_model_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for byom_adapter_autoencoder. "
                "Install it with:  pip install torch"
            )

        self.artifacts_dir = (
            Path(artifacts_dir).expanduser().resolve() if artifacts_dir else None
        )
        self.model_config_path = (
            Path(model_config_path).expanduser().resolve() if model_config_path else None
        )
        self.pretrained_model_path = (
            Path(pretrained_model_path).expanduser().resolve()
            if pretrained_model_path else None
        )
        self.context = context or {}

        self.feature_set: List[str] = list(self.context.get("feature_set", []))
        self.label_set:   List[str] = list(self.context.get("label_set", []))

        self.model_config: Dict[str, Any] = {}
        self.mode = (
            str(
                (self.context.get("provider_cfg", {}) or {}).get(
                    "mode", "load_existing"
                ) or "load_existing"
            )
            .strip()
            .lower()
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model state
        self.best_autoencoder: Optional[Autoencoder] = None
        self.best_encoder_mlp: Optional[EncoderMLP]  = None
        self.ls_scaler: Optional[MinMaxScaler] = None
        self.y_scaler:  Optional[MinMaxScaler] = None

        self.best_latent_dim: Optional[int]   = None
        self.best_r2:         Optional[float] = None
        self.best_rmse:       Optional[float] = None
        self.best_mse:        Optional[float] = None

        self._load_artifacts()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_dir(self) -> Optional[Path]:
        if self.pretrained_model_path and self.pretrained_model_path.is_dir():
            return self.pretrained_model_path
        return self.artifacts_dir

    def _load_artifacts(self) -> None:
        if self.model_config_path and self.model_config_path.exists():
            with open(self.model_config_path, "r", encoding="utf-8") as f:
                self.model_config = json.load(f)

        base_dir = self._base_dir()
        if base_dir is None:
            return

        pth_ae    = base_dir / "autoencoder_model.pth"
        pth_mlp   = base_dir / "encoder_mlp_model.pth"
        pkl_ls    = base_dir / "scaler_ls.pkl"
        pkl_y     = base_dir / "scaler_y.pkl"
        params_p  = base_dir / "ae_ext_params.json"

        if pth_ae.exists() and pth_mlp.exists() and pkl_ls.exists() and pkl_y.exists():
            self.ls_scaler = joblib.load(pkl_ls)
            self.y_scaler  = joblib.load(pkl_y)

            # Reconstruct exact architecture from the saved state_dict weights
            # (avoids mismatch when scaler dims differ from checkpoint dims)
            sd_ae  = torch.load(pth_ae, map_location=self.device)
            size_1, size_0 = sd_ae["encoder.0.weight"].shape
            size_2 = sd_ae["encoder.2.weight"].shape[0]
            size_3 = sd_ae["encoder.4.weight"].shape[0]
            z_dim  = sd_ae["encoder.6.weight"].shape[0]

            self.best_autoencoder = Autoencoder(size_0, size_1, size_2, size_3, z_dim).to(self.device)
            self.best_autoencoder.load_state_dict(sd_ae)
            self.best_autoencoder.eval()

            # Reconstruct EncoderMLP from saved state_dict shapes
            sd_mlp = torch.load(pth_mlp, map_location=self.device)
            n_input  = sd_mlp["net.0.weight"].shape[1]
            n_hidden = sd_mlp["net.0.weight"].shape[0]
            n_latent = sd_mlp["net.4.weight"].shape[0]
            self.best_encoder_mlp = EncoderMLP(n_input, n_hidden, n_latent).to(self.device)
            self.best_encoder_mlp.load_state_dict(sd_mlp)
            self.best_encoder_mlp.eval()

            print(f"AutoencoderAdapter: Loaded pre-trained artifacts from {base_dir}")

        if params_p.exists():
            with open(params_p, "r") as f:
                params = json.load(f)
            self.best_latent_dim = params.get("latent_dim")
            self.best_r2         = params.get("best_r2")
            self.best_rmse       = params.get("best_rmse")
            self.best_mse        = params.get("best_mse")

    # ------------------------------------------------------------------
    # BYOM contract methods
    # ------------------------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "AutoencoderAdapter",
            "provider": "custom_python",
            "mode": self.mode,
            "feature_set": self.feature_set,
            "label_set": self.label_set,
        }

    def validate(
        self,
        feature_set: Optional[Iterable[str]] = None,
        label_set: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        feature_set = list(feature_set) if feature_set is not None else self.feature_set
        label_set   = list(label_set)   if label_set   is not None else self.label_set
        if not feature_set:
            raise ValueError(
                "feature_set is empty. Configure machine_learning_surrogate.feature_set."
            )
        if not label_set:
            raise ValueError(
                "label_set is empty. Configure machine_learning_surrogate.label_set."
            )
        return {
            "validated": True,
            "n_features": len(feature_set),
            "n_labels": len(label_set),
        }

    def fit(self, X: Any, y: Any, context: Optional[Dict[str, Any]] = None) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # --- Hyper-parameters ---
        raw_dims = self.model_config.get("latent_dims", "2,3,4")
        if isinstance(raw_dims, str):
            latent_dims = [
                int(v.strip()) for v in raw_dims.split(",")
                if v.strip() and v.strip().isdigit()
            ]
        elif isinstance(raw_dims, list):
            latent_dims = [int(v) for v in raw_dims]
        else:
            latent_dims = [2, 3, 4]

        epochs     = int(self.model_config.get("epochs", 50))
        lr         = float(self.model_config.get("lr", 1e-3))
        batch_size = int(self.model_config.get("batch_size", 32))

        # --- 1. Scale y → [0, 1] (Sigmoid decoder) ---
        self.y_scaler = MinMaxScaler()
        y_scaled  = self.y_scaler.fit_transform(y)
        y_tensor  = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
        X_tensor  = torch.tensor(X, dtype=torch.float32).to(self.device)

        ae_dataset = TensorDataset(y_tensor, y_tensor)
        ae_loader  = DataLoader(ae_dataset, batch_size=batch_size, shuffle=True)

        size_0 = y.shape[1]
        size_1 = max(1, round(size_0 / 2))
        size_2 = max(1, round(size_1 / 2))
        size_3 = max(1, round(size_2 / 4))
        n_hidden = max(16, X.shape[1] * 4)

        print(
            f"AutoencoderAdapter: Starting latent-dim grid search — "
            f"candidates={latent_dims}, epochs={epochs}, lr={lr}, batch={batch_size}"
        )
        print(f"  {'z_dim':>6}  {'AE MSE':>10}  {'R2 (mean)':>12}  {'RMSE':>12}  {'MSE':>12}")
        print("  " + "-" * 58)

        best_r2    = -np.inf
        best_rmse  =  np.inf
        best_mse   =  np.inf
        best_ae    = None
        best_mlp   = None
        best_ls_sc = None
        best_z_dim = latent_dims[0]

        for z_dim in latent_dims:
            # --- 2. Train Autoencoder for this z_dim ---
            ae = Autoencoder(size_0, size_1, size_2, size_3, z_dim).to(self.device)
            ae_opt  = optim.Adam(ae.parameters(), lr=lr)
            ae_crit = nn.MSELoss()

            ae.train()
            for _ in range(epochs):
                for b_in, b_out in ae_loader:
                    ae_opt.zero_grad()
                    loss = ae_crit(ae(b_in), b_out)
                    loss.backward()
                    ae_opt.step()

            ae.eval()
            with torch.no_grad():
                y_recon = ae(y_tensor).cpu().numpy()
            ae_mse = float(mean_squared_error(y_scaled, y_recon))

            # --- 3. Extract and scale latent targets ---
            with torch.no_grad():
                Z = ae.encoder(y_tensor).cpu().numpy()
            ls_scaler = MinMaxScaler()
            Z_scaled  = ls_scaler.fit_transform(Z)

            Z_scaled_tensor = torch.tensor(Z_scaled, dtype=torch.float32).to(self.device)
            map_dataset = TensorDataset(X_tensor, Z_scaled_tensor)
            map_loader  = DataLoader(map_dataset, batch_size=batch_size, shuffle=True)

            # --- 4. Train EncoderMLP (X → Z_scaled) ---
            mlp = EncoderMLP(X.shape[1], n_hidden, z_dim).to(self.device)
            mlp_opt  = optim.Adam(mlp.parameters(), lr=lr)
            mlp_crit = nn.MSELoss()

            mlp.train()
            for _ in range(epochs):
                for bx, bz in map_loader:
                    mlp_opt.zero_grad()
                    loss = mlp_crit(mlp(bx), bz)
                    loss.backward()
                    mlp_opt.step()

            # --- 5. Evaluate full X → y chain ---
            mlp.eval()
            ae.eval()
            with torch.no_grad():
                Z_pred_scaled   = mlp(X_tensor).cpu().numpy()
                Z_pred          = ls_scaler.inverse_transform(Z_pred_scaled)
                Z_pred_tensor   = torch.tensor(Z_pred, dtype=torch.float32).to(self.device)
                y_pred_scaled   = ae.decoder(Z_pred_tensor).cpu().numpy()

            mse  = float(mean_squared_error(y_scaled, y_pred_scaled))
            rmse = float(np.sqrt(mse))
            r2   = float(np.mean(r2_score(y_scaled, y_pred_scaled, multioutput="raw_values")))

            print(
                f"  {z_dim:>6}  {ae_mse:>10.6f}  {r2:>12.6f}  {rmse:>12.6f}  {mse:>12.6f}"
            )

            # --- 6. Track best by R², tie-break RMSE ---
            if r2 > best_r2 or (r2 == best_r2 and rmse < best_rmse):
                best_r2    = r2
                best_rmse  = rmse
                best_mse   = mse
                best_z_dim = z_dim
                best_ae    = ae
                best_mlp   = mlp
                best_ls_sc = ls_scaler

        print("  " + "-" * 58)
        print(
            f"AutoencoderAdapter: Best → z_dim={best_z_dim}  "
            f"| R²={best_r2:.6f}  RMSE={best_rmse:.6f}  MSE={best_mse:.6f}"
        )

        self.best_autoencoder = best_ae
        self.best_encoder_mlp = best_mlp
        self.ls_scaler        = best_ls_sc
        self.best_latent_dim  = best_z_dim
        self.best_r2          = best_r2
        self.best_rmse        = best_rmse
        self.best_mse         = best_mse

    def save(
        self, output_dir: Optional[str] = None, context: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.best_autoencoder is None or self.best_encoder_mlp is None:
            raise ValueError("Cannot save: model is not trained or loaded yet.")

        save_dir = self.artifacts_dir
        if save_dir is None:
            if output_dir:
                save_dir = Path(str(output_dir)).expanduser().resolve()
            else:
                raise ValueError(
                    "Cannot save: provide artifacts_dir at construction time "
                    "or pass output_dir to save()."
                )
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.best_autoencoder.state_dict(), save_dir / "autoencoder_model.pth")
        torch.save(self.best_encoder_mlp.state_dict(), save_dir / "encoder_mlp_model.pth")
        joblib.dump(self.ls_scaler, save_dir / "scaler_ls.pkl")
        joblib.dump(self.y_scaler,  save_dir / "scaler_y.pkl")

        params = {
            "latent_dim": self.best_latent_dim,
            "best_r2":    self.best_r2,
            "best_rmse":  self.best_rmse,
            "best_mse":   self.best_mse,
        }
        with open(save_dir / "ae_ext_params.json", "w") as f:
            json.dump(params, f, indent=2)

        # Save self-contained encoder_only/ and decoder_only/ subdirectories.
        # Each contains the weights, scaler, a copy of this adapter file, and a
        # model_manifest.json so the ML Data tab needs no further configuration.
        #
        # Use __code__.co_filename instead of inspect.getfile() — the adapter is
        # loaded dynamically via importlib and is not registered in sys.modules,
        # so inspect.getfile() raises OSError in that context.
        import shutil as _shutil
        try:
            _this_file = Path(type(self).__init__.__code__.co_filename).resolve()
        except Exception:
            _this_file = Path(__file__).resolve()
        for sub, entry in (("encoder_only", "get_encoder_model"),
                           ("decoder_only", "get_decoder_model")):
            sub_dir = save_dir / sub
            sub_dir.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(save_dir / "autoencoder_model.pth", sub_dir / "autoencoder_model.pth")
            _shutil.copy2(save_dir / "scaler_y.pkl",          sub_dir / "scaler_y.pkl")
            if _this_file.exists():
                _shutil.copy2(str(_this_file), str(sub_dir / _this_file.name))
            with open(sub_dir / "model_manifest.json", "w") as _f:
                json.dump({
                    "entry_point":  entry,
                    "adapter_file": _this_file.name,
                }, _f, indent=2)

        print(f"AutoencoderAdapter: Artifacts saved to {save_dir}")
        print(f"AutoencoderAdapter: Encoder-only folder → {save_dir / 'encoder_only'}")
        print(f"AutoencoderAdapter: Decoder-only folder → {save_dir / 'decoder_only'}")

    def predict(self, X: Any) -> np.ndarray:
        if self.best_autoencoder is None or self.best_encoder_mlp is None:
            raise ValueError(
                "Model is not available. Train first or provide existing artifacts path."
            )

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.best_encoder_mlp.eval()
        self.best_autoencoder.eval()
        with torch.no_grad():
            Z_pred_scaled = self.best_encoder_mlp(X_tensor).cpu().numpy()
            Z_pred        = self.ls_scaler.inverse_transform(Z_pred_scaled)
            Z_pred_tensor = torch.tensor(Z_pred, dtype=torch.float32).to(self.device)
            y_pred_scaled = self.best_autoencoder.decoder(Z_pred_tensor).cpu().numpy()

        Y_pred = self.y_scaler.inverse_transform(y_pred_scaled)

        n_samples = X.shape[0]
        n_labels  = len(self.label_set)
        if Y_pred.shape != (n_samples, n_labels):
            raise ValueError(
                f"predict() must return shape {(n_samples, n_labels)}, "
                f"but got {Y_pred.shape}. "
                "Make sure your dataset outputs match label_set."
            )

        return Y_pred

    def _assert_loaded(self) -> None:
        if self.best_autoencoder is None or self.y_scaler is None:
            base = self._base_dir()
            raise ValueError(
                "AutoencoderAdapter: model artifacts not loaded.\n"
                f"  artifacts_dir          = {self.artifacts_dir}\n"
                f"  pretrained_model_path  = {self.pretrained_model_path}\n"
                f"  resolved base_dir      = {base}\n"
                "  Expected files in base_dir: autoencoder_model.pth, "
                "encoder_mlp_model.pth, scaler_ls.pkl, scaler_y.pkl\n"
                "  Set 'artifacts_dir' or 'pretrained_model_path' (must be a directory) "
                "in the Latent Bridge → AE Model section of the GUI."
            )

    def encode(self, y: Any) -> np.ndarray:
        """Encode y → raw latent Z via the autoencoder encoder.

        Required by the Latent Bridge pipeline stage.
        Returns Z in the same space the decoder expects (before ls_scaler).
        """
        self._assert_loaded()
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        y_scaled = self.y_scaler.transform(y)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
        self.best_autoencoder.eval()
        with torch.no_grad():
            Z = self.best_autoencoder.encoder(y_tensor).cpu().numpy()
        return Z

    def decode(self, Z: Any) -> np.ndarray:
        """Decode raw latent Z → y via the autoencoder decoder.

        Required by the Latent Bridge pipeline stage.
        Accepts the same Z space returned by encode().
        """
        self._assert_loaded()
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        Z_tensor = torch.tensor(Z, dtype=torch.float32).to(self.device)
        self.best_autoencoder.eval()
        with torch.no_grad():
            y_pred_scaled = self.best_autoencoder.decoder(Z_tensor).cpu().numpy()
        return self.y_scaler.inverse_transform(y_pred_scaled)


# ---------------------------------------------------------------------------
# DecoderAdapter  (Autoencoder decoder  +  Ridge regression  X → Z)
# ---------------------------------------------------------------------------

class DecoderAdapter:
    """
    BYOM adapter: Autoencoder decoder (on y) + Ridge regression (X → Z_scaled).
    Only the decoder half of the autoencoder is used at inference time.
    Best latent dimension selected by mean R² on the full X → y chain.
    """

    def __init__(
        self,
        artifacts_dir: Optional[str] = None,
        model_config_path: Optional[str] = None,
        pretrained_model_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for DecoderAdapter. "
                "Install it with:  pip install torch"
            )

        self.artifacts_dir = (
            Path(artifacts_dir).expanduser().resolve() if artifacts_dir else None
        )
        self.model_config_path = (
            Path(model_config_path).expanduser().resolve() if model_config_path else None
        )
        self.pretrained_model_path = (
            Path(pretrained_model_path).expanduser().resolve()
            if pretrained_model_path else None
        )
        self.context = context or {}

        self.feature_set: List[str] = list(self.context.get("feature_set", []))
        self.label_set:   List[str] = list(self.context.get("label_set", []))

        self.model_config: Dict[str, Any] = {}
        self.mode = (
            str(
                (self.context.get("provider_cfg", {}) or {}).get(
                    "mode", "load_existing"
                ) or "load_existing"
            )
            .strip()
            .lower()
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.best_autoencoder: Optional[Autoencoder] = None
        self.ridge_model:      Optional[Ridge]        = None
        self.ls_scaler:        Optional[MinMaxScaler] = None
        self.y_scaler:         Optional[MinMaxScaler] = None

        self.best_latent_dim: Optional[int]   = None
        self.best_r2:         Optional[float] = None
        self.best_rmse:       Optional[float] = None
        self.best_mse:        Optional[float] = None

        self._load_artifacts()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_dir(self) -> Optional[Path]:
        if self.pretrained_model_path and self.pretrained_model_path.is_dir():
            return self.pretrained_model_path
        return self.artifacts_dir

    def _load_artifacts(self) -> None:
        if self.model_config_path and self.model_config_path.exists():
            with open(self.model_config_path, "r", encoding="utf-8") as f:
                self.model_config = json.load(f)

        base_dir = self._base_dir()
        if base_dir is None:
            return

        pth_ae   = base_dir / "autoencoder_model.pth"
        pkl_r    = base_dir / "ridge_model.pkl"
        pkl_ls   = base_dir / "scaler_ls.pkl"
        pkl_y    = base_dir / "scaler_y.pkl"
        params_p = base_dir / "decoder_ext_params.json"

        if pth_ae.exists() and pkl_r.exists() and pkl_ls.exists() and pkl_y.exists():
            self.ridge_model = joblib.load(pkl_r)
            self.ls_scaler   = joblib.load(pkl_ls)
            self.y_scaler    = joblib.load(pkl_y)

            sd_ae  = torch.load(pth_ae, map_location=self.device)
            size_1, size_0 = sd_ae["encoder.0.weight"].shape
            size_2 = sd_ae["encoder.2.weight"].shape[0]
            size_3 = sd_ae["encoder.4.weight"].shape[0]
            z_dim  = sd_ae["encoder.6.weight"].shape[0]

            self.best_autoencoder = Autoencoder(
                size_0, size_1, size_2, size_3, z_dim
            ).to(self.device)
            self.best_autoencoder.load_state_dict(sd_ae)
            self.best_autoencoder.eval()
            print(f"DecoderAdapter: Loaded pre-trained artifacts from {base_dir}")

        if params_p.exists():
            with open(params_p, "r") as f:
                params = json.load(f)
            self.best_latent_dim = params.get("latent_dim")
            self.best_r2         = params.get("best_r2")
            self.best_rmse       = params.get("best_rmse")
            self.best_mse        = params.get("best_mse")

    # ------------------------------------------------------------------
    # BYOM contract methods
    # ------------------------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "DecoderAdapter",
            "provider": "custom_python",
            "mode": self.mode,
            "feature_set": self.feature_set,
            "label_set": self.label_set,
        }

    def validate(
        self,
        feature_set: Optional[Iterable[str]] = None,
        label_set: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        feature_set = list(feature_set) if feature_set is not None else self.feature_set
        label_set   = list(label_set)   if label_set   is not None else self.label_set
        if not feature_set:
            raise ValueError(
                "feature_set is empty. Configure machine_learning_surrogate.feature_set."
            )
        if not label_set:
            raise ValueError(
                "label_set is empty. Configure machine_learning_surrogate.label_set."
            )
        return {"validated": True, "n_features": len(feature_set), "n_labels": len(label_set)}

    def fit(self, X: Any, y: Any, context: Optional[Dict[str, Any]] = None) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        raw_dims = self.model_config.get("latent_dims", "2,3,4")
        if isinstance(raw_dims, str):
            latent_dims = [int(v.strip()) for v in raw_dims.split(",")
                           if v.strip() and v.strip().isdigit()]
        elif isinstance(raw_dims, list):
            latent_dims = [int(v) for v in raw_dims]
        else:
            latent_dims = [2, 3, 4]

        alpha      = float(self.model_config.get("alpha", 1.0))
        epochs     = int(self.model_config.get("epochs", 50))
        lr         = float(self.model_config.get("lr", 1e-3))
        batch_size = int(self.model_config.get("batch_size", 32))

        self.y_scaler = MinMaxScaler()
        y_scaled = self.y_scaler.fit_transform(y)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)

        ae_dataset = TensorDataset(y_tensor, y_tensor)
        ae_loader  = DataLoader(ae_dataset, batch_size=batch_size, shuffle=True)

        size_0 = y.shape[1]
        size_1 = max(1, round(size_0 / 2))
        size_2 = max(1, round(size_1 / 2))
        size_3 = max(1, round(size_2 / 4))

        print(
            f"DecoderAdapter: Starting latent-dim grid search — "
            f"candidates={latent_dims}, alpha={alpha}, epochs={epochs}, "
            f"lr={lr}, batch={batch_size}"
        )
        print(f"  {'z_dim':>6}  {'AE MSE':>10}  {'R2 (mean)':>12}  {'RMSE':>12}  {'MSE':>12}")
        print("  " + "-" * 58)

        best_r2 = -np.inf; best_rmse = np.inf; best_mse = np.inf
        best_ae = None; best_ridge = None; best_ls_sc = None
        best_z_dim = latent_dims[0]

        for z_dim in latent_dims:
            ae = Autoencoder(size_0, size_1, size_2, size_3, z_dim).to(self.device)
            ae_opt = optim.Adam(ae.parameters(), lr=lr)
            ae_crit = nn.MSELoss()

            ae.train()
            for _ in range(epochs):
                for b_in, b_out in ae_loader:
                    ae_opt.zero_grad()
                    loss = ae_crit(ae(b_in), b_out)
                    loss.backward()
                    ae_opt.step()

            ae.eval()
            with torch.no_grad():
                y_recon = ae(y_tensor).cpu().numpy()
            ae_mse = float(mean_squared_error(y_scaled, y_recon))

            with torch.no_grad():
                Z = ae.encoder(y_tensor).cpu().numpy()
            ls_scaler = MinMaxScaler()
            Z_scaled  = ls_scaler.fit_transform(Z)

            ridge = Ridge(alpha=alpha)
            ridge.fit(X, Z_scaled)
            Z_pred_scaled = ridge.predict(X)

            Z_pred        = ls_scaler.inverse_transform(Z_pred_scaled)
            Z_pred_tensor = torch.tensor(Z_pred, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                y_pred_scaled = ae.decoder(Z_pred_tensor).cpu().numpy()

            mse  = float(mean_squared_error(y_scaled, y_pred_scaled))
            rmse = float(np.sqrt(mse))
            r2   = float(np.mean(r2_score(y_scaled, y_pred_scaled, multioutput="raw_values")))

            print(f"  {z_dim:>6}  {ae_mse:>10.6f}  {r2:>12.6f}  {rmse:>12.6f}  {mse:>12.6f}")

            if (
                r2 > best_r2
                or (r2 == best_r2 and rmse < best_rmse)
                or (r2 == best_r2 and rmse == best_rmse and mse < best_mse)
            ):
                best_r2 = r2; best_rmse = rmse; best_mse = mse
                best_z_dim = z_dim; best_ae = ae
                best_ridge = ridge; best_ls_sc = ls_scaler

        print("  " + "-" * 58)
        print(
            f"DecoderAdapter: Best → z_dim={best_z_dim}  "
            f"| R²={best_r2:.6f}  RMSE={best_rmse:.6f}  MSE={best_mse:.6f}"
        )

        self.best_autoencoder = best_ae
        self.ridge_model      = best_ridge
        self.ls_scaler        = best_ls_sc
        self.best_latent_dim  = best_z_dim
        self.best_r2 = best_r2; self.best_rmse = best_rmse; self.best_mse = best_mse

    def save(
        self, output_dir: Optional[str] = None, context: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.best_autoencoder is None or self.ridge_model is None:
            raise ValueError("Cannot save: model is not trained or loaded yet.")
        if self.artifacts_dir is None:
            raise ValueError(
                "Cannot save: artifacts_dir was not set at construction time."
            )
        save_dir = self.artifacts_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.best_autoencoder.state_dict(), save_dir / "autoencoder_model.pth")
        joblib.dump(self.ridge_model, save_dir / "ridge_model.pkl")
        joblib.dump(self.ls_scaler,   save_dir / "scaler_ls.pkl")
        joblib.dump(self.y_scaler,    save_dir / "scaler_y.pkl")

        params = {
            "latent_dim": self.best_latent_dim,
            "best_r2":    self.best_r2,
            "best_rmse":  self.best_rmse,
            "best_mse":   self.best_mse,
        }
        with open(save_dir / "decoder_ext_params.json", "w") as f:
            json.dump(params, f, indent=2)

        bundle = {
            "autoencoder_state": self.best_autoencoder.state_dict(),
            "ridge_model": self.ridge_model,
            "ls_scaler":   self.ls_scaler,
            "y_scaler":    self.y_scaler,
            "params":      params,
        }
        joblib.dump(bundle, save_dir / "decoder_adapter.joblib")

        print(f"DecoderAdapter: Artifacts saved to {save_dir}")
        print(f"DecoderAdapter: Bundle → {save_dir / 'decoder_adapter.joblib'}")

    def predict(self, X: Any) -> np.ndarray:
        if self.best_autoencoder is None or self.ridge_model is None:
            raise ValueError(
                "Model is not available. Train first or provide existing artifacts path."
            )
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        Z_pred_scaled = self.ridge_model.predict(X)
        Z_pred        = self.ls_scaler.inverse_transform(Z_pred_scaled)
        Z_pred_tensor = torch.tensor(Z_pred, dtype=torch.float32).to(self.device)

        self.best_autoencoder.eval()
        with torch.no_grad():
            y_pred_scaled = self.best_autoencoder.decoder(Z_pred_tensor).cpu().numpy()

        Y_pred = self.y_scaler.inverse_transform(y_pred_scaled)

        n_samples = X.shape[0]
        n_labels  = len(self.label_set)
        if Y_pred.shape != (n_samples, n_labels):
            raise ValueError(
                f"predict() must return shape {(n_samples, n_labels)}, "
                f"but got {Y_pred.shape}."
            )
        return Y_pred

    def _assert_loaded(self) -> None:
        if self.best_autoencoder is None or self.y_scaler is None:
            base = self._base_dir()
            raise ValueError(
                "DecoderAdapter: model artifacts not loaded.\n"
                f"  artifacts_dir          = {self.artifacts_dir}\n"
                f"  pretrained_model_path  = {self.pretrained_model_path}\n"
                f"  resolved base_dir      = {base}\n"
                "  Expected files in base_dir: autoencoder_model.pth, "
                "ridge_model.pkl, scaler_ls.pkl, scaler_y.pkl\n"
                "  Set 'artifacts_dir' or 'pretrained_model_path' (must be a directory) "
                "in the Latent Bridge → AE Model section of the GUI."
            )

    def encode(self, y: Any) -> np.ndarray:
        """Encode y → raw latent Z via the autoencoder encoder.

        Required by the Latent Bridge pipeline stage.
        Returns Z in the same space the decoder expects (before ls_scaler).
        """
        self._assert_loaded()
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        y_scaled = self.y_scaler.transform(y)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
        self.best_autoencoder.eval()
        with torch.no_grad():
            Z = self.best_autoencoder.encoder(y_tensor).cpu().numpy()
        return Z

    def decode(self, Z: Any) -> np.ndarray:
        """Decode raw latent Z → y via the autoencoder decoder.

        Required by the Latent Bridge pipeline stage.
        Accepts the same Z space returned by encode().
        """
        self._assert_loaded()
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        Z_tensor = torch.tensor(Z, dtype=torch.float32).to(self.device)
        self.best_autoencoder.eval()
        with torch.no_grad():
            y_pred_scaled = self.best_autoencoder.decoder(Z_tensor).cpu().numpy()
        return self.y_scaler.inverse_transform(y_pred_scaled)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def get_model(
    artifacts_dir: Optional[str] = None,
    model_config_path: Optional[str] = None,
    pretrained_model_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    adapter_type: str = "autoencoder",
) -> "AutoencoderAdapter | DecoderAdapter":
    """
    Default pipeline entry point for BYOM.

    Parameters
    ----------
    adapter_type : str
        "autoencoder" (default) → AutoencoderAdapter  (AE + EncoderMLP)
        "decoder"               → DecoderAdapter      (AE decoder + Ridge)
    """
    adapter_type = adapter_type.strip().lower()
    if adapter_type == "decoder":
        return DecoderAdapter(
            artifacts_dir=artifacts_dir,
            model_config_path=model_config_path,
            pretrained_model_path=pretrained_model_path,
            context=context,
        )
    return AutoencoderAdapter(
        artifacts_dir=artifacts_dir,
        model_config_path=model_config_path,
        pretrained_model_path=pretrained_model_path,
        context=context,
    )


# ---------------------------------------------------------------------------
# AEEncoderOnlyAdapter  (y → Z,  inference-only)
# ---------------------------------------------------------------------------

class AEEncoderOnlyAdapter:
    """BYOM adapter that exposes only the encoder half of a trained Autoencoder.

    predict(y) → Z

    Loads artifacts from the ``encoder_only/`` subdirectory saved by
    ``AutoencoderAdapter.save()``.  This adapter is inference-only — call
    ``fit()`` on the full ``AutoencoderAdapter`` first, then point this
    adapter at the saved ``encoder_only/`` directory.

    Typical use: ML Data tab — run IV data through the encoder to produce the
    latent-space CSV that trains the bridge / poly-regressor model.

    Entry point for BYOM:  ``get_encoder_model``
    """

    def __init__(
        self,
        artifacts_dir: Optional[str] = None,
        model_config_path: Optional[str] = None,
        pretrained_model_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.artifacts_dir = (
            Path(artifacts_dir).expanduser().resolve() if artifacts_dir else None
        )
        self.model_config_path = (
            Path(model_config_path).expanduser().resolve() if model_config_path else None
        )
        self.pretrained_model_path = (
            Path(pretrained_model_path).expanduser().resolve()
            if pretrained_model_path else None
        )
        self.context = context or {}
        self.feature_set = list(self.context.get("feature_set", []))
        self.label_set   = list(self.context.get("label_set",   []))
        self.model_config: Dict[str, Any] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_autoencoder: Optional[Autoencoder] = None
        self.y_scaler: Optional[MinMaxScaler] = None

        self._load_artifacts()

    def _base_dir(self) -> Optional[Path]:
        if self.pretrained_model_path and self.pretrained_model_path.is_dir():
            return self.pretrained_model_path
        return self.artifacts_dir

    def _load_artifacts(self) -> None:
        if self.model_config_path and self.model_config_path.exists():
            with open(self.model_config_path, "r", encoding="utf-8") as f:
                self.model_config = json.load(f)

        base_dir = self._base_dir()
        if base_dir is None:
            return

        pth_ae = base_dir / "autoencoder_model.pth"
        pkl_y  = base_dir / "scaler_y.pkl"

        if pth_ae.exists() and pkl_y.exists():
            self.y_scaler = joblib.load(pkl_y)
            sd_ae = torch.load(pth_ae, map_location=self.device)
            size_1, size_0 = sd_ae["encoder.0.weight"].shape
            size_2 = sd_ae["encoder.2.weight"].shape[0]
            size_3 = sd_ae["encoder.4.weight"].shape[0]
            z_dim  = sd_ae["encoder.6.weight"].shape[0]
            self.best_autoencoder = Autoencoder(
                size_0, size_1, size_2, size_3, z_dim
            ).to(self.device)
            self.best_autoencoder.load_state_dict(sd_ae)
            self.best_autoencoder.eval()
            print(f"AEEncoderOnlyAdapter: Loaded from {base_dir}")

    def metadata(self) -> Dict[str, Any]:
        return {
            "name":        "AEEncoderOnlyAdapter",
            "provider":    "custom_python",
            "feature_set": self.feature_set,
            "label_set":   self.label_set,
        }

    def validate(
        self,
        feature_set: Optional[Iterable[str]] = None,
        label_set: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        feature_set = list(feature_set) if feature_set is not None else self.feature_set
        if not feature_set:
            raise ValueError("feature_set is empty.")
        return {"validated": True, "n_features": len(feature_set)}

    def fit(self, X: Any, y: Any, context: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError(
            "AEEncoderOnlyAdapter is for inference only. "
            "Train the full AutoencoderAdapter first, then point this adapter "
            "at the saved encoder_only/ subdirectory of that run's artifacts_dir."
        )

    def save(self, output_dir: Optional[str] = None,
             context: Optional[Dict[str, Any]] = None) -> None:
        pass  # artifacts are produced by AutoencoderAdapter.save()

    def predict(self, X: Any) -> np.ndarray:
        """Encode input data through the AE encoder.  X is y-space data (e.g. IV curves)."""
        if self.best_autoencoder is None or self.y_scaler is None:
            base = self._base_dir()
            raise ValueError(
                "AEEncoderOnlyAdapter: model not loaded.\n"
                f"  resolved base_dir = {base}\n"
                "  Expected files: autoencoder_model.pth, scaler_y.pkl\n"
                "  Point pretrained_model_path or artifacts_dir at the "
                "encoder_only/ subdirectory inside the AE artifacts folder."
            )
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.y_scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        self.best_autoencoder.eval()
        with torch.no_grad():
            Z = self.best_autoencoder.encoder(X_tensor).cpu().numpy()
        return Z


def get_encoder_model(
    artifacts_dir: Optional[str] = None,
    model_config_path: Optional[str] = None,
    pretrained_model_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> AEEncoderOnlyAdapter:
    """BYOM entry point for AEEncoderOnlyAdapter."""
    return AEEncoderOnlyAdapter(
        artifacts_dir=artifacts_dir,
        model_config_path=model_config_path,
        pretrained_model_path=pretrained_model_path,
        context=context,
    )


# ---------------------------------------------------------------------------
# AEDecoderOnlyAdapter  (Z → y,  inference-only)
# ---------------------------------------------------------------------------

class AEDecoderOnlyAdapter:
    """BYOM adapter that exposes only the decoder half of a trained Autoencoder.

    predict(Z) → y

    Loads artifacts from the ``decoder_only/`` subdirectory saved by
    ``AutoencoderAdapter.save()``.  This adapter is inference-only.

    Typical use: ML Data tab — run predicted latent Z values through the
    decoder to reconstruct the full IV curve predictions.

    Entry point for BYOM:  ``get_decoder_model``
    """

    def __init__(
        self,
        artifacts_dir: Optional[str] = None,
        model_config_path: Optional[str] = None,
        pretrained_model_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.artifacts_dir = (
            Path(artifacts_dir).expanduser().resolve() if artifacts_dir else None
        )
        self.model_config_path = (
            Path(model_config_path).expanduser().resolve() if model_config_path else None
        )
        self.pretrained_model_path = (
            Path(pretrained_model_path).expanduser().resolve()
            if pretrained_model_path else None
        )
        self.context = context or {}
        self.feature_set = list(self.context.get("feature_set", []))
        self.label_set   = list(self.context.get("label_set",   []))
        self.model_config: Dict[str, Any] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_autoencoder: Optional[Autoencoder] = None
        self.y_scaler: Optional[MinMaxScaler] = None

        self._load_artifacts()

    def _base_dir(self) -> Optional[Path]:
        if self.pretrained_model_path and self.pretrained_model_path.is_dir():
            return self.pretrained_model_path
        return self.artifacts_dir

    def _load_artifacts(self) -> None:
        if self.model_config_path and self.model_config_path.exists():
            with open(self.model_config_path, "r", encoding="utf-8") as f:
                self.model_config = json.load(f)

        base_dir = self._base_dir()
        if base_dir is None:
            return

        pth_ae = base_dir / "autoencoder_model.pth"
        pkl_y  = base_dir / "scaler_y.pkl"

        if pth_ae.exists() and pkl_y.exists():
            self.y_scaler = joblib.load(pkl_y)
            sd_ae = torch.load(pth_ae, map_location=self.device)
            size_1, size_0 = sd_ae["encoder.0.weight"].shape
            size_2 = sd_ae["encoder.2.weight"].shape[0]
            size_3 = sd_ae["encoder.4.weight"].shape[0]
            z_dim  = sd_ae["encoder.6.weight"].shape[0]
            self.best_autoencoder = Autoencoder(
                size_0, size_1, size_2, size_3, z_dim
            ).to(self.device)
            self.best_autoencoder.load_state_dict(sd_ae)
            self.best_autoencoder.eval()
            print(f"AEDecoderOnlyAdapter: Loaded from {base_dir}")

    def metadata(self) -> Dict[str, Any]:
        return {
            "name":        "AEDecoderOnlyAdapter",
            "provider":    "custom_python",
            "feature_set": self.feature_set,
            "label_set":   self.label_set,
        }

    def validate(
        self,
        feature_set: Optional[Iterable[str]] = None,
        label_set: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        feature_set = list(feature_set) if feature_set is not None else self.feature_set
        if not feature_set:
            raise ValueError("feature_set is empty.")
        return {"validated": True, "n_features": len(feature_set)}

    def fit(self, X: Any, y: Any, context: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError(
            "AEDecoderOnlyAdapter is for inference only. "
            "Train the full AutoencoderAdapter first, then point this adapter "
            "at the saved decoder_only/ subdirectory of that run's artifacts_dir."
        )

    def save(self, output_dir: Optional[str] = None,
             context: Optional[Dict[str, Any]] = None) -> None:
        pass  # artifacts are produced by AutoencoderAdapter.save()

    def predict(self, X: Any) -> np.ndarray:
        """Decode latent Z through the AE decoder.  X is Z-space data."""
        if self.best_autoencoder is None or self.y_scaler is None:
            base = self._base_dir()
            raise ValueError(
                "AEDecoderOnlyAdapter: model not loaded.\n"
                f"  resolved base_dir = {base}\n"
                "  Expected files: autoencoder_model.pth, scaler_y.pkl\n"
                "  Point pretrained_model_path or artifacts_dir at the "
                "decoder_only/ subdirectory inside the AE artifacts folder."
            )
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.best_autoencoder.eval()
        with torch.no_grad():
            y_pred_scaled = self.best_autoencoder.decoder(X_tensor).cpu().numpy()
        return self.y_scaler.inverse_transform(y_pred_scaled)


def get_decoder_model(
    artifacts_dir: Optional[str] = None,
    model_config_path: Optional[str] = None,
    pretrained_model_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> AEDecoderOnlyAdapter:
    """BYOM entry point for AEDecoderOnlyAdapter."""
    return AEDecoderOnlyAdapter(
        artifacts_dir=artifacts_dir,
        model_config_path=model_config_path,
        pretrained_model_path=pretrained_model_path,
        context=context,
    )
