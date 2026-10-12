#!/usr/bin/env python3
"""
byom_adapter_poly.py
--------------------
Bring-Your-Own-Model (BYOM) adapter using a pure Polynomial Regressor
(sklearn Ridge regression) mapping directly from polynomial-expanded input
features to the full output space. No latent space, no PyTorch required.

Architecture:
  fit:     X  ──poly features──► X_poly ──Ridge──► y_scaled
  predict: X  ──poly features──► X_poly ──Ridge──► y_scaled ──inv_scaler──► y

Artifacts produced / consumed:
  poly_model.pkl       : fitted Ridge regressor  (X_poly → y_scaled)
  scaler_y.pkl         : MinMaxScaler fitted on y
  poly_ext_params.json : degree / cross_degree / best metrics from fit

model_config keys (all optional, with defaults shown):
  degree       (int)   : maximum total polynomial degree to search up to  [default: 3]
  cross_degree (int)   : maximum cross_degree to search up to             [default: degree]
  alpha        (float) : Ridge regularisation strength                    [default: 1.0]

During fit() the adapter performs a grid search over all (d, cd) pairs where
  d  ∈ [1 .. degree]  and  cd ∈ [1 .. d]
and selects the combination with the best (highest) mean R² across all outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for byom_adapter_poly. "
        "Install it with:  pip install scikit-learn"
    ) from exc

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper: filtered polynomial features
# (identical logic to byom_adapter_autoencoder_poly.py)
# ---------------------------------------------------------------------------

def custom_polynomial_features(
    x: np.ndarray, d: int, p: int, include_bias: bool = True
) -> np.ndarray:
    """
    Build polynomial features up to total degree *d*, keeping only terms where
    the maximum single-variable exponent is <= *p* (cross-term filter).

    Parameters
    ----------
    x            : (n_samples, n_features) array
    d            : maximum total polynomial degree
    p            : maximum individual-variable exponent in cross terms
    include_bias : whether to include the bias column (column of ones)

    Returns
    -------
    (n_samples, n_filtered_terms) array
    """
    n_samples, _ = x.shape
    poly = PolynomialFeatures(d, include_bias=include_bias)
    poly.fit_transform(x)          # populates .powers_
    feature_indices = poly.powers_

    filtered_indices = [
        term
        for term in feature_indices
        if sum(term) <= p or max(term) == sum(term)
    ]

    x_filtered = np.empty((n_samples, len(filtered_indices)), dtype=float)
    for i, index in enumerate(filtered_indices):
        x_filtered[:, i] = np.prod(
            [x[:, j] ** exp for j, exp in enumerate(index)], axis=0
        )
    return x_filtered


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class PolyAdapter:
    """
    BYOM adapter: polynomial regression (Ridge) mapping X directly to y.
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
            Path(model_config_path).expanduser().resolve()
            if model_config_path else None
        )
        self.pretrained_model_path = (
            Path(pretrained_model_path).expanduser().resolve()
            if pretrained_model_path else None
        )
        self.context = context or {}

        self.feature_set = list(self.context.get("feature_set", []))
        self.label_set   = list(self.context.get("label_set", []))

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

        # Model state
        self.poly_model:   Optional[Ridge]        = None
        self.torch_model:  Optional[Any]           = None   # nn.Linear when custom loss used
        self.scaler_X:     Optional[MinMaxScaler]  = None   # only used in torch path
        self.scaler_y:     Optional[MinMaxScaler]  = None

        self.fitted_degree:       Optional[int]   = None
        self.fitted_cross_degree: Optional[int]   = None
        self.best_r2:             Optional[float] = None
        self.best_rmse:           Optional[float] = None
        self.best_mse:            Optional[float] = None

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

        pkl_model   = base_dir / "poly_model.pkl"
        pkl_y       = base_dir / "scaler_y.pkl"
        params_path = base_dir / "poly_ext_params.json"
        if pkl_y.exists():
            self.scaler_y = joblib.load(pkl_y)

        if pkl_model.exists():
            self.poly_model = joblib.load(pkl_model)
            print("PolyAdapter: Loaded Ridge model from", base_dir)

        pkl_scaler_X = base_dir / "scaler_X.pkl"
        if pkl_scaler_X.exists():
            self.scaler_X = joblib.load(pkl_scaler_X)

        pt_model = base_dir / "poly_torch_model.pt"
        pt_arch  = base_dir / "poly_torch_arch.pt"
        if pt_model.exists() and pt_arch.exists() and _TORCH_AVAILABLE:
            arch = torch.load(pt_arch, map_location="cpu")
            m = nn.Linear(arch["in_features"], arch["out_features"])
            m.load_state_dict(torch.load(pt_model, map_location="cpu"))
            m.eval()
            self.torch_model = m
            print("PolyAdapter: Loaded torch model from", base_dir)

        if params_path.exists():
            with open(params_path, "r") as f:
                params = json.load(f)
            self.fitted_degree       = params.get("degree")
            self.fitted_cross_degree = params.get("cross_degree")
            self.best_r2             = params.get("best_r2")
            self.best_rmse           = params.get("best_rmse")
            self.best_mse            = params.get("best_mse")
            self._loss_code_used     = params.get("loss_code_used", "ridge_mse")

    # ------------------------------------------------------------------
    # Custom-loss helpers
    # ------------------------------------------------------------------

    def _load_loss_fn(self):
        """Return a compute_loss callable if loss_formula is configured, else None."""
        provider_cfg = self.context.get("provider_cfg", {}) or {}
        loss_formula = (
            self.model_config.get("loss_formula")
            or provider_cfg.get("loss_formula")
        )
        if loss_formula and isinstance(loss_formula, dict):
            try:
                from physics_loss_builder import build_loss_fn
                print("PolyAdapter: Using physics_loss_builder for loss function.")
                return build_loss_fn(loss_formula)
            except Exception as e:
                print(f"PolyAdapter: Warning — physics_loss_builder failed ({e}). "
                      "Falling back to Ridge.")
        return None

    def _torch_fit(
        self,
        X_poly: np.ndarray,
        y_scaled: np.ndarray,
        compute_loss_fn,
        loss_code_str: str = "",
    ):
        """
        Train a single nn.Linear model on pre-built polynomial features using
        the supplied compute_loss callable.  Returns the trained nn.Linear.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for custom-loss training. "
                "Install it with:  pip install torch"
            )
        import torch.nn.functional as F

        cfg        = self.model_config
        lr         = float(cfg.get("lr",         1e-3))
        epochs     = int(cfg.get("epochs",       500))
        patience   = int(cfg.get("patience",     20))
        batch_size = int(cfg.get("batch_size",   64))
        l2         = float(cfg.get("l2",         1e-4))
        val_split  = float(cfg.get("val_split",  0.1))

        n_in  = X_poly.shape[1]
        n_out = y_scaled.shape[1]

        n_val   = max(1, int(len(X_poly) * val_split))
        n_train = len(X_poly) - n_val
        idx     = np.random.permutation(len(X_poly))
        tr_idx, va_idx = idx[:n_train], idx[n_train:]

        X_tr = torch.from_numpy(X_poly[tr_idx].astype(np.float32))
        y_tr = torch.from_numpy(y_scaled[tr_idx].astype(np.float32))
        X_va = torch.from_numpy(X_poly[va_idx].astype(np.float32))
        y_va = torch.from_numpy(y_scaled[va_idx].astype(np.float32))

        train_ds = TensorDataset(X_tr, y_tr)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model     = nn.Linear(n_in, n_out)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

        best_val   = float("inf")
        best_state = None
        no_improve = 0

        # Echo the loss formula so it's clear what code will run
        print(f"  PolyAdapter (torch): n_poly_features={n_in}, n_out={n_out}, "
              f"lr={lr}, epochs={epochs}, patience={patience}")
        if loss_code_str:
            print("  Loss formula in use:")
            for line in loss_code_str.strip().splitlines():
                print(f"    {line}")
        print()
        print(f"  {'epoch':>6}  {'custom_loss':>13}  {'plain_mse':>11}  {'val_loss':>12}  {'val_mse':>10}")
        print("  " + "-" * 60)

        for epoch in range(1, epochs + 1):
            model.train()
            ep_loss = 0.0
            for X_b, y_b in train_dl:
                optimizer.zero_grad()
                loss = compute_loss_fn(model(X_b), y_b)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item() * len(X_b)
            ep_loss /= n_train

            model.eval()
            with torch.no_grad():
                y_va_pred = model(X_va)
                val_loss  = compute_loss_fn(y_va_pred, y_va).item()
                val_mse   = F.mse_loss(y_va_pred, y_va).item()

            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch == 1 or epoch % 50 == 0 or no_improve == patience:
                # Also compute plain MSE on train batch for comparison
                with torch.no_grad():
                    train_mse = F.mse_loss(model(X_tr), y_tr).item()
                print(f"  {epoch:>6}  {ep_loss:>13.6f}  {train_mse:>11.6f}  "
                      f"{val_loss:>12.6f}  {val_mse:>10.6f}")

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}  (best val={best_val:.6f})")
                break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        return model, best_val

    # ------------------------------------------------------------------
    # BYOM contract methods
    # ------------------------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "PolyAdapter",
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

        # --- Hyper-parameters from model_config ---
        max_degree       = int(self.model_config.get("degree", 3))
        max_cross_degree = int(self.model_config.get("cross_degree", max_degree))
        alpha            = float(self.model_config.get("alpha", 1.0))

        # 1. Scale y → [0, 1]  (done once; search is in scaled space)
        self.scaler_y = MinMaxScaler()
        y_scaled = self.scaler_y.fit_transform(y)

        # 2. Check for custom loss
        custom_loss_fn = self._load_loss_fn()

        if custom_loss_fn is not None:
            # ── PyTorch path: fixed degree, gradient descent, custom loss ───
            degree       = int(self.model_config.get("degree", 2))
            cross_degree = int(self.model_config.get("cross_degree", degree))
            # Scale X to [0,1] BEFORE poly expansion — prevents degree-N terms
            # from exploding (e.g. Nbuf^9 with raw values >> 1 → NaN gradients).
            self.scaler_X = MinMaxScaler()
            X_norm = self.scaler_X.fit_transform(X)
            X_poly = custom_polynomial_features(X_norm, degree, cross_degree, include_bias=True)

            # Extract loss_code string for display/saving
            provider_cfg = self.context.get("provider_cfg", {}) or {}
            loss_formula = (
                self.model_config.get("loss_formula")
                or provider_cfg.get("loss_formula")
            ) or {}
            loss_code_str = str(loss_formula.get("loss_code", "") or "").strip()

            print(
                f"PolyAdapter (custom loss): degree={degree}, cross_degree={cross_degree}, "
                f"n_poly_features={X_poly.shape[1]}"
            )
            torch_model, best_val = self._torch_fit(
                X_poly, y_scaled, custom_loss_fn, loss_code_str=loss_code_str
            )

            # Compute final R²/RMSE on full training set for reporting
            with torch.no_grad():
                X_t = torch.from_numpy(X_poly.astype(np.float32))
                y_pred_scaled = torch_model(X_t).numpy()
            if np.isnan(y_pred_scaled).any():
                print("PolyAdapter: WARNING — predictions contain NaN after training. "
                      "Try reducing degree or increasing l2 regularisation.")
                r2, rmse, mse = float("nan"), float("nan"), float("nan")
            else:
                mse  = float(mean_squared_error(y_scaled, y_pred_scaled))
                rmse = float(np.sqrt(mse))
                r2   = float(np.mean(r2_score(y_scaled, y_pred_scaled,
                                               multioutput="raw_values")))
                print(
                    f"PolyAdapter: Final train metrics → "
                    f"R²={r2:.6f}  RMSE={rmse:.6f}  MSE={mse:.6f}"
                )

            self.torch_model         = torch_model
            self.poly_model          = None
            self.fitted_degree       = degree
            self.fitted_cross_degree = cross_degree
            self.best_r2             = r2
            self.best_rmse           = rmse
            self.best_mse            = mse
            self._loss_code_used     = loss_code_str
            return

        # ── Ridge path: grid search over all (d, cd) combinations ─────────
        best_r2    = -np.inf
        best_rmse  =  np.inf
        best_mse   =  np.inf
        best_d     = 1
        best_cd    = 1
        best_model = None

        total = sum(
            min(d, max_cross_degree)
            for d in range(1, max_degree + 1)
        )
        print(
            f"PolyAdapter: Starting degree grid-search — "
            f"max_degree={max_degree}, max_cross_degree={max_cross_degree}, "
            f"alpha={alpha}, total candidates={total}"
        )
        print(f"  {'d':>4}  {'cd':>4}  {'n_features':>12}  {'R2 (mean)':>12}  {'RMSE':>12}  {'MSE':>12}")
        print("  " + "-" * 62)

        for d in range(1, max_degree + 1):
            for cd in range(1, min(d, max_cross_degree) + 1):
                X_poly = custom_polynomial_features(X, d, cd, include_bias=True)

                model = Ridge(alpha=alpha)
                model.fit(X_poly, y_scaled)
                y_pred_scaled = model.predict(X_poly)

                mse  = float(mean_squared_error(y_scaled, y_pred_scaled))
                rmse = float(np.sqrt(mse))
                r2   = float(np.mean(r2_score(y_scaled, y_pred_scaled,
                                               multioutput='raw_values')))

                print(
                    f"  {d:>4}  {cd:>4}  {X_poly.shape[1]:>12}  "
                    f"{r2:>12.6f}  {rmse:>12.6f}  {mse:>12.6f}"
                )

                if r2 > best_r2 or (r2 == best_r2 and rmse < best_rmse):
                    best_r2    = r2
                    best_rmse  = rmse
                    best_mse   = mse
                    best_d     = d
                    best_cd    = cd
                    best_model = model
                    self.fitted_degree       = d
                    self.fitted_cross_degree = cd

        print("  " + "-" * 62)
        print(
            f"PolyAdapter: Best configuration → degree={best_d}, cross_degree={best_cd}  "
            f"| R²={best_r2:.6f}  RMSE={best_rmse:.6f}  MSE={best_mse:.6f}"
        )

        self.poly_model = best_model
        self.best_r2    = best_r2
        self.best_rmse  = best_rmse
        self.best_mse   = best_mse

    def save(
        self, output_dir: Optional[str] = None, context: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.poly_model is None and self.torch_model is None:
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

        # Save whichever model was trained
        if self.torch_model is not None:
            torch.save(self.torch_model.state_dict(), save_dir / "poly_torch_model.pt")
            torch.save(
                {"in_features": self.torch_model.in_features,
                 "out_features": self.torch_model.out_features},
                save_dir / "poly_torch_arch.pt",
            )
        if self.scaler_X is not None:
            joblib.dump(self.scaler_X, save_dir / "scaler_X.pkl")
        if self.poly_model is not None:
            joblib.dump(self.poly_model, save_dir / "poly_model.pkl")
        joblib.dump(self.scaler_y, save_dir / "scaler_y.pkl")

        params = {
            "degree":          self.fitted_degree       or int(self.model_config.get("degree", 3)),
            "cross_degree":    self.fitted_cross_degree or int(self.model_config.get("cross_degree", 3)),
            "best_r2":         self.best_r2,
            "best_rmse":       self.best_rmse,
            "best_mse":        self.best_mse,
            "use_torch":       self.torch_model is not None,
            "loss_code_used":  getattr(self, "_loss_code_used", None) or "ridge_mse",
        }
        with open(save_dir / "poly_ext_params.json", "w") as f:
            json.dump(params, f, indent=2)

        # Bundled single-file export
        bundle = {
            "poly_model":  self.poly_model,
            "torch_state": self.torch_model.state_dict() if self.torch_model else None,
            "torch_arch":  {"in_features": self.torch_model.in_features,
                            "out_features": self.torch_model.out_features}
                           if self.torch_model else None,
            "scaler_y":    self.scaler_y,
            "params":      params,
        }
        joblib.dump(bundle, save_dir / "poly_adapter.joblib")

        # Self-contained poly/ subdirectory
        import shutil as _shutil
        try:
            _this_file = Path(type(self).__init__.__code__.co_filename).resolve()
        except Exception:
            _this_file = Path(__file__).resolve()

        sub_dir = save_dir / "poly"
        sub_dir.mkdir(parents=True, exist_ok=True)
        if self.torch_model is not None:
            _shutil.copy2(save_dir / "poly_torch_model.pt",  sub_dir / "poly_torch_model.pt")
            _shutil.copy2(save_dir / "poly_torch_arch.pt",   sub_dir / "poly_torch_arch.pt")
        if self.scaler_X is not None:
            _shutil.copy2(save_dir / "scaler_X.pkl", sub_dir / "scaler_X.pkl")
        if self.poly_model is not None:
            _shutil.copy2(save_dir / "poly_model.pkl", sub_dir / "poly_model.pkl")
        _shutil.copy2(save_dir / "scaler_y.pkl",         sub_dir / "scaler_y.pkl")
        _shutil.copy2(save_dir / "poly_ext_params.json", sub_dir / "poly_ext_params.json")
        if _this_file.exists():
            _shutil.copy2(str(_this_file), str(sub_dir / _this_file.name))
        with open(sub_dir / "model_manifest.json", "w") as _f:
            json.dump({
                "entry_point":  "get_model",
                "adapter_file": _this_file.name,
            }, _f, indent=2)

        print(f"PolyAdapter: Artifacts saved to {save_dir}")
        print(f"PolyAdapter: Bundled model saved → {save_dir / 'poly_adapter.joblib'}")
        print(f"PolyAdapter: Self-contained folder → {sub_dir}")

    def predict(self, X: Any) -> np.ndarray:
        if self.poly_model is None and self.torch_model is None:
            raise ValueError(
                "Model is not available. Train first or provide existing artifacts path."
            )

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        degree       = self.fitted_degree       or int(self.model_config.get("degree", 3))
        cross_degree = self.fitted_cross_degree or int(self.model_config.get("cross_degree", degree))

        X_in = self.scaler_X.transform(X) if self.scaler_X is not None else X
        X_poly = custom_polynomial_features(X_in, degree, cross_degree, include_bias=True)

        if self.torch_model is not None:
            X_t = torch.from_numpy(X_poly.astype(np.float32))
            self.torch_model.eval()
            with torch.no_grad():
                y_pred_scaled = self.torch_model(X_t).numpy()
        else:
            y_pred_scaled = self.poly_model.predict(X_poly)

        Y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        # Validate output shape
        n_samples = X.shape[0]
        n_labels  = len(self.label_set)
        if Y_pred.shape != (n_samples, n_labels):
            raise ValueError(
                f"predict() must return shape {(n_samples, n_labels)}, "
                f"but got {Y_pred.shape}. "
                "Make sure your dataset outputs match label_set."
            )

        return Y_pred


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def get_model(
    artifacts_dir: Optional[str] = None,
    model_config_path: Optional[str] = None,
    pretrained_model_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> PolyAdapter:
    """
    Default pipeline entry point for BYOM.
    """
    return PolyAdapter(
        artifacts_dir=artifacts_dir,
        model_config_path=model_config_path,
        pretrained_model_path=pretrained_model_path,
        context=context,
    )
