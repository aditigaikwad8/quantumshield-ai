"""QuantumShield TabDDPM: Benign-distribution anomaly detection via Denoising Diffusion.

Model Type: Denoising Diffusion Probabilistic Model (PyTorch)
Task: Benign-distribution network anomaly detection
Threshold: 0.56279456615448
11 Telemetry Features:
    e2c_total_bytes, e4_entropy_h, e5_entropy_c, e6_time_char, e6b_flow_duration_ms,
    e2_client_size, e2_client_record_len, e2_server_record_len,
    e1_alg_suite_mlkem1024, e1_alg_suite_mlkem768, e1_alg_suite_x25519
"""
from __future__ import annotations

import math
import pathlib
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    TABDDPM_FEATURES,
    TABDDPM_HIDDEN,
    TABDDPM_SCORE_SEED,
    TABDDPM_SCORE_TIMESTEPS,
    TABDDPM_STEPS,
    TABDDPM_THRESHOLD,
)


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(1, half))
    ang = t.float()[:, None] * freqs[None]
    emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class Denoiser(nn.Module):
    def __init__(self, n_features: int = 11, hidden: int = TABDDPM_HIDDEN, n_blocks: int = 3, t_dim: int = 64):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.inp = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 2),
                nn.SiLU(),
                nn.Linear(hidden * 2, hidden),
            )
            for _ in range(n_blocks)
        ])
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(_sinusoidal_time_embedding(t, self.t_dim))
        h = self.inp(x) + t_emb
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


class TabDDPM:
    """TabDDPM anomaly detector trained on benign distribution."""

    def __init__(
        self,
        n_features: int = len(TABDDPM_FEATURES),
        timesteps: int = TABDDPM_STEPS,
        hidden: int = TABDDPM_HIDDEN,
        threshold: float = TABDDPM_THRESHOLD,
        score_timesteps: int = TABDDPM_SCORE_TIMESTEPS,
        device: str = "cpu",
    ):
        self.n_features = n_features
        self.T = timesteps
        self.hidden = hidden
        self.threshold = threshold
        self.score_timesteps = score_timesteps
        self.device = torch.device(device)
        self.model = Denoiser(n_features, hidden).to(self.device)

        betas = torch.linspace(1e-4, 0.02, timesteps, device=self.device).clamp(max=0.999)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.abar = abar

    # ------------------------------------------------------------------ training
    def fit(
        self,
        x: np.ndarray,
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 2e-3,
        verbose: bool = True,
        seed: int = 42,
    ) -> list[float]:
        """Fit on benign network telemetry features only."""
        torch.manual_seed(seed)
        X = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        steps = max(1, math.ceil(len(X) / batch_size))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
        history = []
        self.model.train()
        for ep in range(epochs):
            total = 0.0
            perm = torch.randperm(len(X), device=self.device)
            for i in range(0, len(X), batch_size):
                idx = perm[i:i + batch_size]
                x0 = X[idx]
                t = torch.randint(0, self.T, (len(x0),), device=self.device)
                eps = torch.randn_like(x0)
                ab = self.abar[t][:, None]
                xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps
                pred_eps = self.model(xt, t)
                loss = F.mse_loss(pred_eps, eps)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item()
            mean_loss = total / steps
            history.append(mean_loss)
            if verbose and (ep % 2 == 0 or ep == epochs - 1):
                print(f"    TabDDPM epoch {ep + 1:2d}/{epochs}  loss {mean_loss:.5f}")
        return history

    # ------------------------------------------------------------------ anomaly scoring
    @torch.no_grad()
    def score(
        self,
        x: np.ndarray,
        score_timesteps: int | None = None,
        seed: int = TABDDPM_SCORE_SEED,
    ) -> np.ndarray:
        """Calculate anomaly score for each row in x based on diffusion reconstruction loss."""
        self.model.eval()
        K = score_timesteps or self.score_timesteps
        X = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        N = len(X)
        if N == 0:
            return np.zeros(0, dtype=np.float32)

        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        # Selected deterministic timesteps for scoring evaluation
        t_eval = torch.linspace(10, self.T - 10, K, device=self.device).long()
        loss_accum = torch.zeros(N, device=self.device)

        for t_step in t_eval:
            t = torch.full((N,), t_step.item(), device=self.device, dtype=torch.long)
            eps = torch.randn(X.shape, device=self.device, generator=g)
            ab = self.abar[t][:, None]
            xt = ab.sqrt() * X + (1.0 - ab).sqrt() * eps
            pred_eps = self.model(xt, t)
            mse_per_sample = ((pred_eps - eps) ** 2).mean(dim=1)
            loss_accum += mse_per_sample

        raw_score = (loss_accum / K).cpu().numpy()
        # Scale to an anomaly score roughly centered around benign baseline vs threshold
        return raw_score.astype(np.float32)

    def predict_anomaly(
        self,
        x: np.ndarray,
        score_timesteps: int | None = None,
        seed: int = TABDDPM_SCORE_SEED,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (anomaly_scores, is_anomaly_boolean)."""
        scores = self.score(x, score_timesteps=score_timesteps, seed=seed)
        is_anomaly = scores >= self.threshold
        return scores, is_anomaly

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Sample synthetic benign flows via reverse diffusion."""
        self.model.eval()
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        x = torch.randn(n, self.n_features, device=self.device, generator=g)
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            eps = self.model(x, t)
            ab, a, b = self.abar[i], self.alphas[i], self.betas[i]
            x0 = ((x - (1.0 - ab).sqrt() * eps) / ab.sqrt()).clamp(-6.0, 6.0)
            ab_prev = self.abar[i - 1] if i > 0 else torch.tensor(1.0, device=self.device)
            mean = (ab_prev.sqrt() * b / (1.0 - ab)) * x0 + (a.sqrt() * (1.0 - ab_prev) / (1.0 - ab)) * x
            if i > 0:
                var = b * (1.0 - ab_prev) / (1.0 - ab)
                x = mean + var.sqrt() * torch.randn(x.shape, device=self.device, generator=g)
            else:
                x = mean
        return x.clamp(-10.0, 10.0).cpu().numpy()

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | pathlib.Path) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state": self.model.state_dict(),
            "n_features": self.n_features,
            "timesteps": self.T,
            "hidden": self.hidden,
            "threshold": self.threshold,
            "score_timesteps": self.score_timesteps,
        }, p)

    @classmethod
    def load(cls, path: str | pathlib.Path, device: str = "cpu") -> "TabDDPM":
        ck = torch.load(path, map_location=device, weights_only=False)
        m = cls(
            n_features=ck.get("n_features", len(TABDDPM_FEATURES)),
            timesteps=ck.get("timesteps", TABDDPM_STEPS),
            hidden=ck.get("hidden", TABDDPM_HIDDEN),
            threshold=ck.get("threshold", TABDDPM_THRESHOLD),
            score_timesteps=ck.get("score_timesteps", TABDDPM_SCORE_TIMESTEPS),
            device=device,
        )
        m.model.load_state_dict(ck["state"])
        return m


class TabDDPMPreprocessor:
    """Preprocess the 11 telemetry features: median imputer + standard scaler."""

    def __init__(self):
        self.median = np.zeros(len(TABDDPM_FEATURES), dtype=np.float32)
        self.scale = np.ones(len(TABDDPM_FEATURES), dtype=np.float32)
        self.fitted = False

    @staticmethod
    def extract_tabddpm_features(df: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Extract or synthesize the 11 telemetry features from input dataframe or array."""
        if isinstance(df, np.ndarray):
            if df.shape[1] == len(TABDDPM_FEATURES):
                return df.astype(np.float32)
            # Truncate or pad to 11
            out = np.zeros((len(df), len(TABDDPM_FEATURES)), dtype=np.float32)
            k = min(df.shape[1], len(TABDDPM_FEATURES))
            out[:, :k] = df[:, :k]
            return out

        # If already has all 11 columns:
        if all(col in df.columns for col in TABDDPM_FEATURES):
            arr = np.array(df[TABDDPM_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32), copy=True)
            return arr

        # Otherwise synthesize from available network flow columns:
        arr = np.zeros((len(df), len(TABDDPM_FEATURES)), dtype=np.float32)
        # e2c_total_bytes
        if "Total Length of Fwd Packets" in df.columns:
            arr[:, 0] = pd.to_numeric(df["Total Length of Fwd Packets"], errors="coerce").fillna(500.0).to_numpy()
        elif "Flow Bytes/s" in df.columns:
            arr[:, 0] = pd.to_numeric(df["Flow Bytes/s"], errors="coerce").fillna(500.0).to_numpy()
        else:
            arr[:, 0] = 512.0

        # e4_entropy_h & e5_entropy_c
        if "Packet Length Std" in df.columns:
            arr[:, 1] = pd.to_numeric(df["Packet Length Std"], errors="coerce").fillna(10.0).to_numpy() / 50.0
            arr[:, 2] = arr[:, 1] * 0.8
        else:
            arr[:, 1] = 0.5
            arr[:, 2] = 0.4

        # e6_time_char
        if "Flow IAT Mean" in df.columns:
            arr[:, 3] = np.log1p(np.maximum(0.0, pd.to_numeric(df["Flow IAT Mean"], errors="coerce").fillna(100.0).to_numpy()))
        else:
            arr[:, 3] = 4.0

        # e6b_flow_duration_ms
        if "Flow Duration" in df.columns:
            arr[:, 4] = pd.to_numeric(df["Flow Duration"], errors="coerce").fillna(1000.0).to_numpy() / 1000.0
        else:
            arr[:, 4] = 50.0

        # e2_client_size
        if "Fwd Packet Length Mean" in df.columns:
            arr[:, 5] = pd.to_numeric(df["Fwd Packet Length Mean"], errors="coerce").fillna(200.0).to_numpy()
        else:
            arr[:, 5] = 256.0

        # e2_client_record_len
        if "Fwd Header Length" in df.columns:
            arr[:, 6] = pd.to_numeric(df["Fwd Header Length"], errors="coerce").fillna(32.0).to_numpy()
        else:
            arr[:, 6] = 32.0

        # e2_server_record_len
        if "Bwd Header Length" in df.columns:
            arr[:, 7] = pd.to_numeric(df["Bwd Header Length"], errors="coerce").fillna(32.0).to_numpy()
        else:
            arr[:, 7] = 32.0

        # Algorithm suites: default to ML-KEM-768
        arr[:, 8] = 0.0  # mlkem1024
        arr[:, 9] = 1.0  # mlkem768
        arr[:, 10] = 0.0 # x25519
        return arr

    def fit(self, df_or_arr: pd.DataFrame | np.ndarray) -> "TabDDPMPreprocessor":
        raw = self.extract_tabddpm_features(df_or_arr)
        raw = np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
        med = np.median(raw, axis=0)
        # Avoid zero standard deviation
        scale = np.std(raw, axis=0)
        scale = np.where(scale < 1e-4, 1.0, scale)
        self.median = med.astype(np.float32)
        self.scale = scale.astype(np.float32)
        self.fitted = True
        return self

    def transform(self, df_or_arr: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw = self.extract_tabddpm_features(df_or_arr)
        raw = np.nan_to_num(raw, nan=self.median, posinf=1e6, neginf=-1e6)
        scaled = (raw - self.median) / self.scale
        return np.clip(scaled, -10.0, 10.0).astype(np.float32)

    def save(self, path: str | pathlib.Path) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"median": self.median, "scale": self.scale, "features": TABDDPM_FEATURES}, p)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "TabDDPMPreprocessor":
        d = joblib.load(path)
        p = cls()
        p.median = np.asarray(d["median"], dtype=np.float32)
        p.scale = np.asarray(d["scale"], dtype=np.float32)
        p.fitted = True
        return p
