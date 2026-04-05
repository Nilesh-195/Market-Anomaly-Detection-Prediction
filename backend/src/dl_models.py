"""
Deep Learning Models for Price Forecasting
============================================
Implements LSTM Seq2Seq and Transformer for multi-step price prediction

Architecture:
  - LSTM Seq2Seq: Encoder-decoder with attention mechanism
  - Transformer: Multi-head self-attention with positional encoding
  - Both models: Quantile regression for 95% confidence intervals

Key Features:
  - Attention weights for interpretability
  - Quantile predictions (lower_95, point, upper_95)
  - MinMax scaling for stable training
  - Learning rate scheduling with early stopping
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')

import pickle
from pathlib import Path
import logging

from features import load_all_features

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).parent.parent / "models"



class QuantileRegressionLoss(nn.Module):
    """Quantile regression loss for 3 quantiles (lower, median, upper)."""

    def __init__(self, quantiles=[0.025, 0.5, 0.975]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, output, target):
        """
        Args:
            output: (batch, horizon, 3) - lower, point, upper predictions
            target: (batch, horizon) - true values
        Returns:
            Weighted loss across 3 quantiles
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            pred = output[:, :, i]
            residual = target - pred
            # Quantile loss: rho_q(u) = u*(q-1) if u<0, u*q if u>=0
            loss = torch.where(residual >= 0, q * residual, (q - 1) * residual)
            losses.append(loss.mean())

        return sum(losses) / len(losses)


class LSTMSeq2Seq(nn.Module):
    """LSTM Encoder-Decoder for multi-step price forecasting with attention."""

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        lookback=30,
        horizon=30,
        dropout=0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lookback = lookback
        self.horizon = horizon
        self.num_quantiles = 3  # lower_95, point, upper_95

        # Encoder: LSTM on past 30 days
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Decoder: LSTM — input is a single price value (size=1)
        self.decoder = nn.LSTM(
            input_size=1,  # FIX: input is a scalar price, not hidden_size
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=4, batch_first=True
        )

        # Output layers for 3 quantiles
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.num_quantiles),
        )

    def forward(self, x, last_target, target=None):
        """
        Args:
            x: (batch, lookback, n_features) - past 30 days of features
            last_target: (batch, 1, 1) - the last known target value (for decoding)
            target: (batch, horizon, 1) - next N days (for teacher forcing)
        Returns:
            output: (batch, horizon, 3) - predictions for 3 quantiles
            attention_weights: (batch, horizon, lookback)
        """
        batch_size = x.size(0)

        # Encode the past window
        encoder_out, (h_n, c_n) = self.encoder(x)  # (batch, lookback, hidden)

        # Decoder seed: last observed target
        decoder_input = last_target  # (batch, 1, 1)

        outputs = []
        attention_weights_list = []

        for t in range(self.horizon):
            # Decode one step
            decoder_out, (h_n, c_n) = self.decoder(
                decoder_input, (h_n, c_n)
            )  # (batch, 1, hidden)

            # Cross-attention: query=decoder state, key/value=encoder states
            attn_out, attn_w = self.attention(
                query=decoder_out, key=encoder_out, value=encoder_out
            )  # (batch, 1, hidden), (batch, 1, lookback)
            attention_weights_list.append(
                attn_w.squeeze(1)  # (batch, lookback)
            )

            # Predict quantiles
            pred = self.fc(attn_out)  # (batch, 1, 3)
            outputs.append(pred)

            # Next decoder input: teacher forcing or predicted point estimate
            if target is not None:
                decoder_input = target[:, t : t + 1, :]  # (batch, 1, 1)
            else:
                decoder_input = pred[:, :, 1:2]  # point estimate (batch, 1, 1)

        # Stack outputs
        output = torch.cat(outputs, dim=1)  # (batch, horizon, 3)
        attention_weights = torch.stack(
            attention_weights_list, dim=1
        )  # (batch, horizon, lookback)

        return output, attention_weights


class TransformerForecaster(nn.Module):
    """Transformer-based forecaster with multi-head self-attention."""

    def __init__(
        self,
        input_size=1,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        lookback=30,
        horizon=30,
        dropout=0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.lookback = lookback
        self.horizon = horizon
        self.num_quantiles = 3

        # ── Variable Selection Network (Feature Attention) ──
        self.var_selection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, input_size),
            nn.Softmax(dim=-1)
        )

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        self.target_proj = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoder = self._create_positional_encoding(lookback + horizon)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Autoregressive Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            batch_first=True, 
            dropout=dropout
        )
        
        self.output_proj = nn.Linear(d_model, self.num_quantiles)

    def _create_positional_encoding(self, max_len):
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-np.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x, last_target, target=None):
        """
        Args:
            x: (batch, lookback, input_size) - past context
            last_target: (batch, 1, 1) - decoder seed
            target: None in standard loop
        Returns:
            output: (batch, horizon, 3) - quantile predictions
            feature_weights: (batch, lookback, input_size) - variable selection map
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 1. Dynamic Variable Selection
        feature_weights = self.var_selection(x)
        x_weighted = x * feature_weights

        # 2. Encode
        x_proj = self.input_proj(x_weighted)  # (batch, lookback, d_model)
        pos = self.pos_encoder[:, :seq_len, :].to(x.device)
        x_proj = x_proj + pos

        memory = self.transformer_encoder(x_proj)  # (batch, lookback, d_model)

        # 3. Autoregressive Decode
        decoder_input = self.target_proj(last_target)  # (batch, 1, d_model)
        
        outputs = []
        for t in range(self.horizon):
            # Decode step
            dec_out = self.decoder_layer(decoder_input, memory)  # (batch, 1, d_model)
            pred = self.output_proj(dec_out)  # (batch, 1, 3)
            outputs.append(pred)
            
            # Autoregressive shift
            if target is not None:
                next_target = target[:, t:t+1, :]
            else:
                next_target = pred[:, :, 1:2]  # point estimate
                
            decoder_input = self.target_proj(next_target)

        output = torch.cat(outputs, dim=1)  # (batch, horizon, 3)

        return output, feature_weights


# Allowlist custom model classes so torch.load works with weights_only=True
try:
    torch.serialization.add_safe_globals([LSTMSeq2Seq, TransformerForecaster])
except (AttributeError, RuntimeError):
    pass


def create_sequences(features, targets, lookback=30, horizon=30):
    """Create train sequences for LSTM/Transformer.

    Args:
        features: (n_samples, n_features) array of macro/technical features
        targets: (n_samples, 1) array of target values to predict
        lookback: Number of past steps to use
        horizon: Number of future steps to predict

    Returns:
        X: (n_sequences, lookback, n_features) - past features
        Y: (n_sequences, horizon, 1) - future targets
    """
    X, Y, last_Y = [], [], []
    for i in range(len(features) - lookback - horizon + 1):
        X.append(features[i : i + lookback])
        Y.append(targets[i + lookback : i + lookback + horizon].reshape(-1, 1))
        last_Y.append(targets[i + lookback - 1 : i + lookback].reshape(-1, 1))
    return np.array(X), np.array(Y), np.array(last_Y)


def train_lstm_seq2seq(
    df, asset_name, lookback=30, horizon=30, epochs=50, batch_size=32
):
    """Train LSTM Seq2Seq model.

    Args:
        df: DataFrame containing all features + target (Close)
        asset_name: Asset name for logging
        lookback: Past window size
        horizon: Future prediction horizon
        epochs: Training epochs
        batch_size: Batch size

    Returns:
        model: Trained model
        scaler: Scaler info for target
        scaler_x: Scaler object for features
        feature_cols: List of features used
    """
    from sklearn.preprocessing import StandardScaler

    close_prices = df['Close'].values
    returns = np.log(close_prices[1:] / close_prices[:-1])
    mean_target = returns.mean()
    std_target = returns.std()
    target_normalized = (returns - mean_target) / (std_target + 1e-8)
    
    # Feature engineering for Multi-variate input
    # Ignore dates, prices, and volumes directly to avoid look-ahead leakage
    drop_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Since we are predicting log returns directly, we shift features by 1 
    # to align past feature context with next-step return targets
    feature_df = df[feature_cols].iloc[1:].fillna(0)
    
    scaler_x = StandardScaler()
    features_normalized = scaler_x.fit_transform(feature_df)

    # Create sequences
    X, Y, last_Y = create_sequences(features_normalized, target_normalized, lookback, horizon)

    # Train/test split (70/30)
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    last_Y_train, last_Y_test = last_Y[:split], last_Y[split:]

    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.FloatTensor(Y_train)
    last_Y_train = torch.FloatTensor(last_Y_train)
    X_test = torch.FloatTensor(X_test)
    Y_test = torch.FloatTensor(Y_test)
    last_Y_test = torch.FloatTensor(last_Y_test)

    # DataLoader
    train_dataset = TensorDataset(X_train, Y_train, last_Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LSTMSeq2Seq(
        input_size=len(feature_cols),
        hidden_size=64,
        num_layers=2,
        lookback=lookback,
        horizon=horizon,
        dropout=0.2,
    ).to(device)

    # Loss and optimizer
    criterion = QuantileRegressionLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, Y_batch, last_Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            last_Y_batch = last_Y_batch.to(device)

            optimizer.zero_grad()
            output, _ = model(X_batch, last_Y_batch, Y_batch)
            loss = criterion(output, Y_batch.squeeze(-1))  # FIX: squeeze (batch,horizon,1)->(batch,horizon)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(device)
            Y_test_device = Y_test.to(device)
            last_Y_test_device = last_Y_test.to(device)
            output_test, _ = model(X_test_device, last_Y_test_device)
            val_loss = criterion(output_test, Y_test_device.squeeze(-1)).item()  # FIX: squeeze

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"[{asset_name}] LSTM Epoch {epoch+1}/{epochs} - "
                f"Train: {avg_loss:.6f}, Val: {val_loss:.6f}"
            )

        if patience_counter >= 10:
            print(f"[{asset_name}] LSTM early stopping at epoch {epoch+1}")
            break

    scaler = {"mean": float(mean_target), "std": float(std_target), "mode": "returns"}
    return model.cpu(), scaler, scaler_x, feature_cols


def train_transformer(
    df, asset_name, lookback=30, horizon=30, epochs=50, batch_size=32
):
    """Train Transformer model (similar to LSTM).

    Args:
        df: DataFrame containing all features + target (Close)
        asset_name: Asset name for logging
        lookback: Past window size
        horizon: Future prediction horizon
        epochs: Training epochs
        batch_size: Batch size

    Returns:
        model: Trained model
        scaler: Scaler info for target
        scaler_x: Scaler object for features
        feature_cols: List of features used
    """
    from sklearn.preprocessing import StandardScaler

    close_prices = df['Close'].values
    returns = np.log(close_prices[1:] / close_prices[:-1])
    mean_target = returns.mean()
    std_target = returns.std()
    target_normalized = (returns - mean_target) / (std_target + 1e-8)
    
    # Feature engineering for Multi-variate input
    drop_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    feature_df = df[feature_cols].iloc[1:].fillna(0)
    
    scaler_x = StandardScaler()
    features_normalized = scaler_x.fit_transform(feature_df)

    # Create sequences
    X, Y, last_Y = create_sequences(features_normalized, target_normalized, lookback, horizon)

    # Train/test split
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    last_Y_train, last_Y_test = last_Y[:split], last_Y[split:]

    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.FloatTensor(Y_train)
    last_Y_train = torch.FloatTensor(last_Y_train)
    X_test = torch.FloatTensor(X_test)
    Y_test = torch.FloatTensor(Y_test)
    last_Y_test = torch.FloatTensor(last_Y_test)

    # DataLoader
    train_dataset = TensorDataset(X_train, Y_train, last_Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TransformerForecaster(
        input_size=len(feature_cols),
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        lookback=lookback,
        horizon=horizon,
        dropout=0.2,
    ).to(device)

    # Loss and optimizer
    criterion = QuantileRegressionLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, Y_batch, last_Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            last_Y_batch = last_Y_batch.to(device)

            optimizer.zero_grad()
            output, _ = model(X_batch, last_Y_batch, Y_batch)
            loss = criterion(output, Y_batch.squeeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(device)
            Y_test_device = Y_test.to(device)
            last_Y_test_device = last_Y_test.to(device)
            output_test, _ = model(X_test_device, last_Y_test_device)
            val_loss = criterion(output_test, Y_test_device.squeeze(-1)).item()

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"[{asset_name}] Transformer Epoch {epoch+1}/{epochs} - "
                f"Train: {avg_loss:.6f}, Val: {val_loss:.6f}"
            )

        if patience_counter >= 10:
            print(f"[{asset_name}] Transformer early stopping at epoch {epoch+1}")
            break

    scaler = {"mean": float(mean_target), "std": float(std_target), "mode": "returns"}
    return model.cpu(), scaler, scaler_x, feature_cols


def forecast_dl_model(model, features, last_target, horizon=30, scaler=None, model_type="lstm"):
    """Generate forecast from trained DL model.

    Args:
        model: Trained LSTM or Transformer model
        features: (lookback, n_features) normalized features
        last_target: (1,) normalized last target value (for LSTM seed)
        horizon: Prediction horizon
        scaler: Scaler info
        model_type: "lstm" or "transformer"

    Returns:
        forecast: (horizon,) point predictions (denormalized)
        lower_95: (horizon,) lower bound
        upper_95: (horizon,) upper bound
        attention: (horizon, lookback) attention weights or None
    """
    model.eval()

    with torch.no_grad():
        X = torch.FloatTensor(features).unsqueeze(0)  # (1, lookback, n_features)
        LT = torch.FloatTensor(last_target).unsqueeze(0).unsqueeze(-1) # (1, 1, 1)

        if model_type == "lstm":
            output, attention = model(X, LT)  # (1, horizon, 3), (1, horizon, lookback)
        else:
            output, attention = model(X, LT)  # (1, horizon, 3), (1, lookback, input_size) feature weights


        # Extract quantiles (lower_95, point, upper_95)
        output = output.numpy().squeeze(0)  # (horizon, 3)
        lower = output[:, 0]
        point = output[:, 1]
        upper = output[:, 2]

    # Denormalize
    if scaler:
        min_val = scaler["min"]
        max_val = scaler["max"]
        range_val = max_val - min_val + 1e-8

        point = point * range_val + min_val
        lower = lower * range_val + min_val
        upper = upper * range_val + min_val

    return point, lower, upper, attention

"""
Forecasting Functions for DL Models
====================================
High-level interface for LSTM, Transformer, and XGBoost forecasting

Provides:
  - lstm_seq2seq_forecast()
  - transformer_forecast()
  - xgboost_forecast()

Each returns forecast, confidence intervals, and explainability metrics
"""

import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
import logging


logger = logging.getLogger(__name__)

# Allowlist custom model classes so torch.load works with weights_only=True as well
try:
    torch.serialization.add_safe_globals([LSTMSeq2Seq, TransformerForecaster])
except AttributeError:
    pass  # Older torch versions don't have add_safe_globals

# Model paths
MODEL_DIR = Path(__file__).parent.parent / "models"


def _load_dl_model(asset, model_name):
    """Load LSTM/Transformer model.

    Args:
        asset: Asset name (e.g., 'SP500')
        model_name: 'lstm_seq2seq' or 'transformer'

    Returns:
        model: PyTorch model
        meta: Scaler info {"min": float, "max": float}
    """
    model_path = MODEL_DIR / asset / f"{model_name}.pt"
    meta_path = MODEL_DIR / asset / f"{model_name}_meta.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # weights_only=False needed to load full model objects (PyTorch 2.6+)
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()

    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    else:
        meta = None

    return model, meta


def lstm_seq2seq_forecast(asset, horizon=30):
    """Generate LSTM Seq2Seq forecast for an asset.

    Args:
        asset: Asset name (e.g., 'SP500')
        horizon: Forecast horizon in days (default 30)

    Returns:
        dict with forecast, lower_95, upper_95 (or empty dict if model unavailable)
    """
    try:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        model, meta = _load_dl_model(asset, 'lstm_seq2seq')
        features_data = load_all_features()
        if asset not in features_data:
            raise ValueError(f"No feature data found for {asset}")

        df = features_data[asset]
        lookback = int(getattr(model, "lookback", 30))
        if len(df) < lookback + 2:
            raise ValueError(f"Insufficient history for {asset}: need at least {lookback + 2} rows")

        feature_cols = meta.get("feature_cols") if isinstance(meta, dict) else None
        if not feature_cols:
            drop_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            feature_cols = [c for c in df.columns if c not in drop_cols]

        feature_window = df.reindex(columns=feature_cols, fill_value=0).iloc[-lookback:].fillna(0)
        scaler_x = meta.get("scaler_x") if isinstance(meta, dict) else None
        if scaler_x is not None:
            x_np = scaler_x.transform(feature_window)
        else:
            x_np = feature_window.values
        x_np = np.asarray(x_np, dtype=np.float32)

        close = df["Close"].values
        log_returns = np.log(close[1:] / close[:-1])

        scaler = meta.get("scaler", {}) if isinstance(meta, dict) else {}
        mean_r = float(scaler.get("mean", 0.0))
        std_r = float(scaler.get("std", 1.0))
        last_ret_norm = float((log_returns[-1] - mean_r) / (std_r + 1e-8))

        x_t = torch.from_numpy(x_np).unsqueeze(0)
        last_t = torch.tensor([[[last_ret_norm]]], dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            output, attention = model(x_t, last_t)

        out = output.detach().cpu().numpy().squeeze(0)
        point_r = out[:, 1] * (std_r + 1e-8) + mean_r
        lower_r = out[:, 0] * (std_r + 1e-8) + mean_r
        upper_r = out[:, 2] * (std_r + 1e-8) + mean_r

        forecast, lower_95, upper_95 = [], [], []
        prev_point = float(close[-1])
        prev_lower = float(close[-1])
        prev_upper = float(close[-1])

        n_native = len(point_r)
        n_required = int(horizon)
        for i in range(n_required):
            idx = i if i < n_native else n_native - 1
            prev_point = prev_point * float(np.exp(point_r[idx]))
            prev_lower = prev_lower * float(np.exp(lower_r[idx]))
            prev_upper = prev_upper * float(np.exp(upper_r[idx]))
            forecast.append(float(prev_point))
            lower_95.append(float(prev_lower))
            upper_95.append(float(prev_upper))

        # Build business-day future dates
        dates = []
        d = pd.Timestamp(df.index[-1])
        while len(dates) < n_required:
            d += pd.Timedelta(days=1)
            if d.weekday() < 5:
                dates.append(d.strftime("%Y-%m-%d"))

        attention_weights = attention.detach().cpu().numpy().squeeze(0).tolist()

        return {
            "asset": asset,
            "date": str(pd.Timestamp(df.index[-1]).date()),
            "method": "lstm",
            "forecast": forecast,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "dates": dates,
            "attention_weights": attention_weights,
            "model_info": {
                "type": "lstm_seq2seq",
                "mode": scaler.get("mode", "returns"),
                "lookback": lookback,
            },
        }
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"LSTM Seq2Seq forecast unavailable for {asset}: {e}")
        raise


def transformer_forecast(asset, horizon=30):
    """Generate Transformer forecast for an asset.

    Args:
        asset: Asset name (e.g., 'SP500')
        horizon: Forecast horizon in days (default 30)

    Returns:
        dict with forecast, lower_95, upper_95 (or empty dict if model unavailable)
    """
    try:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        model, meta = _load_dl_model(asset, 'transformer')
        features_data = load_all_features()
        if asset not in features_data:
            raise ValueError(f"No feature data found for {asset}")

        df = features_data[asset]
        lookback = int(getattr(model, "lookback", 30))
        if len(df) < lookback + 2:
            raise ValueError(f"Insufficient history for {asset}: need at least {lookback + 2} rows")

        feature_cols = meta.get("feature_cols") if isinstance(meta, dict) else None
        if not feature_cols:
            drop_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            feature_cols = [c for c in df.columns if c not in drop_cols]

        feature_window = df.reindex(columns=feature_cols, fill_value=0).iloc[-lookback:].fillna(0)
        scaler_x = meta.get("scaler_x") if isinstance(meta, dict) else None
        if scaler_x is not None:
            x_np = scaler_x.transform(feature_window)
        else:
            x_np = feature_window.values
        x_np = np.asarray(x_np, dtype=np.float32)

        close = df["Close"].values
        log_returns = np.log(close[1:] / close[:-1])

        scaler = meta.get("scaler", {}) if isinstance(meta, dict) else {}
        mean_r = float(scaler.get("mean", 0.0))
        std_r = float(scaler.get("std", 1.0))
        last_ret_norm = float((log_returns[-1] - mean_r) / (std_r + 1e-8))

        x_t = torch.from_numpy(x_np).unsqueeze(0)
        last_t = torch.tensor([[[last_ret_norm]]], dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            output, feature_weights = model(x_t, last_t)

        out = output.detach().cpu().numpy().squeeze(0)
        point_r = out[:, 1] * (std_r + 1e-8) + mean_r
        lower_r = out[:, 0] * (std_r + 1e-8) + mean_r
        upper_r = out[:, 2] * (std_r + 1e-8) + mean_r

        forecast, lower_95, upper_95 = [], [], []
        prev_point = float(close[-1])
        prev_lower = float(close[-1])
        prev_upper = float(close[-1])

        n_native = len(point_r)
        n_required = int(horizon)
        for i in range(n_required):
            idx = i if i < n_native else n_native - 1
            prev_point = prev_point * float(np.exp(point_r[idx]))
            prev_lower = prev_lower * float(np.exp(lower_r[idx]))
            prev_upper = prev_upper * float(np.exp(upper_r[idx]))
            forecast.append(float(prev_point))
            lower_95.append(float(prev_lower))
            upper_95.append(float(prev_upper))

        dates = []
        d = pd.Timestamp(df.index[-1])
        while len(dates) < n_required:
            d += pd.Timedelta(days=1)
            if d.weekday() < 5:
                dates.append(d.strftime("%Y-%m-%d"))

        fw = feature_weights.detach().cpu().numpy().squeeze(0)
        mean_weights = fw.mean(axis=0) if fw.ndim == 2 else fw
        feature_importance = {
            str(col): float(w) for col, w in zip(feature_cols, mean_weights)
        }
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        )

        return {
            "asset": asset,
            "date": str(pd.Timestamp(df.index[-1]).date()),
            "method": "transformer",
            "forecast": forecast,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "dates": dates,
            "feature_weights": feature_importance,
            "model_info": {
                "type": "transformer",
                "mode": scaler.get("mode", "returns"),
                "lookback": lookback,
            },
        }
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"Transformer forecast unavailable for {asset}: {e}")
        raise


