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

        # Decoder: LSTM for next N days
        self.decoder = nn.LSTM(
            input_size=hidden_size,  # Changed from input_size to hidden_size
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

    def forward(self, x, target=None):
        """
        Args:
            x: (batch, lookback, 1) - past 30 days of prices
            target: (batch, horizon, 1) - next N days (for teacher forcing)
        Returns:
            output: (batch, horizon, 3) - predictions for 3 quantiles
            attention_weights: (batch, horizon, lookback)
        """
        batch_size = x.size(0)

        # Encoder: get context from past 30 days
        encoder_out, (h_n, c_n) = self.encoder(x)  # (batch, lookback, hidden)

        # Attention
        attn_out, attn_weights = self.attention(
            query=encoder_out, key=encoder_out, value=encoder_out
        )  # (batch, lookback, hidden), (batch, lookback, lookback)

        # Decoder: generate next N days
        decoder_input = attn_out  # Use attention output (batch, 1, hidden) instead of encoder_out
        outputs = []
        attention_weights_list = []

        for t in range(self.horizon):
            decoder_out, (h_n, c_n) = self.decoder(
                decoder_input, (h_n, c_n)
            )  # (batch, 1, hidden)

            # Apply attention
            attn_out, attn_w = self.attention(
                query=decoder_out, key=encoder_out, value=encoder_out
            )  # (batch, 1, hidden)
            attention_weights_list.append(
                attn_w.squeeze(1)
            )  # (batch, lookback)

            # Predict quantiles
            pred = self.fc(attn_out)  # (batch, 1, 3)
            outputs.append(pred)

            # Teacher forcing or use own prediction
            if target is not None:
                decoder_input = target[:, t : t + 1, :]
            else:
                decoder_input = pred[:, :, 1:2]  # Use point estimate

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

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

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

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, self.num_quantiles),
        )

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

    def forward(self, x):
        """
        Args:
            x: (batch, lookback, 1) - past 30 days
        Returns:
            output: (batch, horizon, 3) - quantile predictions
            attention_weights: (batch, nhead, horizon, lookback)
        """
        batch_size = x.size(0)

        # Project to d_model
        x = self.input_proj(x)  # (batch, lookback, d_model)

        # Add positional encoding (ensure size matches)
        pos = self.pos_encoder[:, : x.size(1), :].to(x.device)
        if pos.size(0) == 1:
            pos = pos.expand(batch_size, -1, -1)
        x = x + pos[:batch_size, :x.size(1), :]

        # Transformer
        transformer_out = self.transformer_encoder(x)  # (batch, lookback, d_model)

        # Decode for future steps - use average context
        context = transformer_out.mean(dim=1)  # (batch, d_model)

        outputs = []
        for t in range(self.horizon):
            pred = self.decoder(context.unsqueeze(1))  # (batch, 1, 3)
            outputs.append(pred)

        output = torch.cat(outputs, dim=1)  # (batch, horizon, 3)

        return output, None  # Attention weights would require model modification


def create_sequences(data, lookback=30, horizon=30):
    """Create train sequences for LSTM/Transformer.

    Args:
        data: (n_samples,) array of prices
        lookback: Number of past steps to use
        horizon: Number of future steps to predict

    Returns:
        X: (n_sequences, lookback, 1) - past prices
        Y: (n_sequences, horizon, 1) - future prices
    """
    X, Y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback].reshape(-1, 1))
        Y.append(data[i + lookback : i + lookback + horizon].reshape(-1, 1))
    return np.array(X), np.array(Y)


def train_lstm_seq2seq(
    close_prices, asset_name, lookback=30, horizon=30, epochs=50, batch_size=32
):
    """Train LSTM Seq2Seq model.

    Args:
        close_prices: (n_samples,) close prices
        asset_name: Asset name for logging
        lookback: Past window size
        horizon: Future prediction horizon
        epochs: Training epochs
        batch_size: Batch size

    Returns:
        model: Trained model
        scaler: MinMax scaler (min, max values)
    """
    # Normalize
    min_val = close_prices.min()
    max_val = close_prices.max()
    normalized = (close_prices - min_val) / (max_val - min_val + 1e-8)

    # Create sequences
    X, Y = create_sequences(normalized, lookback, horizon)

    # Train/test split (70/30)
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.FloatTensor(Y_train)
    X_test = torch.FloatTensor(X_test)
    Y_test = torch.FloatTensor(Y_test)

    # DataLoader
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSeq2Seq(
        input_size=1,
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

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            output, _ = model(X_batch, Y_batch)
            loss = criterion(output, Y_batch)
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
            output_test, _ = model(X_test_device)
            val_loss = criterion(output_test, Y_test_device).item()

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

    scaler = {"min": float(min_val), "max": float(max_val)}
    return model.cpu(), scaler


def train_transformer(
    close_prices, asset_name, lookback=30, horizon=30, epochs=50, batch_size=32
):
    """Train Transformer model (similar to LSTM).

    Args:
        close_prices: (n_samples,) close prices
        asset_name: Asset name for logging
        lookback: Past window size
        horizon: Future prediction horizon
        epochs: Training epochs
        batch_size: Batch size

    Returns:
        model: Trained model
        scaler: MinMax scaler
    """
    # Normalize
    min_val = close_prices.min()
    max_val = close_prices.max()
    normalized = (close_prices - min_val) / (max_val - min_val + 1e-8)

    # Create sequences
    X, Y = create_sequences(normalized, lookback, horizon)

    # Train/test split
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.FloatTensor(Y_train)
    X_test = torch.FloatTensor(X_test)
    Y_test = torch.FloatTensor(Y_test)

    # DataLoader
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerForecaster(
        input_size=1,
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

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            output, _ = model(X_batch)
            loss = criterion(output, Y_batch)
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
            output_test, _ = model(X_test_device)
            val_loss = criterion(output_test, Y_test_device).item()

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

    scaler = {"min": float(min_val), "max": float(max_val)}
    return model.cpu(), scaler


def forecast_dl_model(model, last_prices, horizon=30, scaler=None, model_type="lstm"):
    """Generate forecast from trained DL model.

    Args:
        model: Trained LSTM or Transformer model
        last_prices: (lookback,) last N prices (normalized)
        horizon: Prediction horizon
        scaler: MinMax scaler info
        model_type: "lstm" or "transformer"

    Returns:
        forecast: (horizon,) point predictions (denormalized)
        lower_95: (horizon,) lower bound
        upper_95: (horizon,) upper bound
        attention: (horizon, lookback) attention weights or None
    """
    model.eval()

    with torch.no_grad():
        X = torch.FloatTensor(last_prices).unsqueeze(0)  # (1, lookback, 1)

        if model_type == "lstm":
            output, attention = model(X)  # (1, horizon, 3), (1, horizon, lookback)
        else:
            output, attention = model(X)  # (1, horizon, 3), None

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
