import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Utility: MEE Metric (PyTorch)
# -----------------------------
def mee_torch(y_true, y_pred):
    """
    Compute Mean Euclidean Error in Torch.
    Input shape: (n_samples, 4)
    """
    diffs = y_pred - y_true
    dists = torch.norm(diffs, dim=1)
    return dists.mean().item()


# -----------------------------
# Custom MLP with He Init + Dropout
# -----------------------------
class AdvancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim=4,
                 activation="relu", dropout=0.0):
        super().__init__()

        act_layer = nn.ReLU if activation == "relu" else nn.Tanh

        layers = []
        prev = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, output_dim))

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # He initialization for ReLU
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train_mlp_advanced(
        X_train, y_train,
        X_val, y_val,
        hidden_layers=[64, 32],
        activation="relu",
        lr=1e-3,
        weight_decay=1e-4,
        dropout=0.0,
        batch_size=32,
        max_epochs=3000,
        patience=200,
        clip_value=5.0,
        scheduler_patience=50,
        seed=0,
        verbose=False):

    # Ensure reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_v  = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v  = torch.tensor(y_val, dtype=torch.float32).to(device)

    # DataLoader
    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model
    model = AdvancedMLP(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        output_dim=y_train.shape[1],   # <-- ADD THIS LINE
        activation=activation,
        dropout=dropout
    ).to(device)

    # Optimizer: ADAM (better for regression than SGD)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler (reduce on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=scheduler_patience    )

    best_val_mee = float("inf")
    best_state = None
    epochs_no_improve = 0

    history = {"train_mee": [], "val_mee": [], "lr": []}

    for epoch in range(max_epochs):

        # --- Train ---
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = nn.MSELoss()(preds, yb)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

        # --- Evaluate ---
        model.eval()
        with torch.no_grad():
            train_pred = model(X_tr)
            val_pred = model(X_v)

        train_mee = mee_torch(y_tr, train_pred)
        val_mee = mee_torch(y_v, val_pred)

        history["train_mee"].append(train_mee)
        history["val_mee"].append(val_mee)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # --- Scheduler step ---
        scheduler.step(val_mee)

        # --- Early stopping ---
        if val_mee < best_val_mee - 1e-6:
            best_val_mee = val_mee
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and epoch % 100 == 0:
            print(
                f"Epoch {epoch} | Train MEE={train_mee:.4f}, "
                f"Val MEE={val_mee:.4f}, LR={optimizer.param_groups[0]['lr']:.2e}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    # Load best model state
    model.load_state_dict(best_state)

    return model, history, best_val_mee
