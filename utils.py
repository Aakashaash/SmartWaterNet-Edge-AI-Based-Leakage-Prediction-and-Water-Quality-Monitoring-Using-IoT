
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error,
)
import warnings
warnings.filterwarnings("ignore")

# --------------------- Global config ---------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# --------------------- Data loading ---------------------
def load_and_split(csv_path):
    """Load CSV and split into 70 / 15 / 15 stratified on both targets."""
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    strat_key = df["leak_state"].astype(str) + "_" + df["quality_class"].astype(str)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=strat_key)

    temp_key = temp_df["leak_state"].astype(str) + "_" + temp_df["quality_class"].astype(str)

    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_key)

    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
    return train_df, val_df, test_df


def print_class_distribution(tag, y):
    u, c = np.unique(y, return_counts=True)
    print(f"  [{tag}] " + " | ".join(f"c{v}:{n}" for v, n in zip(u, c)))


def scale_features(Xtr, Xva, Xte):
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xva), sc.transform(Xte), sc


# --------------------- Tensor / loader helpers ---------------------
def to_tensor(X, y=None):
    Xt = torch.tensor(X, dtype=torch.float32)
    if y is None:
        return Xt
    dt = torch.long if np.issubdtype(np.asarray(y).dtype, np.integer) else torch.float32
    return Xt, torch.tensor(y, dtype=dt)


def make_loader(X, y, batch=128, shuffle=True):
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=shuffle)


# --------------------- Losses ---------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, reduction="none", weight=self.weight)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# --------------------- Training routines ---------------------
def train_classifier(model, tr_loader, va_loader, epochs=100, lr=5e-4,
                     extra_loss=False, class_weights=None, task_name=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = FocalLoss(gamma=2.0, weight=class_weights) if class_weights is not None \
        else nn.CrossEntropyLoss()
    best_score, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            if extra_loss:
                recon, _, q = model(xb)
                loss = crit(q, yb) + F.mse_loss(recon, xb)
            else:
                loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                out = model(xb.to(DEVICE))
                logits = out[2] if extra_loss else out
                yp.extend(logits.argmax(1).cpu().numpy())
                yt.extend(yb.numpy())
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        score = 0.5 * acc + 0.5 * f1
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    print(f"[{task_name}] Best Val Acc/F1 score: {best_score:.4f}")
    return model


def train_regressor(model, tr, va, epochs=100, lr=5e-4, task_name=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    best_rmse, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        pr, tr_ = [], []
        with torch.no_grad():
            for xb, yb in va:
                pr.append(model(xb.to(DEVICE)).cpu().numpy())
                tr_.append(yb.numpy())
        pr = np.concatenate(pr)
        tr_ = np.concatenate(tr_)
        rmse = np.sqrt(mean_squared_error(tr_.reshape(pr.shape), pr))
        if rmse < best_rmse:
            best_rmse, best_state = rmse, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    print(f"[{task_name}] Best Val RMSE: {best_rmse:.4f}")
    return model


def eval_classifier(model, loader, class_names, tag, adqcl=False):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(DEVICE))
            logits = out[2] if adqcl else out
            yp.extend(logits.argmax(1).cpu().numpy())
            yt.extend(yb.numpy())
    yt, yp = np.array(yt), np.array(yp)
    print(f"\n===== {tag} - TEST =====")
    print(classification_report(yt, yp, target_names=class_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(yt, yp))
    m = {
        "Accuracy": accuracy_score(yt, yp),
        "Precision(macro)": precision_score(yt, yp, average="macro", zero_division=0),
        "Recall(macro)": recall_score(yt, yp, average="macro", zero_division=0),
        "F1(macro)": f1_score(yt, yp, average="macro", zero_division=0),
    }
    cm = confusion_matrix(yt, yp)
    if adqcl and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        m["False Alarm Rate"] = fp / (fp + tn) if (fp + tn) else 0.0
    return m