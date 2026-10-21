

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from utils import DEVICE, make_loader, to_tensor


class Autoencoder(nn.Module):
    def __init__(self, in_dim, latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, latent), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(),
            nn.Linear(64, in_dim))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class ADQCL(nn.Module):
    def __init__(self, in_dim, latent=32, n_actions=2):
        super().__init__()
        self.ae = Autoencoder(in_dim, latent)
        self.q_head = nn.Sequential(
            nn.Linear(latent, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, n_actions))

    def forward(self, x):
        recon, z = self.ae(x)
        return recon, z, self.q_head(z)


def estimate_fisher(model, loader, n_batches=10):
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    crit = nn.CrossEntropyLoss()
    count = 0
    for xb, yb in loader:
        if count >= n_batches:
            break
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        _, _, q = model(xb)
        loss = crit(q, yb)
        model.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)
        count += 1
    for n in fisher:
        fisher[n] /= max(count, 1)
    return fisher


def ewc_penalty(model, fisher, prev_params, lam=1e2):
    loss = 0.0
    for n, p in model.named_parameters():
        if n in fisher:
            loss = loss + (fisher[n] * (p - prev_params[n]).pow(2)).sum()
    return lam * loss


def train_adqcl_continual(model, batches, val_loader, epochs_per_task=100, lr=5e-4,
                          lam_ewc=1e2, task_name="ADQ-CL"):
    print(f"\n--- {task_name}: Continual Learning across {len(batches)} rounds ---")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5], device=DEVICE),
                               label_smoothing=0.05)
    fisher, prev_params = None, None
    for r, (Xb, yb) in enumerate(batches):
        loader = make_loader(*to_tensor(Xb, yb))
        best_state, best_score = None, -1
        for ep in range(epochs_per_task):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                recon, _, q = model(xb)
                loss = crit(q, yb) + F.mse_loss(recon, xb)
                if fisher is not None and prev_params is not None:
                    loss = loss + ewc_penalty(model, fisher, prev_params, lam_ewc)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            yt, yp = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    _, _, logits = model(xb.to(DEVICE))
                    yp.extend(logits.argmax(1).cpu().numpy())
                    yt.extend(yb.numpy())
            acc = accuracy_score(yt, yp)
            f1 = f1_score(yt, yp, average="macro", zero_division=0)
            score = 0.5 * acc + 0.5 * f1
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        fisher = estimate_fisher(model, loader)
        prev_params = {n: p.detach().clone() for n, p in model.named_parameters()}
        print(f"  Round {r+1}: val score = {best_score:.4f}")
    return model