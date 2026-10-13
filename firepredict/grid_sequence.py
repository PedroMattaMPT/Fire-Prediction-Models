"""Sequence models on the grid (docs/improvement-plan.md, Phase 4.5).

The shipped GRU/BiGRU read 71 h of 14 raw ERA5-Land variables and nothing else,
so they have to infer place and season from weather proxies — and were
early-stopped on the fold they reported. These are trained the grid way instead:

- the **same explicit features** as the tree models (a static vector per sample:
  time, place, fire history, FWI system, dryness — ``static_groups``),
- plus a **feed-robust hourly sequence** (temperature, humidity, wind,
  precipitation, VPD) over the last ``lookback`` hours up to and including the
  scored hour,
- the **same case-base sample** (all fires, uniform negatives),
- early stopping and epoch selection on the **validation year** (sample PR-AUC),
  never on the test period.

They are too slow to score on every cell-hour; the harness scores them with the
weighted-sample estimator (``scripts/evaluate_grid.py --weighted`` puts the tree
models on the same sample for a like-for-like comparison).
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .features import FirePriors
from .fwi import vpd_kpa
from .grid_models import GridContext, Recipe, training_sample

SEQ_VARS: tuple[str, ...] = ("temp", "humidity", "wind_speed", "precip", "vpd")


@dataclass
class SeqRecipe:
    name: str = "gru_explicit"
    arch: str = "gru"                               # "gru" | "bigru" (+ attention pooling)
    static_groups: tuple[str, ...] = ("time", "space", "prior", "fwi", "dry")
    seq_vars: tuple[str, ...] = SEQ_VARS
    lookback: int = 72
    hidden: int = 64
    dropout: float = 0.2
    epochs: int = 30
    batch_size: int = 4096
    lr: float = 2e-3
    patience: int = 4
    neg_ratio: int = config.NEG_RATIO
    exclude_rekindles: bool = True
    extra: dict = field(default_factory=dict)


def _net(arch: str, n_seq: int, n_static: int, hidden: int, dropout: float):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            bi = arch == "bigru"
            self.gru = nn.GRU(n_seq, hidden, num_layers=2, batch_first=True, dropout=dropout, bidirectional=bi)
            out = hidden * (2 if bi else 1)
            self.attn = nn.Linear(out, 1) if bi else None
            self.static = nn.Sequential(nn.Linear(n_static, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 64), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(out + 64, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

        def forward(self, seq, stat):
            o, h = self.gru(seq)
            if self.attn is not None:
                w = torch.softmax(self.attn(o).squeeze(-1), dim=1).unsqueeze(-1)
                z = (o * w).sum(dim=1)
            else:
                z = h[-1]
            return self.head(torch.cat([z, self.static(stat)], dim=1)).squeeze(-1)

    return Net()


class GridSequenceModel:
    """Trained sequence model + the arrays needed to rebuild its inputs."""

    unit = "hour"
    expensive = True

    def __init__(self, recipe: SeqRecipe, state_dict, static_cols, static_mu, static_sd, seq_mu, seq_sd, priors, meta):
        self.recipe, self.state_dict = recipe, state_dict
        self.static_cols, self.static_mu, self.static_sd = static_cols, static_mu, static_sd
        self.seq_mu, self.seq_sd, self.priors, self.meta = seq_mu, seq_sd, priors, meta
        self.name = recipe.name
        self.ctx = self.builder = self.net = None

    def bind(self, ctx: GridContext, device: str | None = None) -> "GridSequenceModel":
        import torch

        self.ctx = ctx
        self.builder = ctx.builder()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        r = self.recipe
        self.net = _net(r.arch, len(r.seq_vars), len(self.static_cols), r.hidden, r.dropout)
        self.net.load_state_dict(self.state_dict)
        self.net.to(self.device).eval()
        return self

    # inputs --------------------------------------------------------------------
    def _seq(self, hi: np.ndarray, ci: np.ndarray) -> np.ndarray:
        return _gather_seq(self.ctx.store, hi, ci, self.recipe.seq_vars, self.recipe.lookback, self.seq_mu, self.seq_sd)

    def _static(self, hi, ci, priors=None) -> np.ndarray:
        X = self.builder.hourly(hi, ci, self.recipe.static_groups, priors or self.priors)[self.static_cols]
        return np.nan_to_num((X.to_numpy(np.float32) - self.static_mu) / self.static_sd)

    def score_rows(self, hi, ci, batch: int = 16384) -> np.ndarray:
        import torch

        hi, ci = np.asarray(hi, dtype=np.int64), np.asarray(ci, dtype=np.int64)
        out = np.empty(len(hi), np.float32)
        with torch.no_grad():
            for i in range(0, len(hi), batch):
                h, c = hi[i:i + batch], ci[i:i + batch]
                s = torch.from_numpy(self._seq(h, c)).to(self.device)
                x = torch.from_numpy(self._static(h, c)).to(self.device)
                out[i:i + batch] = torch.sigmoid(self.net(s, x)).float().cpu().numpy()
        return out

    def score_hours(self, h0, h1):
        C = self.ctx.store.n_cells
        return self.score_rows(np.repeat(np.arange(h0, h1), C), np.tile(np.arange(C), h1 - h0)).reshape(h1 - h0, C)

    def save(self, path: str | Path | None = None) -> Path:
        import torch

        path = Path(path or config.GRID_MODELS_DIR / f"{self.name}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"recipe": asdict(self.recipe), "state_dict": self.state_dict, "static_cols": self.static_cols,
                    "static_mu": self.static_mu, "static_sd": self.static_sd, "seq_mu": self.seq_mu,
                    "seq_sd": self.seq_sd, "priors": self.priors, "meta": self.meta}, path)
        return path

    @classmethod
    def load(cls, path, ctx: GridContext | None = None) -> "GridSequenceModel":
        import torch

        a = torch.load(path, map_location="cpu", weights_only=False)
        rec = a["recipe"]
        for k in ("static_groups", "seq_vars"):
            rec[k] = tuple(rec[k])
        m = cls(SeqRecipe(**rec), a["state_dict"], a["static_cols"], a["static_mu"], a["static_sd"],
                a["seq_mu"], a["seq_sd"], a["priors"], a["meta"])
        return m.bind(ctx) if ctx is not None else m


def _gather_seq(store, hi, ci, seq_vars, lookback, mu, sd) -> np.ndarray:
    """(n, lookback, n_vars) standardised, oldest -> newest, ending at the scored hour."""
    rows = np.clip(hi[:, None] - np.arange(lookback - 1, -1, -1)[None, :], 0, None)
    cols = ci[:, None]
    raw = {}
    for v in set(seq_vars) | ({"temp", "dewpoint"} if "vpd" in seq_vars else set()):
        if v != "vpd":
            raw[v] = np.asarray(store.var(v)[rows, cols], dtype=np.float32)
    if "vpd" in seq_vars:
        raw["vpd"] = vpd_kpa(raw["temp"], raw["dewpoint"]).astype(np.float32)
    x = np.stack([(raw[v] - mu[i]) / sd[i] for i, v in enumerate(seq_vars)], axis=-1)
    return np.nan_to_num(x).astype(np.float32)


def train_sequence(recipe: SeqRecipe, ctx: GridContext, *, test_from, train_start=config.GRID_TRAIN_START,
                   val_years: int = config.GRID_VALIDATION_YEARS, device: str | None = None,
                   seed: int = 0, verbose: bool = True) -> GridSequenceModel:
    """Train on ``[train_start, test_from)``; pick the epoch on the validation year."""
    import torch
    from sklearn.metrics import average_precision_score

    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    st = ctx.store
    log = (lambda *a: print(f"  [{recipe.name}]", *a, f"[{time.time() - t0:.0f}s]", flush=True)) if verbose else (lambda *a: None)
    t_start, t_test = pd.Timestamp(train_start), pd.Timestamp(test_from)
    t_val = t_test - pd.DateOffset(years=val_years)
    base = Recipe(name=recipe.name, groups=recipe.static_groups, neg_ratio=recipe.neg_ratio,
                  exclude_rekindles=recipe.exclude_rekindles)
    U, Cc, y, info = training_sample(base, ctx, t_start, t_test)
    priors = FirePriors(ctx.events, st.n_cells, t_start, t_test)
    builder = ctx.builder()
    Xs = builder.hourly_training(U, Cc, recipe.static_groups, priors)
    static_cols = list(Xs.columns)
    X = Xs.to_numpy(np.float32)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-6
    X = np.nan_to_num((X - mu) / sd).astype(np.float32)
    # per-variable sequence normalisation from a subsample of training sequences
    sub = rng.choice(len(U), min(200_000, len(U)), replace=False)
    probe = _gather_seq(st, U[sub], Cc[sub], recipe.seq_vars, recipe.lookback,
                        np.zeros(len(recipe.seq_vars)), np.ones(len(recipe.seq_vars)))
    seq_mu = probe.reshape(-1, len(recipe.seq_vars)).mean(axis=0)
    seq_sd = probe.reshape(-1, len(recipe.seq_vars)).std(axis=0) + 1e-6
    del probe
    u_val = st.hour_range(t_val, t_val)[0]
    tr, va = np.nonzero(U < u_val)[0], np.nonzero(U >= u_val)[0]
    log(f"sample {info['n_pos']:,}+{info['n_neg']:,}; static {len(static_cols)}; seq {recipe.lookback}h x {len(recipe.seq_vars)}; "
        f"train {len(tr):,} / val {len(va):,}; device {dev}")

    net = _net(recipe.arch, len(recipe.seq_vars), len(static_cols), recipe.hidden, recipe.dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=recipe.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=recipe.epochs, eta_min=recipe.lr / 20)
    lossf = torch.nn.BCEWithLogitsLoss()
    yt = torch.from_numpy(y.astype(np.float32))

    def batches(idx, shuffle):
        idx = rng.permutation(idx) if shuffle else idx
        for i in range(0, len(idx), recipe.batch_size):
            yield idx[i:i + recipe.batch_size]

    def run_eval(idx):
        net.eval()
        out = np.empty(len(idx), np.float32)
        with torch.no_grad():
            for i in range(0, len(idx), 16384):
                b = idx[i:i + 16384]
                s = torch.from_numpy(_gather_seq(st, U[b], Cc[b], recipe.seq_vars, recipe.lookback, seq_mu, seq_sd)).to(dev)
                x = torch.from_numpy(X[b]).to(dev)
                out[i:i + len(b)] = torch.sigmoid(net(s, x)).float().cpu().numpy()
        return out

    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    for ep in range(1, recipe.epochs + 1):
        net.train()
        tot = 0.0
        for b in batches(tr, True):
            s = torch.from_numpy(_gather_seq(st, U[b], Cc[b], recipe.seq_vars, recipe.lookback, seq_mu, seq_sd)).to(dev)
            x = torch.from_numpy(X[b]).to(dev)
            opt.zero_grad()
            loss = lossf(net(s, x), yt[b].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        ap = float(average_precision_score(y[va], run_eval(va)))
        mark = ap > best + 1e-5
        log(f"epoch {ep:>2}: train loss {tot / len(tr):.4f}  validation-year sample PR-AUC {ap:.4f}{'  *' if mark else ''}")
        if mark:
            best, best_state, best_epoch, bad = ap, copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()}), ep, 0
        else:
            bad += 1
            if bad >= recipe.patience:
                break
    meta = {"region": config.REGION, "protocol": ctx.protocol, "train_start": str(t_start), "validation_from": str(t_val),
            "test_from": str(t_test), "sample": info, "best_epoch": best_epoch,
            "validation_sample_pr_auc": best, "seconds": round(time.time() - t0, 1), "device": dev}
    m = GridSequenceModel(recipe, best_state, static_cols, mu.astype(np.float32), sd.astype(np.float32),
                          seq_mu.astype(np.float32), seq_sd.astype(np.float32), priors.arrays(None), meta)
    return m.bind(ctx, dev)
