"""Daily neural and alternative tabular models on the grid — the model-family comparison.

Every model here gets what the production tree model gets (recipe R5, daily unit,
case-base sample, validation year for early stopping) and is scored on every
cell-day of the test period, so it can be compared with ``firerisk_daily`` on the
grid harness. Families:

tabular (the R5 feature vector only)
    ``catboost``  CatBoost gradient boosting (Apache 2.0)
    ``mlp``       plain MLP — the neural reference
    ``tabm``      TabM, parameter-efficient ensemble of MLPs (Apache 2.0)
    ``ftt``       FT-Transformer, attention across features (Apache 2.0)

temporal (R5 vector + the last ``lookback`` days of daily weather, FWI and fires)
    ``gru``, ``lstm``   recurrent
    ``slstm``           xLSTM's sLSTM cell (exponential gating), in plain PyTorch
    ``tcn``             dilated causal convolutions
    ``transformer``     self-attention across days
    ``tft``             Temporal Fusion Transformer (Lim et al. 2021): static covariate
                        encoders, variable selection, LSTM, interpretable attention
    ``spatial``         attention across the cell and its 24 ring-1/2 neighbours on the day

The daily sequence channels are the store's daily arrays plus fire counts in the
cell and its ring-1 neighbours, shifted by ``config.RECENCY_LAG_DAYS`` so a fire is
only seen once it is that old (the same rule as the ``recency`` features).
``NeuralMember`` is the servable form of a trained network: ``firerisk_daily`` is
XGBoost + an sLSTM member (``Recipe.neural``), averaged in log-odds before
calibration (``grid_models.GridModel``).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import pandas as pd

from . import config
from .grid_models import GridContext, Recipe, training_sample
from .history_features import ring_matrix

SEQ_DAILY: tuple[str, ...] = ("tmax", "rhmin", "wsmax", "vpdmax", "rain24", "ffmc", "dmc", "dc", "isi", "fwi")
SEQ_FIRE: tuple[str, ...] = ("own_fires", "nbr1_fires")
TEMPORAL = ("gru", "lstm", "slstm", "tcn", "transformer", "tft", "spatial")
TABULAR_NN = ("mlp", "tabm", "ftt")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def daily_cube(store, events: pd.DataFrame, lag_days: int) -> np.ndarray:
    """Raw (days, cells, channels) daily sequence channels for any store (the weather
    store or the live window): ``SEQ_DAILY`` arrays (rain as log1p), then log1p fire
    counts in the cell and its ring-1 neighbours on the day ``lag_days`` earlier —
    counted from the events by date, so fires before the store's first day count too."""
    chans = [np.nan_to_num(np.asarray(store.daily(v), dtype=np.float32)) for v in SEQ_DAILY]
    k = SEQ_DAILY.index("rain24")
    chans[k] = np.log1p(np.maximum(chans[k], 0))
    f = events[events["cell"] >= 0]
    di = store.day_index(pd.to_datetime(f["day"]) + pd.Timedelta(days=int(lag_days)))
    ok = di >= 0
    F = np.zeros((store.n_days, store.n_cells), np.float32)
    np.add.at(F, (di[ok], f["cell"].to_numpy()[ok]), 1.0)
    chans += [np.log1p(F), np.log1p(F @ ring_matrix(store.cells, 1))]
    return np.stack(chans, axis=-1)


class DailyData:
    """R5 training sample + transformed static matrix + daily sequence cube, for one split."""

    def __init__(self, ctx: GridContext, *, test_from, train_start=config.GRID_TRAIN_START, val_years: int = 1,
                 lookback: int = 90, recipe: "str | Recipe" = "R5", sample=None, log=print):
        """``sample`` = ``(U, C, y, X)`` from ``grid_models.train`` (X the raw feature frame)
        skips drawing the sample and building the features again."""
        st = ctx.store
        self.ctx, self.lookback, self.log = ctx, int(lookback), log
        self.recipe = Recipe.named(recipe, unit="day") if isinstance(recipe, str) else recipe
        self.t_start, self.t_test = pd.Timestamp(train_start), pd.Timestamp(test_from)
        t_val = self.t_test - pd.DateOffset(years=val_years)
        self.builder = ctx.builder(lag_days=self.recipe.recency_lag_days)
        self.priors = ctx.priors(self.recipe, self.t_start, self.t_test)
        t0 = time.time()
        if sample is None:
            U, C, y, info = training_sample(self.recipe, ctx, self.t_start, self.t_test)
            X = self.builder.daily_training(U, C, self.recipe.groups, self.priors)
            log(f"sample {info['n_pos']:,} + {info['n_neg']:,}; features {X.shape} [{time.time() - t0:.0f}s]")
        else:
            U, C, y, X = sample
        self.cols = list(X.columns)
        self.U, self.C, self.y = np.asarray(U, np.int64), np.asarray(C, np.int64), np.asarray(y, np.float32)
        u_val = st.day_range(t_val, t_val)[0]
        self.tr, self.va = self.U < u_val, self.U >= u_val
        self.X_raw = X.to_numpy(np.float32)
        # static transform: quantile -> normal, fitted on training rows
        from sklearn.preprocessing import QuantileTransformer

        rng = np.random.default_rng(0)
        fit_rows = rng.choice(np.nonzero(self.tr)[0], min(300_000, int(self.tr.sum())), replace=False)
        self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10**9,
                                      random_state=0).fit(np.nan_to_num(self.X_raw[fit_rows]))
        self.X = self._static(self.X_raw)
        self._cube(u_val)
        self._neighbours()

    def _static(self, X_raw: np.ndarray) -> np.ndarray:
        return np.clip(self.qt.transform(np.nan_to_num(X_raw)), -5, 5).astype(np.float32)

    def _cube(self, u_val: int) -> None:
        """(days, cells, channels) standardized daily cube; fire channels lagged."""
        st = self.ctx.store
        cube = daily_cube(st, self.ctx.events, self.recipe.recency_lag_days)
        d0 = st.day_range(self.t_start, self.t_start)[0]
        ref = cube[d0:u_val]
        self.seq_mu = ref.reshape(-1, cube.shape[-1]).mean(axis=0)
        self.seq_sd = ref.reshape(-1, cube.shape[-1]).std(axis=0) + 1e-6
        self.cube = ((cube - self.seq_mu) / self.seq_sd).astype(np.float32)
        self.n_seq = cube.shape[-1]

    def _neighbours(self) -> None:
        """(C, 25) index of the cell and its ring-1/2 neighbours, offsets 0..24, and a validity mask."""
        cells = self.ctx.store.cells
        step = config.WEATHER_GRID_STEP
        i = np.rint(cells["slat"].to_numpy() / step).astype(int)
        j = np.rint(cells["slon"].to_numpy() / step).astype(int)
        lut = {(a, b): k for k, (a, b) in enumerate(zip(i, j))}
        C = len(cells)
        idx = np.zeros((C, 25), np.int64)
        mask = np.zeros((C, 25), bool)
        for c in range(C):
            for o, (da, db) in enumerate((a, b) for a in range(-2, 3) for b in range(-2, 3)):
                k = lut.get((i[c] + da, j[c] + db))
                idx[c, o] = c if k is None else k
                mask[c, o] = k is not None
        self.nbr_idx, self.nbr_mask = idx, mask

    def grid_static(self, d0: int, d1: int, chunk: int = 60) -> np.ndarray:
        """Transformed R5 features for days [d0, d1) x every cell, day-major."""
        pri = self.priors.arrays(None)
        parts = []
        for a in range(d0, d1, chunk):
            b = min(a + chunk, d1)
            parts.append(self._static(self.builder.daily_block(a, b, self.recipe.groups, pri).to_numpy(np.float32)))
        return np.vstack(parts)

    def grid_raw(self, d0: int, d1: int, chunk: int = 60) -> np.ndarray:
        pri = self.priors.arrays(None)
        return np.vstack([self.builder.daily_block(a, min(a + chunk, d1), self.recipe.groups, pri).to_numpy(np.float32)
                          for a in range(d0, d1, chunk)])


# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------
def _nets():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class StaticMLP(nn.Module):
        def __init__(self, n_in, d=256, depth=3, dropout=0.1):
            super().__init__()
            layers, k = [], n_in
            for _ in range(depth):
                layers += [nn.Linear(k, d), nn.ReLU(), nn.Dropout(dropout)]
                k = d
            self.net = nn.Sequential(*layers)
            self.d_out = d

        def forward(self, x):
            return self.net(x)

    class Head(nn.Module):
        """Sequence encoder output + static MLP -> logit."""

        def __init__(self, n_static, d_seq):
            super().__init__()
            self.stat = StaticMLP(n_static, 256, 2)
            self.out = nn.Sequential(nn.Linear(256 + d_seq, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))

        def forward(self, s, z):
            return self.out(torch.cat([self.stat(s), z], dim=-1)).squeeze(-1)

    class MLP(nn.Module):
        def __init__(self, n_static, n_seq=0):
            super().__init__()
            self.m = StaticMLP(n_static, 512, 3)
            self.o = nn.Linear(512, 1)

        def forward(self, s, seq=None, nbr=None):
            return self.o(self.m(s)).squeeze(-1)

    class RNN(nn.Module):
        def __init__(self, n_static, n_seq, cell="gru", d=64):
            super().__init__()
            self.rnn = (nn.GRU if cell == "gru" else nn.LSTM)(n_seq, d, num_layers=2, batch_first=True, dropout=0.1)
            self.head = Head(n_static, d)

        def forward(self, s, seq, nbr=None):
            o, _ = self.rnn(seq)
            return self.head(s, o[:, -1])

    class SLSTM(nn.Module):
        """sLSTM (Beck et al. 2024): exponential input/forget gates with a log-space
        stabilizer and a normalizer state; recurrent connections from h."""

        def __init__(self, n_static, n_seq, d=64):
            super().__init__()
            self.d = d
            self.wx = nn.Linear(n_seq, 4 * d)
            self.rh = nn.Linear(d, 4 * d, bias=False)
            self.norm = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            self.head = Head(n_static, d)

        def forward(self, s, seq, nbr=None):
            B, L, _ = seq.shape
            xs = self.wx(seq)
            h = seq.new_zeros(B, self.d)
            c = seq.new_zeros(B, self.d)
            n = seq.new_ones(B, self.d)
            m = seq.new_zeros(B, self.d)
            for t in range(L):
                zi, fi, zz, oo = (xs[:, t] + self.rh(h)).float().chunk(4, dim=-1)
                logf = F.logsigmoid(fi)
                m_new = torch.maximum(logf + m, zi)
                i_g = torch.exp(zi - m_new)
                f_g = torch.exp(logf + m - m_new)
                c = f_g * c + i_g * torch.tanh(zz)
                n = f_g * n + i_g
                h = torch.sigmoid(oo) * c / n.clamp_min(1e-6)
                m = m_new
            z = self.norm(h)
            return self.head(s, z + self.ffn(z))

    class TCN(nn.Module):
        def __init__(self, n_static, n_seq, d=64, dilations=(1, 2, 4, 8, 16, 32)):
            super().__init__()
            self.inp = nn.Conv1d(n_seq, d, 1)
            self.blocks = nn.ModuleList(nn.Conv1d(d, d, 3, dilation=q) for q in dilations)
            self.dil = dilations
            self.head = Head(n_static, d)

        def forward(self, s, seq, nbr=None):
            x = self.inp(seq.transpose(1, 2))
            for conv, q in zip(self.blocks, self.dil):
                x = x + F.gelu(conv(F.pad(x, (2 * q, 0))))
            return self.head(s, x[:, :, -1])

    class SeqTransformer(nn.Module):
        def __init__(self, n_static, n_seq, lookback, d=64, heads=4, layers=2):
            super().__init__()
            self.inp = nn.Linear(n_seq, d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.pos = nn.Parameter(torch.randn(1, lookback + 1, d) * 0.02)
            enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.head = Head(n_static, d)

        def forward(self, s, seq, nbr=None):
            x = self.inp(seq)
            x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos[:, : x.shape[1] + 1]
            return self.head(s, self.enc(x)[:, 0])

    class GRN(nn.Module):
        def __init__(self, d_in, d_hid, d_out, d_ctx=None, dropout=0.1):
            super().__init__()
            self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
            self.fc1 = nn.Linear(d_in, d_hid)
            self.ctx = nn.Linear(d_ctx, d_hid, bias=False) if d_ctx else None
            self.fc2 = nn.Linear(d_hid, d_hid)
            self.gate = nn.Linear(d_hid, 2 * d_out)
            self.norm = nn.LayerNorm(d_out)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, c=None):
            h = self.fc1(x)
            if self.ctx is not None and c is not None:
                h = h + self.ctx(c)
            h = self.drop(self.fc2(F.elu(h)))
            return self.norm(self.skip(x) + F.glu(self.gate(h), dim=-1))

    class TFT(nn.Module):
        """Temporal Fusion Transformer (Lim et al. 2021), sized for this task."""

        def __init__(self, n_static, n_seq, d=64, heads=4):
            super().__init__()
            self.n_seq, self.d, self.heads = n_seq, d, heads
            self.static_enc = GRN(n_static, 256, d)
            self.ctx = nn.ModuleList(GRN(d, d, d) for _ in range(4))          # selection, h0, c0, enrichment
            self.var_emb = nn.Parameter(torch.randn(n_seq, d) * 0.1)          # per-variable scalar embedding
            self.var_bias = nn.Parameter(torch.zeros(n_seq, d))
            self.var_grn = GRN(d, d, d)
            self.sel = GRN(n_seq * d, d, n_seq, d_ctx=d)
            self.lstm = nn.LSTM(d, d, batch_first=True)
            self.post_lstm = nn.LayerNorm(d)
            self.enrich = GRN(d, d, d, d_ctx=d)
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            self.v = nn.Linear(d, d // heads)                                  # shared value (interpretable)
            self.att_out = nn.Linear(d // heads, d)
            self.post_att = nn.LayerNorm(d)
            self.pw = GRN(d, d, d)
            self.out = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, s, seq, nbr=None):
            B, L, V = seq.shape
            st = self.static_enc(s)
            c_sel, c_h, c_c, c_en = (g(st) for g in self.ctx)
            e = self.var_grn(seq.unsqueeze(-1) * self.var_emb + self.var_bias)          # (B, L, V, d)
            w = torch.softmax(self.sel(e.reshape(B, L, V * self.d), c_sel.unsqueeze(1).expand(-1, L, -1)), dim=-1)
            x = (w.unsqueeze(-1) * e).sum(dim=2)                                       # (B, L, d)
            o, _ = self.lstm(x, (c_h.unsqueeze(0).contiguous(), c_c.unsqueeze(0).contiguous()))
            o = self.post_lstm(o + x)
            en = self.enrich(o, c_en.unsqueeze(1).expand(-1, L, -1))
            dh = self.d // self.heads
            q = self.q(en[:, -1:]).view(B, 1, self.heads, dh).transpose(1, 2)
            k = self.k(en).view(B, L, self.heads, dh).transpose(1, 2)
            a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1).mean(dim=1)   # (B, 1, L)
            att = self.att_out(a @ self.v(en))                                         # (B, 1, d)
            z = self.pw(self.post_att(att[:, 0] + en[:, -1]))
            return self.out(torch.cat([z, st], dim=-1)).squeeze(-1)

    class Spatial(nn.Module):
        """The cell's static vector queries the day's channels of itself and its 24 neighbours."""

        def __init__(self, n_static, n_seq, d=64, heads=4):
            super().__init__()
            self.kv = nn.Linear(n_seq, d)
            self.pos = nn.Parameter(torch.randn(25, d) * 0.02)
            self.q = nn.Linear(n_static, d)
            self.att = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
            self.head = Head(n_static, d)

        def forward(self, s, seq, nbr):
            feats, mask = nbr
            kv = self.kv(feats) + self.pos
            z, _ = self.att(self.q(s).unsqueeze(1), kv, kv, key_padding_mask=~mask)
            return self.head(s, z[:, 0])

    class TabMNet(nn.Module):
        def __init__(self, n_static, n_seq=0):
            super().__init__()
            import tabm

            self.m = tabm.TabM.make(n_num_features=n_static, d_out=1)

        def forward(self, s, seq=None, nbr=None):
            return self.m(s).squeeze(-1)                                               # (B, k)

    class FTT(nn.Module):
        def __init__(self, n_static, n_seq=0):
            super().__init__()
            from rtdl_revisiting_models import FTTransformer

            self.m = FTTransformer(n_cont_features=n_static, cat_cardinalities=[], d_out=1,
                                   **FTTransformer.get_default_kwargs())

        def forward(self, s, seq=None, nbr=None):
            return self.m(s, None).squeeze(-1)

    return {"mlp": MLP, "gru": lambda a, b, L: RNN(a, b, "gru"), "lstm": lambda a, b, L: RNN(a, b, "lstm"),
            "slstm": lambda a, b, L: SLSTM(a, b), "tcn": lambda a, b, L: TCN(a, b),
            "transformer": lambda a, b, L: SeqTransformer(a, b, L), "tft": lambda a, b, L: TFT(a, b),
            "spatial": lambda a, b, L: Spatial(a, b), "tabm": TabMNet, "ftt": FTT}


# -----------------------------------------------------------------------------
# Training / scoring
# -----------------------------------------------------------------------------
DEFAULTS = {  # (batch, lr, max epochs, patience)
    "mlp": (4096, 1e-3, 20, 3), "tabm": (4096, 2e-3, 20, 3), "ftt": (2048, 3e-4, 10, 2),
    "gru": (4096, 1e-3, 15, 3), "lstm": (4096, 1e-3, 15, 3), "slstm": (4096, 1e-3, 12, 3),
    "tcn": (4096, 1e-3, 15, 3), "transformer": (2048, 5e-4, 12, 3), "tft": (2048, 5e-4, 12, 3),
    "spatial": (4096, 1e-3, 15, 3),
}


class NeuralModel:
    def __init__(self, name: str, data: DailyData, device: str = "cuda", seed: int = 0, log=print):
        import torch

        self.name, self.data, self.device, self.log = name, data, device, log
        torch.manual_seed(seed)
        np.random.seed(seed)
        L = data.lookback
        make = _nets()[name]
        self.net = (make(len(data.cols), data.n_seq, L) if name in TEMPORAL else make(len(data.cols))).to(device)
        self.cube = torch.from_numpy(data.cube).to(device)                           # (T, C, V)
        self.nbr_idx = torch.from_numpy(data.nbr_idx).to(device)
        self.nbr_mask = torch.from_numpy(data.nbr_mask).to(device)
        self.offsets = torch.arange(L - 1, -1, -1, device=device)

    def _inputs(self, s, d, c):
        import torch

        seq = nbr = None
        if self.name in TEMPORAL and self.name != "spatial":
            idx = (d[:, None] - self.offsets[None, :]).clamp_min(0)
            seq = self.cube[idx, c[:, None]]
        if self.name == "spatial":
            nb = self.nbr_idx[c]
            nbr = (self.cube[d[:, None], nb], self.nbr_mask[c])
            seq = torch.empty(0, device=s.device)
        return seq, nbr

    def _logit(self, s, d, c):
        out = self.net(s, *self._inputs(s, d, c))
        return out

    def fit(self):
        import torch
        from sklearn.metrics import average_precision_score

        D, dev = self.data, self.device
        bs, lr, epochs, patience = DEFAULTS[self.name]
        S = torch.from_numpy(D.X).to(dev)
        Y = torch.from_numpy(D.y).to(dev)
        Ud = torch.from_numpy(D.U).to(dev)
        Cc = torch.from_numpy(D.C).to(dev)
        tr = torch.from_numpy(np.nonzero(D.tr)[0]).to(dev)
        va = torch.from_numpy(np.nonzero(D.va)[0]).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        steps = epochs * math.ceil(len(tr) / bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        lossf = torch.nn.BCEWithLogitsLoss()
        best, best_state, bad, t0 = -1.0, None, 0, time.time()
        for ep in range(epochs):
            self.net.train()
            perm = tr[torch.randperm(len(tr), device=dev)]
            tot = 0.0
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self._logit(S[b], Ud[b], Cc[b])
                out = out.float()
                yb = Y[b] if out.dim() == 1 else Y[b][:, None].expand_as(out)
                loss = lossf(out, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += float(loss) * len(b)
            p = self._predict_idx(S, Ud, Cc, va)
            ap = float(average_precision_score(D.y[D.va], p))
            self.log(f"  [{self.name}] epoch {ep + 1}: train loss {tot / len(tr):.4f}, validation PR-AUC {ap:.4f} "
                     f"[{time.time() - t0:.0f}s]")
            if ap > best + 1e-4:
                best, best_state, bad = ap, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)
        self.val_ap, self.seconds = best, round(time.time() - t0, 1)
        return self

    def _predict_idx(self, S, Ud, Cc, idx, bs: int = 16384) -> np.ndarray:
        import torch

        self.net.eval()
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                o = self._logit(S[b], Ud[b], Cc[b]).float()
                p = torch.sigmoid(o)
                out.append((p.mean(dim=1) if p.dim() == 2 else p).cpu().numpy())
        return np.concatenate(out)

    def score_days(self, d0: int, d1: int, static: np.ndarray | None = None) -> np.ndarray:
        import torch

        C = self.data.ctx.store.n_cells
        static = self.data.grid_static(d0, d1) if static is None else static
        S = torch.from_numpy(static).to(self.device)
        Ud = torch.arange(d0, d1, device=self.device).repeat_interleave(C)
        Cc = torch.arange(C, device=self.device).repeat(d1 - d0)
        p = self._predict_idx(S, Ud, Cc, torch.arange(len(S), device=self.device))
        return p.reshape(d1 - d0, C)


def fit_catboost(data: DailyData, device: str = "cuda", log=print):
    from catboost import CatBoostClassifier

    X, y = data.X_raw, data.y
    kw = dict(iterations=6000, learning_rate=0.06, depth=8, loss_function="Logloss", od_type="Iter", od_wait=150,
              random_seed=42, verbose=500)
    t0 = time.time()
    try:
        m = CatBoostClassifier(task_type="GPU" if device == "cuda" else "CPU", **kw)
        m.fit(X[data.tr], y[data.tr], eval_set=(X[data.va], y[data.va]))
    except Exception as exc:  # noqa: BLE001
        log(f"  [catboost] GPU failed ({exc}); CPU")
        m = CatBoostClassifier(task_type="CPU", thread_count=16, **kw)
        m.fit(X[data.tr], y[data.tr], eval_set=(X[data.va], y[data.va]))
    log(f"  [catboost] best iteration {m.get_best_iteration()} [{time.time() - t0:.0f}s]")
    return m


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


# -----------------------------------------------------------------------------
# Servable member of an ensemble
# -----------------------------------------------------------------------------
class NeuralMember:
    """A trained network with everything needed to score any (day, cell) of any store:
    the static transform, the sequence scaling and the weights (CPU tensors)."""

    def __init__(self, arch: str, state_dict: dict, cols: list[str], qt, seq_mu, seq_sd, lookback: int,
                 lag_days: int, meta: dict | None = None):
        self.arch, self.state_dict, self.cols, self.qt = arch, state_dict, cols, qt
        self.seq_mu, self.seq_sd = np.asarray(seq_mu, np.float32), np.asarray(seq_sd, np.float32)
        self.lookback, self.lag_days, self.meta = int(lookback), int(lag_days), meta or {}
        self._net = None
        self._cube_key = self._cube = None

    @classmethod
    def from_model(cls, m: "NeuralModel") -> "NeuralMember":
        d = m.data
        sd = {k: v.detach().cpu() for k, v in m.net.state_dict().items()}
        return cls(m.name, sd, d.cols, d.qt, d.seq_mu, d.seq_sd, d.lookback, d.recipe.recency_lag_days,
                   {"validation_sample_pr_auc": m.val_ap, "seconds": m.seconds})

    def to_dict(self) -> dict:
        return {"arch": self.arch, "state_dict": self.state_dict, "cols": self.cols, "qt": self.qt,
                "seq_mu": self.seq_mu, "seq_sd": self.seq_sd, "lookback": self.lookback,
                "lag_days": self.lag_days, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "NeuralMember":
        return cls(**d)

    def _device(self) -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def net(self):
        if self._net is None:
            make = _nets()[self.arch]
            n_static, n_seq = len(self.cols), len(self.seq_mu)
            net = make(n_static, n_seq, self.lookback) if self.arch in TEMPORAL else make(n_static)
            net.load_state_dict(self.state_dict)
            self._net = net.to(self._device()).eval()
        return self._net

    def predict(self, X: pd.DataFrame, di: np.ndarray, ci: np.ndarray, store, events: pd.DataFrame,
                batch: int = 16384) -> np.ndarray:
        """Sample-space probability for rows ``(di, ci)`` of ``store``; ``X`` their raw features."""
        import torch

        dev = self._device()
        key = (id(store), store.n_days, len(events))
        if self._cube_key != key:
            cube = (daily_cube(store, events, self.lag_days) - self.seq_mu) / self.seq_sd
            self._cube, self._cube_key = torch.from_numpy(cube.astype(np.float32)).to(dev), key
        S = np.clip(self.qt.transform(np.nan_to_num(X[self.cols].to_numpy(np.float32))), -5, 5).astype(np.float32)
        net = self.net()
        offs = torch.arange(self.lookback - 1, -1, -1, device=dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(S), batch):
                s = torch.from_numpy(S[i:i + batch]).to(dev)
                d = torch.from_numpy(np.asarray(di[i:i + batch], np.int64)).to(dev)
                c = torch.from_numpy(np.asarray(ci[i:i + batch], np.int64)).to(dev)
                seq = self._cube[(d[:, None] - offs[None, :]).clamp_min(0), c[:, None]] if self.arch in TEMPORAL else None
                o = net(s, seq, None).float()          # full precision: same scores on GPU and CPU
                p = torch.sigmoid(o)
                out.append((p.mean(dim=1) if p.dim() == 2 else p).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def train_member(arch: str, ctx: GridContext, recipe: Recipe, sample, *, test_from, train_start, val_years: int = 1,
                 lookback: int = 90, seed: int = 0, log=print) -> NeuralMember:
    """Train one network on the same sample as the tree model (early stopping on the
    validation year) and return it as a servable member."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = DailyData(ctx, test_from=test_from, train_start=train_start, val_years=val_years, lookback=lookback,
                     recipe=recipe, sample=sample, log=log)
    m = NeuralModel(arch, data, device=dev, seed=seed, log=log).fit()
    member = NeuralMember.from_model(m)
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return member
