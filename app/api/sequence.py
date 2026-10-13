"""Sequence (RNN) fire / no-fire models — GRU and BiGRU+attention.

The XGBoost model in ``predictor.py`` scores a single instant (one feature
dict). The RNN models here instead consume a **72-hour (3-day) weather
sequence** per sample — the ``<var>_t-72 … <var>_t-1`` columns produced by
stage 2 — together with the static terrain features. This module ports the
exact architectures + training recipe from ``notebooks/gru_*.ipynb`` and adds
the missing piece for inference: **persistence** (the notebooks train in
memory and never save a reusable model).

Public surface
--------------
    from app.api.sequence import SequencePredictor

    sp = SequencePredictor("portugal", model_type="gru")   # loads saved model if present
    sp.train(epochs=60)                                     # fits + saves outputs/models/fire_gru_portugal.pt
    sp.predict_proba(
        sequence=seq,              # DataFrame[lookback rows][seq_vars] / ndarray (T,V) / {var: [..T..]}
        static={"slope": 18, "roughness": 0.4, "aspect_sin": 0.0, "aspect_cos": -1.0},
    )                              # -> probability of ignition
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from firepredict import config

from . import _paths

# Static terrain features the RNN concatenates after the sequence encoder
# (same set the GRU notebooks use).
STATIC_FEATURES: list[str] = ["slope", "roughness", "aspect_sin", "aspect_cos"]
DEFAULT_LOOKBACK: int = config.WEATHER_LOOKBACK_HOURS  # 72

_SEQ_COL_RE = re.compile(r"^(.+)_t-(\d+)$")


# ---------------------------------------------------------------------------
# Models (ported verbatim from notebooks/gru_era5_land_2014_2023.ipynb)
# ---------------------------------------------------------------------------
def _import_torch():
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401

    return torch, nn


def _build_model(model_type: str, seq_input_dim: int, static_dim: int,
                 hidden_dim: int | None = None, n_layers: int = 2, dropout: float = 0.3):
    torch, nn = _import_torch()

    class FireGRU(nn.Module):
        """GRU over the weather sequence + static terrain → fire logit.

        sequence (T, V) → GRU(n_layers, hidden) → last hidden (hidden)
        static (S) ────────────────────────────→ concat → (hidden+S)
        → Linear(32) → ReLU → Dropout → Linear(1)
        """

        def __init__(self, seq_input_dim, static_dim, hidden_dim=64, n_layers=2, dropout=0.3):
            super().__init__()
            self.gru = nn.GRU(
                input_size=seq_input_dim, hidden_size=hidden_dim, num_layers=n_layers,
                batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim + static_dim, 32), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(32, 1),
            )

        def forward(self, seq, static):
            _, h_n = self.gru(seq)
            combined = torch.cat([h_n[-1], static], dim=1)
            return self.head(combined).squeeze(-1)

    class FireBiGRUAttn(nn.Module):
        """Bidirectional GRU + temporal-attention pooling + static terrain → logit."""

        def __init__(self, seq_input_dim, static_dim, hidden_dim=128, n_layers=2, dropout=0.3):
            super().__init__()
            self.gru = nn.GRU(
                input_size=seq_input_dim, hidden_size=hidden_dim, num_layers=n_layers,
                batch_first=True, bidirectional=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            out_dim = hidden_dim * 2
            self.attn_proj = nn.Linear(out_dim, out_dim)
            self.attn_v = nn.Linear(out_dim, 1, bias=False)
            self.norm = nn.LayerNorm(out_dim + static_dim)
            self.head = nn.Sequential(
                nn.Linear(out_dim + static_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
            )

        def forward(self, seq, static):
            out, _ = self.gru(seq)
            scores = self.attn_v(torch.tanh(self.attn_proj(out))).squeeze(-1)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (out * weights).sum(dim=1)
            combined = torch.cat([pooled, static], dim=1)
            return self.head(self.norm(combined)).squeeze(-1)

    if model_type == "gru":
        return FireGRU(seq_input_dim, static_dim, hidden_dim or 64, n_layers, dropout)
    if model_type == "bigru":
        return FireBiGRUAttn(seq_input_dim, static_dim, hidden_dim or 128, n_layers, dropout)
    raise ValueError(f"Unknown model_type {model_type!r}; expected 'gru' or 'bigru'.")


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def detect_seq_vars(columns, lookback: int = DEFAULT_LOOKBACK) -> list[str]:
    """Return the weather-variable stems that have a complete ``t-lookback…t-1``
    set of columns, in first-seen order (e.g. ``['temp','hum','wind','precip']``)."""
    have: dict[str, set[int]] = {}
    for c in columns:
        m = _SEQ_COL_RE.match(c)
        if m:
            have.setdefault(m.group(1), set()).add(int(m.group(2)))
    needed = set(range(1, lookback + 1))
    order, seen = [], set()
    for c in columns:
        m = _SEQ_COL_RE.match(c)
        if m and m.group(1) not in seen and needed.issubset(have.get(m.group(1), ())):
            order.append(m.group(1))
            seen.add(m.group(1))
    return order


def _seq_groups(seq_vars, lookback):
    # column order t-lookback, …, t-1 (oldest → newest), matching the notebooks.
    return [[f"{v}_t-{h}" for h in range(lookback, 0, -1)] for v in seq_vars]


def _load_sequence_frame(region, ts_path, terr_path, lookback, seq_vars):
    ts_path, terr_path = Path(ts_path), Path(terr_path)
    for label, p in (("timeseries", ts_path), ("terrain", terr_path)):
        if not p.exists():
            raise FileNotFoundError(
                f"{label} dataset for region {region!r} not found at:\n  {p}\n"
                f"Build it first (stage 2 writes the timeseries CSV, stage 3 the terrain CSV), "
                f"e.g. app.api.pipeline.preprocess({region!r})."
            )
    ts = pd.read_csv(ts_path, low_memory=False)
    ts["DH_Inicio"] = pd.to_datetime(ts["DH_Inicio"], format="ISO8601", errors="coerce")
    terr = pd.read_csv(terr_path, low_memory=False)
    merge_cols = ["sample_id"] + [c for c in STATIC_FEATURES if c in terr.columns]
    merged = ts.merge(terr[merge_cols], on="sample_id", how="inner")

    if seq_vars is None:
        seq_vars = detect_seq_vars(merged.columns, lookback)
    seq_vars = [v for v in seq_vars if all(f"{v}_t-{h}" in merged.columns
                                           for h in range(lookback, 0, -1))]
    if not seq_vars:
        raise ValueError(
            f"No complete {lookback}h sequence variables found in {ts_path.name}."
        )
    missing_static = [c for c in STATIC_FEATURES if c not in merged.columns]
    if missing_static:
        raise ValueError(f"terrain CSV missing static features {missing_static}.")

    groups = _seq_groups(seq_vars, lookback)
    need = [c for g in groups for c in g] + STATIC_FEATURES + ["label"]
    clean = merged.dropna(subset=need).sort_values("DH_Inicio").reset_index(drop=True)
    return clean, seq_vars, groups


# ---------------------------------------------------------------------------
# Training + persistence
# ---------------------------------------------------------------------------
def _sequence_model_path(region, model_type) -> Path:
    return _paths.MODELS_DIR / f"fire_{model_type}_{region}.pt"


def _build_suffix(ts_path: Path) -> str:
    """Recover the build suffix from a timeseries filename (``""`` = canonical)."""
    for region in _paths.region_keys():
        base = _paths.paths_for(region)["weather_timeseries"].stem
        if ts_path.stem.startswith(base):
            return ts_path.stem[len(base) + 1:]
    return ""


def train_sequence_model(
    region: str,
    *,
    model_type: str = "gru",
    build: str | None = None,
    timeseries_csv: str | Path | None = None,
    terrain_csv: str | Path | None = None,
    lookback: int | None = None,
    seq_vars: list[str] | None = None,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 15,
    val_fraction: float = 0.1,
    select_on: str = "pr_auc",
    hidden_dim: int | None = None,
    n_layers: int = 2,
    dropout: float = 0.3,
    device: str | None = None,
    save: bool = True,
    model_out: str | Path | None = None,
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train (and persist) a GRU/BiGRU sequence model for ``region``.

    Reads the region's lookback timeseries CSV + terrain CSV, builds
    ``(T, n_vars)`` sequences + static terrain, normalises per-variable on the
    chronological train fold, trains with ``pos_weight``-balanced
    ``BCEWithLogitsLoss`` and cosine warm-restart LR, and saves a ``.pt``
    artifact holding the weights + normalisation stats + the variable/lookback
    schema.

    Three chronological folds: the last 20 % is the **test** fold, reported and
    never looked at during training; the last ``val_fraction`` of the rest is the
    **validation** fold, on which the best epoch is picked (``select_on``:
    ``"pr_auc"`` — threshold-free — or ``"f1"`` at 0.5) and early stopping runs.
    Until 2026-09 the epoch was picked on the test fold itself, which made the
    reported test metrics optimistic (docs/improvement-plan.md, Phase 1.4).

    Returns ``(artifact, metrics)``.
    """
    import copy

    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    torch, nn = _import_torch()
    from torch.utils.data import DataLoader, TensorDataset

    from . import _datasets

    lookback = lookback or DEFAULT_LOOKBACK
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Resolve which dataset build to train on. Explicit CSVs win; then an
    # explicit `build` suffix; otherwise auto-pick the richest build available
    # (so the model uses ALL the ERA5 variables that were collected, not just
    # the few in the legacy bulk file).
    if timeseries_csv and terrain_csv:
        ts_path, terr_path = Path(timeseries_csv), Path(terrain_csv)
    elif build is not None:
        ts_path, terr_path = _datasets.resolve_timeseries(region, build)
    else:
        best = _datasets.best_timeseries_build(region, lookback)
        if best is not None:
            ts_path, terr_path = best["timeseries"], best["terrain"]
            if verbose:
                print(f"[{region}/{model_type}] auto-selected build "
                      f"'{best['build']}' ({best['n_seq_vars']} weather vars). "
                      f"Pass build=... or timeseries_csv=... to override.")
        else:
            p = _paths.paths_for(region)
            ts_path, terr_path = p["weather_timeseries"], p["final_dataset"]

    clean, seq_vars, groups = _load_sequence_frame(region, ts_path, terr_path, lookback, seq_vars)
    if verbose:
        print(f"[{region}/{model_type}] {len(clean):,} samples with complete "
              f"{lookback}h sequence | {len(seq_vars)} vars={seq_vars} | device={dev.type}")

    split = int(len(clean) * 0.8)
    fit_df, test_df = clean.iloc[:split], clean.iloc[split:]
    vsplit = int(len(fit_df) * (1 - val_fraction))
    train_df, val_df = fit_df.iloc[:vsplit], fit_df.iloc[vsplit:]
    if select_on not in ("pr_auc", "f1"):
        raise ValueError(f"select_on must be 'pr_auc' or 'f1', got {select_on!r}")

    # Per-variable normalisation on TRAIN statistics.
    seq_means, seq_stds = {}, {}
    for v, g in zip(seq_vars, groups):
        vals = train_df[g].to_numpy().flatten()
        seq_means[v] = float(np.nanmean(vals))
        seq_stds[v] = float(np.nanstd(vals) + 1e-8)
    static_means = {c: float(train_df[c].mean()) for c in STATIC_FEATURES}
    static_stds = {c: float(train_df[c].std() + 1e-8) for c in STATIC_FEATURES}

    def _tensors(df):
        seq = np.stack(
            [((df[g].to_numpy() - seq_means[v]) / seq_stds[v]) for v, g in zip(seq_vars, groups)],
            axis=-1,
        ).astype("float32")                                   # (N, T, V)
        stat = np.stack(
            [((df[c].to_numpy() - static_means[c]) / static_stds[c]) for c in STATIC_FEATURES],
            axis=-1,
        ).astype("float32")                                   # (N, S)
        y = df["label"].to_numpy().astype("float32")
        return (torch.tensor(seq), torch.tensor(stat), torch.tensor(y))

    Xtr_seq, Xtr_stat, ytr = _tensors(train_df)
    Xva_seq, Xva_stat, yva = _tensors(val_df)
    Xte_seq, Xte_stat, yte = _tensors(test_df)
    train_loader = DataLoader(TensorDataset(Xtr_seq, Xtr_stat, ytr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva_seq, Xva_stat, yva), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(Xte_seq, Xte_stat, yte), batch_size=batch_size, shuffle=False)

    model = _build_model(model_type, len(seq_vars), len(STATIC_FEATURES),
                         hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout).to(dev)

    n_pos = int(ytr.sum())
    n_neg = len(ytr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt_cls = torch.optim.AdamW if model_type == "bigru" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=lr, **({"weight_decay": 1e-4} if model_type == "bigru" else {}))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(min(20, epochs), 1), T_mult=2, eta_min=lr / 1e3
    )

    from sklearn.metrics import average_precision_score

    best_score, best_state, best_epoch, no_improve = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for sb, tb, yb in train_loader:
            sb, tb, yb = sb.to(dev), tb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(sb, tb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        probs, labs = [], []
        with torch.no_grad():
            for sb, tb, yb in val_loader:          # the validation fold — never the test fold
                logits = model(sb.to(dev), tb.to(dev))
                probs.extend(torch.sigmoid(logits).cpu().numpy())
                labs.extend(yb.numpy())
        probs, labs = np.array(probs), np.array(labs)
        f1 = f1_score(labs, (probs >= 0.5).astype(int), zero_division=0)
        ap = average_precision_score(labs, probs) if labs.any() else 0.0
        score = ap if select_on == "pr_auc" else f1
        if verbose and (epoch == 1 or epoch % 5 == 0 or score > best_score):
            try:
                auc = roc_auc_score(labs, probs)
            except ValueError:
                auc = float("nan")
            print(f"  epoch {epoch:>3d}/{epochs}  val_PR-AUC={ap:.4f}  val_F1={f1:.4f}  val_AUC={auc:.4f}"
                  + ("  *" if score > best_score + 1e-6 else ""))
        if score > best_score + 1e-6:
            best_score, best_state, best_epoch, no_improve = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            no_improve += 1
        if no_improve >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch}, val {select_on}={best_score:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test metrics.
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for sb, tb, yb in test_loader:
            probs.extend(torch.sigmoid(model(sb.to(dev), tb.to(dev))).cpu().numpy())
            labs.extend(yb.numpy())
    probs, labs = np.array(probs), np.array(labs)
    preds = (probs >= 0.5).astype(int)
    try:
        roc = float(roc_auc_score(labs, probs))
    except ValueError:
        roc = None
    metrics = {
        "region": region, "model_type": model_type, "seq_vars": seq_vars,
        "lookback": lookback, "static_features": STATIC_FEATURES,
        # Which build the sequences came from. The ERA5-Land accumulated columns
        # only mean "per hour" on a `_hourly` build (scripts/deaccumulate_era5.py);
        # app.api.openmeteo.check_model reads this to catch the mismatch.
        "build": _build_suffix(ts_path), "dataset": ts_path.name,
        "n_rows": int(len(clean)), "n_train": int(len(train_df)), "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "train_through": str(train_df["DH_Inicio"].max()), "val_from": str(val_df["DH_Inicio"].min()),
        "test_from": str(test_df["DH_Inicio"].min()),
        "positives_train": int(ytr.sum()), "positives_val": int(yva.sum()), "positives_test": int(yte.sum()),
        "best_epoch": int(best_epoch), "selected_on": f"validation {select_on}",
        "best_val_score": float(best_score),
        "f1": float(f1_score(labs, preds, zero_division=0)), "roc_auc": roc,
        "report": classification_report(labs, preds, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labs, preds).tolist(),
    }

    artifact = {
        "framework": "torch", "model_type": model_type,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "seq_vars": seq_vars, "lookback": lookback, "static_features": STATIC_FEATURES,
        "hidden_dim": hidden_dim, "n_layers": n_layers, "dropout": dropout,
        "seq_means": seq_means, "seq_stds": seq_stds,
        "static_means": static_means, "static_stds": static_stds,
        "region": region, "trained_through": metrics["train_through"],
        "metrics": {k: v for k, v in metrics.items() if k != "report"},
    }
    if save:
        out = Path(model_out) if model_out else _sequence_model_path(region, model_type)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, out)
        metrics["model_path"] = str(out)
        if verbose:
            print(f"  saved → {out}")
    return artifact, metrics


def load_sequence_artifact(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load a saved sequence-model artifact and rebuild the (eval-mode) model."""
    torch, _ = _import_torch()
    art = torch.load(path, map_location=device, weights_only=False)
    model = _build_model(
        art["model_type"], len(art["seq_vars"]), len(art["static_features"]),
        hidden_dim=art.get("hidden_dim"), n_layers=art.get("n_layers", 2),
        dropout=art.get("dropout", 0.3),
    )
    model.load_state_dict(art["state_dict"])
    model.to(device).eval()
    art["_model"] = model
    return art


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
class SequencePredictor:
    """Load (or train) a GRU/BiGRU model and predict on an hourly weather window.

    Parameters
    ----------
    region:        country/region key.
    model_type:    ``"gru"`` (default) or ``"bigru"``.
    model_path:    override the artifact path (default ``outputs/models/fire_<type>_<region>.pt``).
    device:        ``"cpu"`` (default) / ``"cuda"``.
    autoload:      load a saved model immediately if present.

    Inference input
    ---------------
    ``predict_proba(sequence, static)`` where, for ONE sample:
      - ``sequence`` is a DataFrame of ``lookback`` rows whose columns include
        ``seq_vars`` (oldest→newest), OR an ``ndarray`` of shape
        ``(lookback, len(seq_vars))`` in ``seq_vars`` order, OR a dict
        ``{var: [..lookback values..]}``.
      - ``static`` is a dict ``{feature: value}`` over ``static_features`` (or an
        array in that order).
    For a BATCH, pass a list of sequences and a matching list of statics (or a
    ``(N, lookback, V)`` array + ``(N, S)`` array). A single static dict is
    broadcast across a batch of sequences.
    """

    def __init__(self, region: str = "portugal", *, model_type: str = "gru",
                 model_path: str | Path | None = None, device: str = "cpu",
                 autoload: bool = True) -> None:
        if region not in _paths.region_keys():
            raise KeyError(f"Unknown region {region!r}. Available: {_paths.region_keys()}")
        if model_type not in ("gru", "bigru"):
            raise ValueError(f"model_type must be 'gru' or 'bigru', got {model_type!r}")
        self.region = region
        self.model_type = model_type
        self.device = device
        self._model_path = Path(model_path) if model_path else _sequence_model_path(region, model_type)
        self.model = None
        self.seq_vars: list[str] = []
        self.lookback: int = DEFAULT_LOOKBACK
        self.static_features: list[str] = list(STATIC_FEATURES)
        self.metadata: dict[str, Any] = {}
        self._seq_means = self._seq_stds = self._static_means = self._static_stds = None
        if autoload and self._model_path.exists():
            self.load()

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def load(self, path: str | Path | None = None) -> "SequencePredictor":
        art = load_sequence_artifact(path or self._model_path, device=self.device)
        self.model = art["_model"]
        self.seq_vars = list(art["seq_vars"])
        self.lookback = int(art["lookback"])
        self.static_features = list(art["static_features"])
        self._seq_means, self._seq_stds = art["seq_means"], art["seq_stds"]
        self._static_means, self._static_stds = art["static_means"], art["static_stds"]
        self.metadata = dict(art.get("metrics", {}) or {})
        self.metadata.setdefault("trained_through", art.get("trained_through"))
        if path:
            self._model_path = Path(path)
        return self

    def train(self, **kw) -> dict[str, Any]:
        """Train + persist this region's sequence model; returns metrics."""
        kw.setdefault("model_out", self._model_path)
        _, metrics = train_sequence_model(self.region, model_type=self.model_type, **kw)
        self.load()
        return metrics

    # -- input coercion -----------------------------------------------------
    def _one_seq(self, s) -> np.ndarray:
        T, V = self.lookback, len(self.seq_vars)
        if isinstance(s, pd.DataFrame):
            missing = [v for v in self.seq_vars if v not in s.columns]
            if missing:
                raise ValueError(f"sequence DataFrame missing column(s) {missing}; "
                                 f"need {self.seq_vars}")
            if len(s) != T:
                raise ValueError(f"sequence has {len(s)} rows, expected lookback={T}")
            return s[self.seq_vars].to_numpy(dtype="float32")
        if isinstance(s, Mapping):
            cols = []
            for v in self.seq_vars:
                if v not in s:
                    raise ValueError(f"sequence dict missing variable {v!r}; need {self.seq_vars}")
                col = np.asarray(s[v], dtype="float32")
                if col.shape != (T,):
                    raise ValueError(f"sequence[{v!r}] has length {col.shape}, expected ({T},)")
                cols.append(col)
            return np.stack(cols, axis=-1)
        arr = np.asarray(s, dtype="float32")
        if arr.shape != (T, V):
            raise ValueError(f"sequence array shape {arr.shape}, expected ({T}, {V}) "
                             f"in seq_vars order {self.seq_vars}")
        return arr

    def _to_seq(self, sequence):
        T, V = self.lookback, len(self.seq_vars)
        if isinstance(sequence, np.ndarray) and sequence.ndim == 3:
            if sequence.shape[1:] != (T, V):
                raise ValueError(f"batch array shape {sequence.shape}, expected (N, {T}, {V})")
            return sequence.astype("float32"), False
        if isinstance(sequence, (list, tuple)) and sequence and isinstance(
            sequence[0], (pd.DataFrame, dict, np.ndarray, list)
        ):
            return np.stack([self._one_seq(s) for s in sequence]), False
        return self._one_seq(sequence)[None, ...], True

    def _to_static(self, static, n: int) -> np.ndarray:
        S = len(self.static_features)

        def row(d):
            missing = [f for f in self.static_features if f not in d]
            if missing:
                raise ValueError(f"static missing feature(s) {missing}; need {self.static_features}")
            return [float(d[f]) for f in self.static_features]

        if isinstance(static, Mapping):
            arr = np.array([row(static)], dtype="float32")
        elif isinstance(static, (list, tuple)) and static and isinstance(static[0], Mapping):
            arr = np.array([row(d) for d in static], dtype="float32")
        else:
            arr = np.asarray(static, dtype="float32")
            if arr.ndim == 1:
                arr = arr[None, :]
        if arr.shape[1] != S:
            raise ValueError(f"static has {arr.shape[1]} features, expected {S} ({self.static_features})")
        if arr.shape[0] == 1 and n > 1:
            arr = np.repeat(arr, n, axis=0)
        if arr.shape[0] != n:
            raise ValueError(f"got {arr.shape[0]} static rows for {n} sequence(s)")
        return arr

    def predict_proba(self, sequence, static):
        """Probability of ignition. ``float`` for one sample, ``ndarray`` for a batch."""
        if not self.is_ready:
            raise RuntimeError(
                f"No {self.model_type} model loaded for region {self.region!r}. "
                f"Train one with .train(), or pass model_path=<saved .pt>."
            )
        torch, _ = _import_torch()
        seq, single = self._to_seq(sequence)            # (N, T, V)
        stat = self._to_static(static, seq.shape[0])    # (N, S)

        seq = seq.astype("float32").copy()
        for i, v in enumerate(self.seq_vars):
            seq[:, :, i] = (seq[:, :, i] - self._seq_means[v]) / self._seq_stds[v]
        stat = stat.astype("float32").copy()
        for j, f in enumerate(self.static_features):
            stat[:, j] = (stat[:, j] - self._static_means[f]) / self._static_stds[f]

        with torch.no_grad():
            logits = self.model(
                torch.tensor(seq, device=self.device),
                torch.tensor(stat, device=self.device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
        return float(probs[0]) if single else probs

    def predict(self, sequence, static, *, threshold: float = 0.5):
        """Binary fire(1)/no-fire(0) at ``threshold``."""
        proba = self.predict_proba(sequence, static)
        if np.isscalar(proba):
            return int(proba >= threshold)
        return (np.asarray(proba) >= threshold).astype(int)

    def __repr__(self) -> str:
        state = "ready" if self.is_ready else "no-model"
        return (f"SequencePredictor(region={self.region!r}, model_type={self.model_type!r}, "
                f"{state}, lookback={self.lookback}, seq_vars={self.seq_vars})")
