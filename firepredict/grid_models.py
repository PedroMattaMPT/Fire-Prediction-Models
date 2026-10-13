"""Models scored on the weather-store grid (docs/improvement-plan.md, Phases 1.5, 2 and 4).

Three kinds, one interface (``unit``, ``score_hours``, ``score_days``,
``score_rows``):

- **baselines** — ``FWIBaseline`` (the day's FWI), ``FWIHistoryBaseline``
  (``log1p(FWI) + log1p(fires per year in the cell)``), ``ClimatologyLegacy``
  (XGBoost on hour, day of year, lat/lon and cell fire history, trained on the
  shipped models' own training fold — the audit's reference baseline); the
  case-base climatology is recipe ``R0``;
- **shipped models** — ``LegacyPointModel`` (xgb7 / xgb18 / xgb16 artifacts, terrain
  at the cell centre) and ``LegacySequenceModel`` (GRU / BiGRU, weighted sample only);
- **grid models** — ``GridModel``, trained by :func:`train` from a :class:`Recipe`:
  all fires, case-base negatives, explicit time and place, leave-one-year-out
  fire-history priors, early stopping on a validation year, prior-shift +
  isotonic calibration.

A :class:`GridContext` bundles what scoring needs: the store, the fire events and
a feature builder. ``GridContext.reference()`` reproduces the Sept 2026 audit's
setup (registry clock read as UTC, no de-duplication).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config, registry
from .features import GROUPS, MONOTONE, FeatureBuilder, FirePriors, FireRecency, SoilClimatology, columns, features_without
from .history_features import PRIOR_EXTRA_GROUPS, FireNeighbourhood
from .sampling import fire_cellday_keys, generate_case_base_negatives, halo_forbidden_celldays
from .weather_store import WeatherStore

GRID_BLOCK_HOURS = 2000          # ~2 M cell-hours per scoring chunk
ARTIFACT_VERSION = 1


# -----------------------------------------------------------------------------
# Context
# -----------------------------------------------------------------------------
class GridContext:
    """Store + fire events + recency index, shared by the models scored on them."""

    def __init__(self, store: WeatherStore | None = None, events: pd.DataFrame | None = None, *,
                 local_tz: str | None = None, protocol: str = "current"):
        self.store = store or WeatherStore()
        self.protocol = protocol
        self.local_tz = local_tz
        if events is None:
            events = registry.read_fire_events(self.store)
        self.events = events
        self._recency: dict[int, FireRecency] = {}
        self._neighbourhood: dict[int, FireNeighbourhood] = {}

    @classmethod
    def reference(cls, store: WeatherStore | None = None) -> "GridContext":
        """The audit's setup: registry clock taken as UTC, no id de-duplication,
        calendar features on that same clock."""
        store = store or WeatherStore()
        ev = registry.load_fire_events(store=store, to_utc=False, dedup=False)
        return cls(store, ev, local_tz="UTC", protocol="reference")

    def recency(self, lag_days: int = config.RECENCY_LAG_DAYS) -> FireRecency:
        if lag_days not in self._recency:
            self._recency[lag_days] = FireRecency(self.events, lag_days)
        return self._recency[lag_days]

    def neighbourhood(self, lag_days: int = config.RECENCY_LAG_DAYS) -> FireNeighbourhood:
        if lag_days not in self._neighbourhood:          # lazy: built on first use
            self._neighbourhood[lag_days] = FireNeighbourhood(self.events, self.store.cells, lag_days)
        return self._neighbourhood[lag_days]

    def builder(self, *, lag_days: int = config.RECENCY_LAG_DAYS, soil_clim=None) -> FeatureBuilder:
        return FeatureBuilder(self.store, local_tz=self.local_tz, recency=self.recency(lag_days), soil_clim=soil_clim,
                              neighbourhood=self.neighbourhood(lag_days))

    def priors(self, recipe: "Recipe", start, end) -> FirePriors:
        """Fire-history priors of ``[start, end)``; with the nbr / regime extras only if the recipe uses them."""
        extra = any(g in PRIOR_EXTRA_GROUPS for g in recipe.groups)
        return FirePriors(self.events, self.store.n_cells, start, end, cells=self.store.cells if extra else None)

    def cell_pool(self, cells: str = "all") -> np.ndarray:
        if cells == "land":
            return np.nonzero(~self.store.cells["remapped"].to_numpy(bool))[0]
        return np.arange(self.store.n_cells)


# -----------------------------------------------------------------------------
# Shared scoring helpers
# -----------------------------------------------------------------------------
class _Hourly:
    unit = "hour"
    expensive = False

    def score_days(self, d0: int, d1: int) -> np.ndarray:
        """Daily score = the day's maximum hourly score."""
        h = self.score_hours(d0 * 24, d1 * 24)
        return h.reshape(d1 - d0, 24, -1).max(axis=1)


def score_hours_chunked(fn, h0: int, h1: int, n_cells: int, block: int = GRID_BLOCK_HOURS) -> np.ndarray:
    out = np.empty((h1 - h0, n_cells), dtype=np.float32)
    for a in range(h0, h1, block):
        b = min(a + block, h1)
        out[a - h0:b - h0] = np.asarray(fn(a, b), dtype=np.float32).reshape(b - a, n_cells)
    return out


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------
class FWIBaseline(_Hourly):
    """The day's Fire Weather Index (ERA5-Land), broadcast to every hour of the day."""

    name = "fwi"

    def __init__(self, ctx: GridContext):
        self.ctx = ctx
        self.fwi = np.asarray(ctx.store.daily("fwi"))

    def score_hours(self, h0, h1):
        return self.fwi[self.ctx.store.day_of_hour[h0:h1]]

    def score_rows(self, hi, ci):
        return self.fwi[self.ctx.store.day_of_hour[hi], ci]


class FWIHistoryBaseline(FWIBaseline):
    """``log1p(FWI) + log1p(fires per year in the cell)`` — fire history from the
    store's first day up to ``test_from``."""

    name = "fwi_history"

    def __init__(self, ctx: GridContext, test_from, start=None):
        super().__init__(ctx)
        st = ctx.store
        start = pd.Timestamp(start) if start is not None else st.hours[0]
        test_from = pd.Timestamp(test_from)
        ev = ctx.events
        f = ev[(ev["cell"] >= 0) & (ev["t"] >= start) & (ev["t"] < test_from)]
        years = max((test_from - f["t"].min()).days / 365.25, 1e-9) if len(f) else 1.0
        cnt = np.bincount(f["cell"].to_numpy(), minlength=st.n_cells)
        self.prior = np.log1p(cnt / years).astype(np.float32)

    def score_hours(self, h0, h1):
        return np.log1p(super().score_hours(h0, h1)) + self.prior[None, :]

    def score_rows(self, hi, ci):
        return np.log1p(super().score_rows(hi, ci)) + self.prior[ci]


class ClimatologyLegacy(_Hourly):
    """The audit's climatology baseline: XGBoost on hour, day of year, lat/lon and
    the cell's fire history, trained on the shipped models' own training fold
    (the legacy 1:10 sample table, first 80 % chronologically)."""

    name = "climatology_legacy"
    FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "lat", "lon", "cell_prior"]

    def __init__(self, ctx: GridContext, test_from=config.GRID_REFERENCE_TEST_FROM,
                 build: str = "era5_land_2014_2023_hourly", test_fraction: float = 0.2):
        import xgboost as xgb

        self.ctx = ctx
        st = ctx.store
        final = config.processed_path(f"final_fire_weather_terrain_{config.TARGET_YEAR}_bulk_{build}.csv")
        need = list(config.OPERATIONAL_WEATHER_FEATURES) + list(config.TERRAIN_FEATURES)
        df = pd.read_csv(final, usecols=["DH_Inicio", "label", "lat", "lon", "snapped_lat", "snapped_lon"] + need,
                         low_memory=False)
        df["DH_Inicio"] = pd.to_datetime(df["DH_Inicio"], errors="coerce", format="ISO8601")
        df = df.dropna(subset=["DH_Inicio", "label"]).sort_values("DH_Inicio", kind="stable").reset_index(drop=True)
        df = df.dropna(subset=need).reset_index(drop=True)
        tr = df.iloc[: int(len(df) * (1 - test_fraction))].copy()
        ev = ctx.events[(ctx.events["cell"] >= 0) & (ctx.events["t"] >= st.hours[0])]   # history from the store's first year
        first = ev["t"].min()
        cut = tr["DH_Inicio"].max()
        hist = ev[ev["t"] <= cut].groupby(["slat", "slon"]).size().rename("h")
        yrs = (cut - first).days / 365.25
        key = tr[["snapped_lat", "snapped_lon"]].round(1).rename(columns={"snapped_lat": "slat", "snapped_lon": "slon"})
        tr["cell_prior"] = np.log1p(key.merge(hist, left_on=["slat", "slon"], right_index=True, how="left")["h"]
                                    .fillna(0).to_numpy() / yrs)
        t = tr["DH_Inicio"]
        tr["hour_sin"], tr["hour_cos"] = np.sin(2 * np.pi * t.dt.hour / 24), np.cos(2 * np.pi * t.dt.hour / 24)
        tr["doy_sin"], tr["doy_cos"] = np.sin(2 * np.pi * t.dt.dayofyear / 365.25), np.cos(2 * np.pi * t.dt.dayofyear / 365.25)
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective="binary:logistic",
                                       random_state=42, verbosity=0, n_jobs=8, tree_method="hist")
        self.model.fit(tr[self.FEATURES], tr["label"])
        tf = pd.Timestamp(test_from)
        f = ev[ev["t"] < tf]
        self.prior = np.log1p(np.bincount(f["cell"].to_numpy(), minlength=st.n_cells) / ((tf - first).days / 365.25))

    def _X(self, hi, ci):
        st = self.ctx.store
        t = st.hours[hi]
        h, doy = t.hour.to_numpy(), t.dayofyear.to_numpy()
        return pd.DataFrame({
            "hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "lat": st.cells["slat"].to_numpy()[ci], "lon": st.cells["slon"].to_numpy()[ci],
            "cell_prior": self.prior[ci],
        }).astype(np.float32)

    def score_rows(self, hi, ci):
        return self.model.predict_proba(self._X(np.asarray(hi), np.asarray(ci)))[:, 1]

    def score_hours(self, h0, h1):
        C = self.ctx.store.n_cells
        hi = np.repeat(np.arange(h0, h1), C)
        ci = np.tile(np.arange(C), h1 - h0)
        return self.score_rows(hi, ci).reshape(h1 - h0, C)


# -----------------------------------------------------------------------------
# Shipped models
# -----------------------------------------------------------------------------
def _cell_terrain(store) -> pd.DataFrame:
    p = store.path / "cell_terrain.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run stage 1d")
    return pd.read_parquet(p)


class LegacyPointModel(_Hourly):
    """A shipped ``{model, scaler, features}`` XGBoost artifact on the grid; terrain
    is sampled at the cell centre (the models' terrain effect is ~nil anyway)."""

    def __init__(self, ctx: GridContext, path: str | Path, name: str | None = None):
        art = joblib.load(path)
        self.ctx, self.name = ctx, name or Path(path).stem
        self.model, self.scaler, self.features = art["model"], art["scaler"], list(art["features"])
        self.terrain = _cell_terrain(ctx.store)
        self.mean = self.scaler.mean_.astype(np.float32)
        self.scale = self.scaler.scale_.astype(np.float32)

    def _predict(self, cols) -> np.ndarray:
        X = (np.column_stack(cols).astype(np.float32) - self.mean) / self.scale
        return self.model.predict_proba(X)[:, 1]

    def score_hours(self, h0, h1):
        st, C = self.ctx.store, self.ctx.store.n_cells
        n = h1 - h0
        cols = [np.tile(self.terrain[f].to_numpy(np.float32), n) if f in self.terrain
                else np.asarray(st.var(f)[h0:h1]).reshape(-1) for f in self.features]
        return self._predict(cols).reshape(n, C)

    def score_rows(self, hi, ci):
        st = self.ctx.store
        cols = [self.terrain[f].to_numpy(np.float32)[ci] if f in self.terrain else np.asarray(st.var(f)[hi, ci])
                for f in self.features]
        return self._predict(cols)


class LegacySequenceModel(_Hourly):
    """A shipped GRU / BiGRU on store sequences (the ``lookback`` hours before
    each hour). Too slow for every cell-hour: score it on a weighted sample."""

    expensive = True

    def __init__(self, ctx: GridContext, model_type: str = "gru", device: str | None = None):
        import torch
        from app.api.sequence import SequencePredictor

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ctx, self.name = ctx, model_type
        self.sp = SequencePredictor(config.REGION, model_type=model_type, device=dev)
        self.terrain = _cell_terrain(ctx.store)[self.sp.static_features].to_numpy(np.float32)

    def score_rows(self, hi, ci, batch: int = 20000):
        st, L = self.ctx.store, self.sp.lookback
        hi, ci = np.asarray(hi), np.asarray(ci)
        out = np.empty(len(hi), np.float32)
        for i in range(0, len(hi), batch):
            h, c = hi[i:i + batch], ci[i:i + batch]
            rows = np.clip(h[:, None] - np.arange(L, 0, -1)[None, :], 0, None)
            seq = np.stack([np.asarray(st.var(v))[rows, c[:, None]] for v in self.sp.seq_vars], axis=-1)
            out[i:i + batch] = np.atleast_1d(self.sp.predict_proba(np.nan_to_num(seq.astype(np.float32)), self.terrain[c]))
        return out

    def score_hours(self, h0, h1):
        C = self.ctx.store.n_cells
        return self.score_rows(np.repeat(np.arange(h0, h1), C), np.tile(np.arange(C), h1 - h0)).reshape(h1 - h0, C)


# -----------------------------------------------------------------------------
# Grid models
# -----------------------------------------------------------------------------
@dataclass
class Recipe:
    """What a grid model is: feature groups, unit, target and sampling."""

    name: str
    groups: tuple[str, ...]
    unit: str = "hour"                     # "hour" | "day"
    target: str = "y_any"                  # "y_any" | "y_ge1ha" | "y_ge10ha" | "y_ge100ha"
    neg_sampling: str = config.NEG_SAMPLING
    neg_ratio: int = config.NEG_RATIO
    exclude_rekindles: bool = True
    monotone: bool = False
    params: dict = field(default_factory=dict)
    refit: bool = True                     # refit on train + validation years with 1.1x the trees
    cells: str = "all"                     # "land" = the audit's 987-cell grid
    recency_lag_days: int = config.RECENCY_LAG_DAYS
    max_positives: int | None = None       # subsample positives (data-size ablation only)
    calib_years: int = 1                   # years of out-of-sample predictions pooled for the isotonic step
    neural: tuple[str, ...] = ()           # grid_neural networks averaged with XGBoost in log-odds (daily only)
    neural_lookback: int = 90              # days of daily history the temporal members read
    neural_seed: int = 0
    without: tuple[str, ...] = ()          # measured weather variables the model must not use
                                           # (features.WEATHER_DEPENDS), e.g. ("wind_speed",)

    @classmethod
    def named(cls, name: str, **kw) -> "Recipe":
        """A recipe by name.

        ``config.GRID_MODELS`` entries (the production models) carry their unit,
        target and options; ``config.GRID_RECIPES`` entries (``R0``-``R3``) are
        hourly group tuples; ``"R3+recency-dry"`` adds / removes groups from one of
        those (used by the ablations). Extra keyword arguments override fields;
        unknown ones are passed to XGBoost as parameters.
        """
        base: dict = {}
        if name in getattr(config, "GRID_MODELS", {}):
            base = dict(config.GRID_MODELS[name])
            groups = tuple(base.pop("groups"))
        else:
            groups = parse_groups(name)
        fields = cls.__dataclass_fields__
        params = dict(base.pop("params", {}))
        for k, v in list(kw.items()):
            if k not in fields:
                params[k] = kw.pop(k)
        base.update(kw)
        return cls(name=name, groups=groups, params=params, **base)

    @property
    def features(self) -> list[str]:
        drop = features_without(self.without)
        return [c for c in columns(self.groups, daily=self.unit == "day") if c not in drop]


def parse_groups(spec: str) -> tuple[str, ...]:
    """``"R3"`` -> its groups; ``"R3+recency-dry"`` -> R3 plus recency minus dry."""
    import re

    parts = re.findall(r"[+-]?[A-Za-z0-9_]+", spec)
    head = parts[0].lstrip("+")
    if head not in config.GRID_RECIPES:
        raise KeyError(f"unknown recipe {head!r}; known: {sorted(config.GRID_RECIPES)}")
    groups = list(config.GRID_RECIPES[head])
    for p in parts[1:]:
        g = p[1:]
        if g not in GROUPS:
            raise KeyError(f"unknown feature group {g!r} in {spec!r}")
        if p[0] == "+" and g not in groups:
            groups.append(g)
        elif p[0] == "-" and g in groups:
            groups.remove(g)
    return tuple(groups)


def is_recipe(name: str) -> bool:
    if name in getattr(config, "GRID_MODELS", {}):
        return True
    try:
        parse_groups(name)
        return True
    except KeyError:
        return False


def _iso_fit(x: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    iso.fit(np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel())
    return {"x": iso.X_thresholds_.astype(np.float64), "y": iso.y_thresholds_.astype(np.float64)}


def _iso_apply(m: dict | None, x: np.ndarray) -> np.ndarray:
    if m is None:
        return np.asarray(x, dtype=np.float64)
    return np.interp(np.asarray(x, dtype=np.float64), m["x"], m["y"])


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def prior_shift(p: np.ndarray, r: float) -> np.ndarray:
    """Sample-space probability -> population probability, for a known odds ratio
    ``r`` = population odds / sample odds: ``p r / (p r + 1 - p)``."""
    p = np.asarray(p, dtype=np.float64)
    return p * r / (p * r + 1.0 - p)


def noisy_or(hourly_p: np.ndarray, axis: int = 1) -> np.ndarray:
    """P(at least one event) from independent per-hour probabilities."""
    return 1.0 - np.exp(np.log1p(-np.clip(hourly_p, 0.0, 1.0 - 1e-12)).sum(axis=axis))


class GridModel:
    """A trained grid model: XGBoost booster + the arrays needed to rebuild its
    features on any store hour/day + calibration. Bind it to a ``GridContext``
    before scoring (``model.bind(ctx)``)."""

    def __init__(self, recipe: Recipe, model, feature_names: list[str], priors: dict, meta: dict,
                 calibration: dict | None = None, soil_clim: dict | None = None, input_quantiles: dict | None = None,
                 members: list | None = None):
        self.recipe, self.model, self.features = recipe, model, feature_names
        self.members = list(members or [])          # grid_neural.NeuralMember
        self.priors, self.meta, self.calibration = priors, meta, calibration or {}
        self.soil_clim, self.input_quantiles = soil_clim, input_quantiles or {}
        self.name, self.unit, self.expensive = recipe.name, recipe.unit, False
        self.ctx = self.builder = None

    # -- binding / persistence ---------------------------------------------------
    def bind(self, ctx: GridContext) -> "GridModel":
        sc = None
        if self.soil_clim is not None:
            sc = SoilClimatology.__new__(SoilClimatology)
            sc.clim = self.soil_clim
        self.ctx = ctx
        self.builder = ctx.builder(lag_days=self.recipe.recency_lag_days, soil_clim=sc)
        return self

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or config.GRID_MODELS_DIR / f"{self.name}.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "artifact_version": ARTIFACT_VERSION, "recipe": asdict(self.recipe), "model": self.model,
            "features": self.features, "priors": self.priors, "meta": self.meta,
            "calibration": self.calibration, "soil_clim": self.soil_clim, "input_quantiles": self.input_quantiles,
            "members": [mb.to_dict() for mb in self.members],
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path, ctx: GridContext | None = None) -> "GridModel":
        a = joblib.load(path)
        rec = a["recipe"]
        rec["groups"] = tuple(rec["groups"])
        rec["neural"] = tuple(rec.get("neural", ()))
        rec["without"] = tuple(rec.get("without", ()))
        members = []
        if a.get("members"):
            from .grid_neural import NeuralMember

            members = [NeuralMember.from_dict(d) for d in a["members"]]
        m = cls(Recipe(**rec), a["model"], a["features"], a["priors"], a["meta"], a.get("calibration"),
                a.get("soil_clim"), a.get("input_quantiles"), members)
        return m.bind(ctx) if ctx is not None else m

    # -- raw scores (sample space) -----------------------------------------------------
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """The XGBoost member alone (sample-space probability)."""
        return self.model.predict_proba(X[self.features])[:, 1].astype(np.float32)

    def predict_rows(self, X: pd.DataFrame, di, ci, store=None) -> np.ndarray:
        """Sample-space probability of daily rows ``(di, ci)`` whose features are ``X``:
        XGBoost, averaged in log-odds with the neural members if the recipe has any.
        ``store`` defaults to the bound context's (the live service passes its window)."""
        p = self._predict(X)
        if not self.members:
            return p
        store = store if store is not None else self.ctx.store
        z = [logit(p)] + [logit(mb.predict(X, np.asarray(di), np.asarray(ci), store, self.ctx.events))
                          for mb in self.members]
        return (1.0 / (1.0 + np.exp(-np.mean(z, axis=0)))).astype(np.float32)

    def predict_block(self, X: pd.DataFrame, d0: int, d1: int, store=None) -> np.ndarray:
        """``predict_rows`` for days ``[d0, d1)`` x every cell (``X`` from ``daily_block``)."""
        C = (store if store is not None else self.ctx.store).n_cells
        di = np.repeat(np.arange(d0, d1), C)
        ci = np.tile(np.arange(C), d1 - d0)
        return self.predict_rows(X, di, ci, store)

    def score_hours(self, h0: int, h1: int) -> np.ndarray:
        if self.unit != "hour":
            raise ValueError(f"{self.name} is a daily model")
        C = self.ctx.store.n_cells
        return score_hours_chunked(
            lambda a, b: self._predict(self.builder.hourly_block(a, b, self.recipe.groups, self.priors)), h0, h1, C)

    def score_rows(self, hi, ci) -> np.ndarray:
        if self.unit == "hour":
            return self._predict(self.builder.hourly(hi, ci, self.recipe.groups, self.priors))
        return self.predict_rows(self.builder.daily(hi, ci, self.recipe.groups, self.priors), hi, ci)

    def score_days(self, d0: int, d1: int) -> np.ndarray:
        C = self.ctx.store.n_cells
        if self.unit == "day":
            X = self.builder.daily_block(d0, d1, self.recipe.groups, self.priors)
            return self.predict_block(X, d0, d1).reshape(d1 - d0, C)
        return self.score_hours(d0 * 24, d1 * 24).reshape(d1 - d0, 24, C).max(axis=1)

    # -- probabilities -------------------------------------------------------------------
    def probability(self, raw: np.ndarray) -> np.ndarray:
        """Calibrated probability per unit (hour or day) from raw scores."""
        c = self.calibration
        return _iso_apply(c.get("iso_unit"), prior_shift(raw, c.get("odds_ratio", 1.0)))

    def daily_probability(self, d0: int, d1: int, raw: np.ndarray | None = None) -> np.ndarray:
        """P(at least one fire in the cell on the day), (days, cells)."""
        C = self.ctx.store.n_cells
        if self.unit == "day":
            raw = self.score_days(d0, d1) if raw is None else raw
            return self.probability(raw)
        raw = self.score_hours(d0 * 24, d1 * 24) if raw is None else raw
        ph = self.probability(raw).reshape(d1 - d0, 24, C)
        return _iso_apply(self.calibration.get("iso_daily"), noisy_or(ph, axis=1))


def _xgb(params: dict, device: str | None):
    import xgboost as xgb

    p = {**config.GRID_XGB_PARAMS, **params}
    if device:
        p["device"] = device
    else:
        p.setdefault("n_jobs", 16)
    p["verbosity"] = 0
    return xgb.XGBClassifier(**p)


def training_sample(recipe: Recipe, ctx: GridContext, t_start, t_end, *, seed: int = config.SAMPLE_RANDOM_SEED):
    """Positives + case-base (or halo) negatives for ``[t_start, t_end)``.

    Returns ``(unit_index, cell_index, y, info)``; units are hours or days."""
    st, ev = ctx.store, ctx.events
    pool = ctx.cell_pool(recipe.cells)
    allowed = np.zeros(st.n_cells, dtype=bool)
    allowed[pool] = True
    t_start, t_end = pd.Timestamp(t_start), pd.Timestamp(t_end)
    on = ev["cell"].to_numpy() >= 0
    on[on] = allowed[ev["cell"].to_numpy()[on]]
    f = ev[on & (ev["t"] >= t_start) & (ev["t"] < t_end)]
    if recipe.unit == "hour":
        f = f[f["timed"]]
    if recipe.target != "y_any":
        f = f[f[recipe.target].astype(bool)]
    if recipe.exclude_rekindles:
        f = f[~f["is_rekindle"]]
    if recipe.unit == "hour":
        u = st.hour_index(f["t"])
        lo, hi = st.hour_range(t_start, t_end)
        per_day = 24
    else:
        u = st.day_index(f["day"])
        lo, hi = st.day_range(t_start, t_end)
        per_day = 1
    pos = pd.DataFrame({"u": u, "c": f["cell"].to_numpy()})
    pos = pos[(pos["u"] >= lo) & (pos["u"] < hi)].drop_duplicates(["u", "c"])
    if recipe.max_positives is not None and len(pos) > recipe.max_positives:
        pos = pos.sample(n=recipe.max_positives, random_state=seed).sort_index()
    # a unit's cell-day is off-limits for negatives if it holds a target fire (any
    # cause, whole registry) — rekindles included, so they are neither class
    tgt = ev[ev["cell"] >= 0]
    if recipe.target != "y_any":
        tgt = tgt[tgt[recipe.target].astype(bool)]
    keys = fire_cellday_keys(st.day_index(tgt["day"]), tgt["cell"].to_numpy(), st.n_cells)
    forbidden = None
    if recipe.neg_sampling == "halo":
        halo = config.WEATHER_LOOKBACK_DAYS + config.ACTIVE_NEG_BUFFER_DAYS
        forbidden = halo_forbidden_celldays(st.day_index(tgt["day"]), tgt["cell"].to_numpy(),
                                            tgt["duration_days"].to_numpy(), st.n_days, st.n_cells, halo)
    elif recipe.neg_sampling != "case_base":
        raise ValueError(f"unknown neg_sampling {recipe.neg_sampling!r}")
    n_neg = recipe.neg_ratio * len(pos)
    nu, nc = generate_case_base_negatives(
        n_neg, n_cells=st.n_cells, unit_range=(lo, hi), fire_keys=keys, units_per_day=per_day,
        forbidden=forbidden, cells=None if recipe.cells == "all" else pool, random_state=seed)
    U = np.concatenate([pos["u"].to_numpy(), nu])
    Cc = np.concatenate([pos["c"].to_numpy(), nc])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(nu))]).astype(np.int8)
    n_units = len(pool) * (hi - lo)
    info = {"n_pos": int(len(pos)), "n_neg": int(len(nu)), "n_units": int(n_units),
            "unit_range": [int(lo), int(hi)], "neg_sampling": recipe.neg_sampling}
    return U, Cc, y, info


def train(
    recipe: Recipe,
    ctx: GridContext,
    *,
    test_from,
    train_start=config.GRID_TRAIN_START,
    val_years: int = config.GRID_VALIDATION_YEARS,
    calibrate: bool = True,
    device: str | None = None,
    verbose: bool = True,
) -> GridModel:
    """Train a grid model on ``[train_start, test_from)``.

    The last ``val_years`` before ``test_from`` are the validation period: early
    stopping (log-loss on the validation sample) and calibration (isotonic
    regression on the validation period's *full grid*, after the analytic
    prior-shift correction) are fitted there — never on the test period. With
    ``recipe.refit`` the booster is then refitted on train + validation with 1.1x
    the early-stopped number of trees, and keeps the validation calibration.
    """
    t0 = time.time()
    st = ctx.store
    t_start, t_test = pd.Timestamp(train_start), pd.Timestamp(test_from)
    t_val = t_test - pd.DateOffset(years=val_years)
    groups, daily = recipe.groups, recipe.unit == "day"
    log = (lambda *a: print(f"  [{recipe.name}]", *a, f"[{time.time() - t0:.0f}s]", flush=True)) if verbose else (lambda *a: None)

    U, Cc, y, info = training_sample(recipe, ctx, t_start, t_test)
    log(f"sample: {info['n_pos']:,} positives + {info['n_neg']:,} negatives ({recipe.neg_sampling}, 1:{recipe.neg_ratio})")
    soil = SoilClimatology(st, t_start, t_test) if "soil_anom" in groups else None
    builder = ctx.builder(lag_days=recipe.recency_lag_days, soil_clim=soil)
    priors = ctx.priors(recipe, t_start, t_test)
    X = (builder.daily_training if daily else builder.hourly_training)(U, Cc, groups, priors)
    if recipe.without:
        if recipe.neural:
            raise ValueError("Recipe.without is XGBoost-only: the neural members read every weather channel")
        X = X.drop(columns=sorted(features_without(recipe.without) & set(X.columns)))
    feats = list(X.columns)
    log(f"features {X.shape}")

    u_val = st.day_range(t_val, t_val)[0] if daily else st.hour_range(t_val, t_val)[0]
    tr, va = U < u_val, U >= u_val
    params = dict(recipe.params)
    if recipe.monotone:
        params["monotone_constraints"] = {f: MONOTONE[f] for f in feats if f in MONOTONE}
    m = _xgb(params, device).fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    best = int(m.best_iteration) + 1
    from sklearn.metrics import average_precision_score

    p_val = m.predict_proba(X[va])[:, 1]
    val_ap = float(average_precision_score(y[va], p_val))
    log(f"early-stopped at {best} trees; validation-sample PR-AUC {val_ap:.4f}")

    # neural members (grid_neural): same sample, early stopping on the same validation year
    members = []
    if recipe.neural:
        if not daily:
            raise ValueError("neural members are daily only")
        from .grid_neural import train_member

        for arch in recipe.neural:
            members.append(train_member(arch, ctx, recipe, (U, Cc, y, X), test_from=t_test, train_start=t_start,
                                        val_years=val_years, lookback=recipe.neural_lookback,
                                        seed=recipe.neural_seed, log=log))
            log(f"neural member {arch}: validation-sample PR-AUC {members[-1].meta['validation_sample_pr_auc']:.4f}")

    # --- calibration on the validation period's full grid (priors from the training years only)
    pi_pop = info["n_pos"] / info["n_units"]
    pi_s = info["n_pos"] / (info["n_pos"] + info["n_neg"])
    odds_ratio = (pi_pop / (1 - pi_pop)) / (pi_s / (1 - pi_s))
    calibration: dict = {"odds_ratio": float(odds_ratio), "population_rate": float(pi_pop), "sample_rate": float(pi_s)}
    if calibrate:
        # out-of-sample predictions on the validation year (this model), plus — with
        # calib_years > 1 — on each earlier year from a model trained only on the
        # years before it; one isotonic map is fitted on the pooled set
        parts = [_calibration_set(m, recipe, ctx, builder, feats, t_start, t_val, t_test, odds_ratio, members)]
        for k in range(1, max(int(recipe.calib_years), 1)):
            y1 = t_test - pd.DateOffset(years=k)
            y0 = y1 - pd.DateOffset(years=1)
            if y0 - pd.DateOffset(years=2) < t_start:
                break
            sub = Recipe(**{**asdict(recipe), "groups": tuple(recipe.groups), "refit": False, "calib_years": 1})
            mk = train(sub, ctx, test_from=y1, train_start=t_start, val_years=1, calibrate=False,
                       device=device, verbose=False)
            parts.append(_calibration_set(mk.model, recipe, ctx, builder, feats, t_start, y0, y1,
                                          mk.calibration["odds_ratio"], mk.members))
            log(f"calibration year {y0:%Y-%m-%d}..{y1:%Y-%m-%d} scored by a model trained before it")
        ps = np.concatenate([p[0] for p in parts])
        yu = np.concatenate([p[1] for p in parts])
        if daily:
            calibration["iso_unit"] = _iso_fit(ps, yu)
            pd_ = _iso_apply(calibration["iso_unit"], parts[0][0])
        else:
            calibration["iso_unit"] = _iso_fit(ps, yu)
            ph = [_iso_apply(calibration["iso_unit"], p[0]).reshape(p[3], 24, -1) for p in parts]
            nd = np.concatenate([noisy_or(x, axis=1).ravel() for x in ph])
            calibration["iso_daily"] = _iso_fit(nd, np.concatenate([p[2].ravel() for p in parts]))
            pd_ = _iso_apply(calibration["iso_daily"], noisy_or(ph[0], axis=1))
        calibration["calibration_years"] = len(parts)
        calibration["validation_daily_mean_pred"] = float(np.mean(pd_))
        calibration["validation_daily_obs_rate"] = float(parts[0][2].mean())
        log(f"calibrated on {len(parts)} out-of-sample year(s): validation mean daily p {np.mean(pd_):.5f} "
            f"vs observed {parts[0][2].mean():.5f}")

    if recipe.refit:
        params2 = {**params, "n_estimators": int(best * 1.1)}
        params2["early_stopping_rounds"] = None
        m = _xgb(params2, device).fit(X, y, verbose=False)
        log(f"refitted on train + validation with {int(best * 1.1)} trees")

    # training-input distribution (negatives = the population), overall and by
    # calendar month, for serving-time drift checks
    neg = X[y == 0]
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    unit_month = (st.days if daily else st.hours)[U[y == 0]].month.to_numpy()
    input_quantiles = {
        "q": qs,
        "features": {c: np.nanquantile(neg[c].to_numpy(np.float64), qs).tolist() for c in feats},
        "by_month": {int(mo): {c: np.nanquantile(neg[c].to_numpy(np.float64)[unit_month == mo], qs).tolist()
                               for c in feats} for mo in np.unique(unit_month)},
    }
    meta = {
        "region": config.REGION, "protocol": ctx.protocol, "local_tz": ctx.local_tz,
        "train_start": str(t_start), "validation_from": str(t_val), "test_from": str(t_test),
        "sample": info, "best_iteration": best, "validation_sample_pr_auc": val_ap,
        "store": str(st.path), "store_built": st.meta.get("built"), "n_cells": int(st.n_cells),
        "trained": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
        "importance_total_gain": _importance(m, feats),
    }
    gm = GridModel(recipe, m, feats, priors.arrays(None), meta, calibration,
                   soil.clim if soil is not None else None, input_quantiles, members)
    return gm.bind(ctx)


def _calibration_set(model, recipe: Recipe, ctx: GridContext, builder, feats, t_start, y0, y1, odds_ratio,
                     members=None):
    """Prior-shifted predictions of ``model`` on every unit of ``[y0, y1)`` (priors
    from ``[t_start, y0)`` only) and the labels: ``(p, y_unit, y_daily, n_days)``."""
    from .evaluation import GridTarget

    st = ctx.store
    g = GridTarget(st, ctx.events, start=y0, end=y1, cells=recipe.cells, target=recipe.target, include_rekindles=True)
    prior = ctx.priors(recipe, t_start, y0).arrays(None)
    tmp = GridModel(recipe, model, feats, prior, {}, members=members).bind(ctx)
    tmp.builder = builder
    if recipe.unit == "day":
        p = prior_shift(g.select(tmp.score_days(g.d0, g.d1), "day"), odds_ratio).ravel()
        return p, g.Yd.ravel(), g.Yd, g.ND
    p = prior_shift(g.select(tmp.score_hours(g.h0, g.h1)), odds_ratio).ravel()
    return p, g.Yh.ravel(), g.Yd, g.ND


def _importance(m, feats) -> dict:
    try:
        g = m.get_booster().get_score(importance_type="total_gain")
    except Exception:  # noqa: BLE001
        return {}
    tot = sum(g.values()) or 1.0
    return {k: round(g.get(k, 0.0) / tot, 5) for k in feats}


# -----------------------------------------------------------------------------
# Registry of named models for the harness
# -----------------------------------------------------------------------------
SHIPPED = {
    "xgb7": "fire_xgb_{region}.joblib",
    "xgb18": "fire_xgb_{region}_op.joblib",
    "xgb16": "fire_xgb_{region}_op_strict.joblib",
}


def make_model(name: str, ctx: GridContext, *, test_from=None, **train_kw):
    """Resolve a model name for the harness.

    ``fwi``, ``fwi_history``, ``climatology_legacy``; shipped ``xgb7`` / ``xgb18`` /
    ``xgb16`` / ``gru`` / ``bigru``; a recipe in ``config.GRID_RECIPES`` (trained
    here with ``test_from``); or a path to a saved grid model / shipped artifact.
    """
    test_from = test_from or config.GRID_REFERENCE_TEST_FROM
    models_dir = config.OUTPUTS_DIR / "models"
    if name == "fwi":
        return FWIBaseline(ctx)
    if name == "fwi_history":
        return FWIHistoryBaseline(ctx, test_from)
    if name == "climatology_legacy":
        return ClimatologyLegacy(ctx, test_from)
    if name in SHIPPED:
        return LegacyPointModel(ctx, models_dir / SHIPPED[name].format(region=config.REGION), name)
    if name in ("gru", "bigru"):
        return LegacySequenceModel(ctx, name)
    if is_recipe(name):
        kw = {k: train_kw.pop(k) for k in list(train_kw) if k in ("train_start", "val_years", "calibrate", "device", "verbose")}
        return train(Recipe.named(name, **train_kw), ctx, test_from=test_from, **kw)
    p = Path(name)
    if p.suffix == ".joblib" and p.exists():
        art = joblib.load(p)
        if "recipe" in art:
            return GridModel.load(p, ctx)
        return LegacyPointModel(ctx, p)
    raise KeyError(f"unknown model {name!r}")
