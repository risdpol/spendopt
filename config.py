# optimizer_context.py

from dataclasses import dataclass, field
import pandas as pd
import numpy as np

@dataclass
class OptimizerContext:
    # -----------------------------
    # Dynamic user inputs
    # -----------------------------
    df: pd.DataFrame
    budget: float
    objective: str               # "revenue", "profit", "roi", "npm"
    roi_min: float
    promo_share: dict | None

    # -----------------------------
    # Static configuration params
    # -----------------------------
    EPS: float = 1e-6
    DECISION_KEY_COLS: list = field(default_factory=lambda: ["sku_id", "store_id", "year", "week"])
    ELASTICITY_KEY_COLS: list = field(default_factory=lambda: ["sku_id", "store_id", "promo_mechanics"])

    MIN_UNIQUE_SPEND: int = 3
    MIN_ROWS_PER_PAIR: int = 6

    N_LEVELS: int = 5
    LEVEL_RULE: str = "quantile"
    FIXED_STEP: float = 1000.0
    INCLUDE_ZERO_LEVEL: bool = True

    MARGIN_BY_SKU: dict | None = None

    PER_SKU_MAX_WEEKS: int | dict | None = None
    PER_PROMO_CAP: dict | None = None
    WEEKLY_BUDGETS: dict | None = None

    MUST_RUN: set = field(default_factory=set)
    MUST_NOT_RUN: set = field(default_factory=set)

    RANDOM_STATE: int = 42

    # These are produced inside pipeline steps
    base_for_bp: pd.DataFrame | None = None
    beta_by_key: dict | None = None
    global_beta: float | None = None
