import os
import pickle
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException, Body, Request, Query
from typing import Dict, Any
import json
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNK = "UNKNOWN"
OTHER = "OTHER"

# Auto-fill rules: target column -> default value
AUTO_FILL_RULES = {
    "PHA": "EVN.PHA_3P",
    "KIEU_MC": "TBI_CT_MC_KIEU_MC_01",
    "KIEU_DAPHQ": "TBI_TT_MC_KIEU_DAPHQ.00001",
    "KIEU_CD": "TBI_CT_MC_CC_CD.00001",
    "U_TT": "TBI_CT_MC_U_TT_02",
}

# Validation rules: invalid values that should trigger error
INVALID_NATIONALFACT = "TB040.00023"

# Determine MODEL_DIR - try multiple paths
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
MODEL_DIR = os.path.join(_project_root, "model")

# Fallback: try relative to current working directory
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = os.path.join(os.getcwd(), "model")
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = os.path.join(os.getcwd(), "..", "model")


# ---------------------------------------------------------------------------
# Utils / preprocessing
# ---------------------------------------------------------------------------
def norm_cat(val: Union[str, int, float]) -> str:
    """Normalize categorical text to a safe string."""
    s = str(val).strip()
    if s == "" or s.lower() in {"na", "n/a", "null", "none", "unknown", "nan"}:
        return UNK
    return s


def le_transform_safe(le: LabelEncoder, s: pd.Series) -> np.ndarray:
    """Safely transform using LabelEncoder, mapping unknown values to UNK."""
    s = s.astype(str).fillna(UNK)
    known = set(le.classes_)
    s = s.where(s.isin(known), UNK)
    return le.transform(s)


# ---------------------------------------------------------------------------
# Model helper classes (must match training definitions for unpickle)
# ---------------------------------------------------------------------------
class TrainOnlyImputer:
    """Imputer that fills missing values based on training data statistics."""
    def __init__(self, unk=UNK):
        self.unk = unk
        self.fill = {}

    def fit(self, df_fit: pd.DataFrame):
        for c in df_fit.columns:
            if df_fit[c].dtype == "object" or str(df_fit[c].dtype).startswith("string"):
                m = df_fit[c].mode(dropna=True)
                self.fill[c] = m.iloc[0] if len(m) else self.unk
            else:
                self.fill[c] = float(df_fit[c].median())
        return self

    def transform(self, df_in: pd.DataFrame):
        out = df_in.copy()
        for c, v in self.fill.items():
            out[c] = out[c].fillna(v)
        return out


def generate_backoff_strategy(input_cols: List[str]) -> List[List[str]]:
    """
    Generate backoff strategy based on input columns.
    Creates all possible subsets of input_cols in order of decreasing size.
    """
    from itertools import combinations
    
    n = len(input_cols)
    backoffs = []
    
    # Add all subsets from size n down to 0
    for r in range(n, -1, -1):
        if r == 0:
            backoffs.append([])  # Global fallback
        else:
            # Add all combinations of size r
            for combo in combinations(input_cols, r):
                backoffs.append(list(combo))
    
    return backoffs


class TransitionTable:
    """Transition table model with backoff strategy."""
    def __init__(self, alpha=1.0, input_cols=None):
        self.alpha = alpha
        self.classes = None
        self.counts = None
        self.input_cols = input_cols  # Store input_cols for prediction
        self.backoffs = None  # Store backoff strategy

    def fit(self, X, y, input_cols=None):
        """
        Fit TransitionTable model.
        
        Args:
            X: Input features DataFrame
            y: Target Series
            input_cols: List of input column names. If None, uses X.columns
        """
        y = y.astype(str).reset_index(drop=True)
        X = X.reset_index(drop=True)
        self.classes = sorted(y.unique().tolist())
        
        # Determine input_cols
        if input_cols is None:
            if self.input_cols is None:
                self.input_cols = list(X.columns)
            else:
                input_cols = self.input_cols
        else:
            self.input_cols = input_cols
        
        # Ensure all input_cols exist in X
        missing_cols = [c for c in self.input_cols if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Input columns not found in X: {missing_cols}")
        
        # Generate backoff strategy based on input_cols
        self.backoffs = generate_backoff_strategy(self.input_cols)
        self.counts = {tuple(b): {} for b in self.backoffs}

        for i in range(len(X)):
            row = X.loc[i]
            for b in self.backoffs:
                k = tuple(b)
                feat_key = tuple(row[c] for c in b) if b else ("__GLOBAL__",)
                d = self.counts[k].setdefault(feat_key, {})
                d[y[i]] = d.get(y[i], 0) + 1
        return self

    def predict_proba_one(self, xrow):
        """Predict probability for a single row."""
        if self.backoffs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        for b in self.backoffs:
            k = tuple(b)
            feat_key = tuple(xrow[c] for c in b) if b else ("__GLOBAL__",)
            if feat_key in self.counts[k]:
                counts = self.counts[k][feat_key]
                denom = sum(counts.values()) + self.alpha * len(self.classes)
                return np.array([(counts.get(c, 0) + self.alpha) / denom for c in self.classes], dtype=float)
        return np.ones(len(self.classes)) / len(self.classes)

    def predict_proba(self, X):
        rows = X.to_dict(orient="records")
        return np.vstack([self.predict_proba_one(r) for r in rows])

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.array(self.classes)[np.argmax(proba, axis=1)]

    def predict_topk(self, X: pd.DataFrame, k: int = 3) -> List[List[str]]:
        """Predict top-k classes for each row."""
        proba = self.predict_proba(X)
        topk_idx = np.argsort(-proba, axis=1)[:, :k]
        return [[self.classes[j] for j in row] for row in topk_idx]


class MiniTabTransformer(nn.Module):
    """TabTransformer model for tabular data."""
    def __init__(self, cardinals, n_classes, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, d_model) for c in cardinals])
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        toks = torch.stack([self.embs[i](x[:, i]) for i in range(x.shape[1])], dim=1)
        z = self.encoder(toks)
        pooled = z.mean(dim=1)
        return self.head(pooled)


@dataclass
class AutoFillBundle:
    """Bundle containing all autofill models and metadata."""
    version: str
    created_at: str
    input_cols: List[str]
    target_cols: List[str]
    all_cols: List[str]
    stats: Dict
    imputer: object
    best_pack: Dict


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
CLASS_MAP = {
    # Pickles created in notebooks (module may be __main__ or __mp_main__)
    ("__main__", "TransitionTable"): TransitionTable,
    ("__main__", "TrainOnlyImputer"): TrainOnlyImputer,
    ("__main__", "MiniTabTransformer"): MiniTabTransformer,
    ("__main__", "AutoFillBundle"): AutoFillBundle,
    ("__mp_main__", "TransitionTable"): TransitionTable,
    ("__mp_main__", "TrainOnlyImputer"): TrainOnlyImputer,
    ("__mp_main__", "MiniTabTransformer"): MiniTabTransformer,
    ("__mp_main__", "AutoFillBundle"): AutoFillBundle,
}


class _CompatUnpickler(pickle.Unpickler):
    """Allow loading pickles created in notebooks where classes lived in __main__ or __mp_main__."""

    def find_class(self, module, name):  # type: ignore[override]
        key = (module, name)
        if key in CLASS_MAP:
            return CLASS_MAP[key]
        return super().find_class(module, name)


def load_pickle(path: str):
    """Load pickle file with compatibility for notebook-created classes."""
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


def catboost_topk(model: CatBoostClassifier, X_df: pd.DataFrame, k: int = 3) -> Tuple[List[List[str]], List[List[float]]]:
    """Get top-k predictions and probabilities from CatBoost model."""
    proba = model.predict_proba(X_df)
    # Recover original string classes from training if available
    if hasattr(model, "_classes"):
        classes = list(model._classes)
    elif hasattr(model, "_y_enc"):
        classes = model._y_enc.classes_.tolist()
    else:
        classes = model.classes_.tolist()
    topk_idx = np.argsort(-proba, axis=1)[:, :k]
    topk_labels = [[classes[j] for j in row] for row in topk_idx]
    topk_proba = [[proba[i][j] for j in row] for i, row in enumerate(topk_idx)]
    return topk_labels, topk_proba


def tabtransformer_topk(model, x_enc: Dict[str, LabelEncoder], y_enc: LabelEncoder, 
                        X_df: pd.DataFrame, input_cols: List[str], k: int = 3) -> Tuple[List[List[str]], List[List[float]]]:
    """Get top-k predictions and probabilities from TabTransformer model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X_df.astype(str)
    Xi = np.stack([le_transform_safe(x_enc[c], X[c]) for c in input_cols], axis=1).astype(np.int64)
    xb = torch.tensor(Xi).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(xb).cpu().numpy()
        proba = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = proba / proba.sum(axis=1, keepdims=True)

    classes = y_enc.classes_.tolist()
    topk_idx = np.argsort(-proba, axis=1)[:, :k]
    topk_labels = [[classes[j] for j in row] for row in topk_idx]
    topk_proba = [[proba[i][j] for j in row] for i, row in enumerate(topk_idx)]
    return topk_labels, topk_proba


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="PMIS AutoFill API", version="1.0.0")

# Global variables
BUNDLE = None


class PredictRequest(BaseModel):
    """Request model for autofill prediction (fixed 1 field: CATEGORYID)."""
    categoryid: str


class DynamicPredictRequest(BaseModel):
    """Request model for dynamic autofill prediction - accepts any fields."""
    class Config:
        extra = "allow"  # Allow any additional fields
    
    def dict(self):
        """Return all fields as dict, excluding internal Pydantic fields."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def _load_bundle():
    """Load the autofill bundle (contains all models in best_pack).
    
    Ưu tiên dùng bundle đã cải tiến nếu tồn tại: `autofill_bundle_improved.pkl`,
    nếu không có thì fallback về `autofill_bundle.pkl`.
    """
    global BUNDLE

    bundle_path_improved = os.path.join(MODEL_DIR, "autofill_bundle_improved.pkl")
    bundle_path_default = os.path.join(MODEL_DIR, "autofill_bundle.pkl")

    if os.path.exists(bundle_path_improved):
        bundle_path = bundle_path_improved
    else:
        bundle_path = bundle_path_default

    print(f"[INFO] Looking for bundle at: {bundle_path}")
    print(f"[INFO] MODEL_DIR: {MODEL_DIR}")
    print(f"[INFO] Bundle exists: {os.path.exists(bundle_path)}")

    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Bundle file not found: {bundle_path}. Please check MODEL_DIR: {MODEL_DIR}"
        )

    try:
        BUNDLE = load_pickle(bundle_path)
        print(f"[INFO] Bundle loaded successfully. Target cols: {BUNDLE.target_cols}")
        print(f"[INFO] Models in bundle: {len(BUNDLE.best_pack)}")
        print(f"[INFO] Available targets: {list(BUNDLE.best_pack.keys())}")
    except Exception as e:
        raise Exception(f"Failed to load bundle: {str(e)}")


def _validate_input(req: PredictRequest):
    """Validate input and check for invalid values."""
    # No specific validation needed for CATEGORYID
    pass


def _build_input_df(req: PredictRequest) -> pd.DataFrame:
    """Build input dataframe from request (fixed 1 field: CATEGORYID)."""
    data = {
        "CATEGORYID": norm_cat(req.categoryid),
    }
    return pd.DataFrame([data])


def _build_input_df_from_dict(input_dict: Dict[str, str]) -> pd.DataFrame:
    """Build input dataframe from dynamic dictionary."""
    # Normalize all values
    data = {k: norm_cat(v) for k, v in input_dict.items()}
    return pd.DataFrame([data])


def _get_target_cols_from_input(input_cols: List[str], all_cols: List[str]) -> List[str]:
    """Get target columns based on input columns."""
    return [c for c in all_cols if c not in input_cols]


def _apply_auto_fill_rules(target_col: str) -> Optional[str]:
    """Check if target column has auto-fill rule and return the value."""
    return AUTO_FILL_RULES.get(target_col)


def _generate_topn_combinations(predictions: Dict, target_cols: List[str], n: int = 5) -> List[Dict]:
    """
    Generate top-n combinations of predicted values sorted by overall confidence.
    
    Args:
        predictions: Dictionary with predictions for each target column
        target_cols: List of target columns to predict
        n: Number of combinations to generate (default: 5)
    
    Returns:
        List of n dictionaries, each containing predicted values for all target_cols,
        sorted by overall confidence (descending)
    """
    # Collect top-5 predictions and probabilities for each target
    field_options = {}  # {target_col: [(value, proba), ...]}
    
    for target_col in target_cols:
        if target_col not in predictions:
            continue
        
        pred = predictions[target_col]
        if "error" in pred:
            # For error cases, use top1 if available, otherwise skip
            if "top1" in pred and pred["top1"]:
                field_options[target_col] = [(pred["top1"], 0.5)]
            continue
        
        options = []
        
        # Get top-5 values and probabilities
        # Prefer top5 if available, otherwise use top3
        if "top5" in pred and pred["top5"]:
            top_values = pred["top5"]
            if "proba" in pred and pred["proba"]:
                top_proba = pred["proba"]
            else:
                # Estimate probabilities: decreasing from top1
                top_proba = [0.5, 0.25, 0.15, 0.07, 0.03][:len(top_values)]
        elif "top3" in pred and pred["top3"]:
            top_values = pred["top3"]
            if "proba" in pred and pred["proba"]:
                top_proba = pred["proba"]
            else:
                # Estimate probabilities: top1=0.8, top2=0.15, top3=0.05
                top_proba = [0.8, 0.15, 0.05] if len(top_values) >= 3 else [0.8, 0.2] if len(top_values) >= 2 else [1.0]
        else:
            top_values = []
            top_proba = []
        
        if top_values:
            for i, (val, prob) in enumerate(zip(top_values[:5], top_proba[:5] if len(top_proba) >= len(top_values) else top_proba + [0.0] * (len(top_values) - len(top_proba)))):
                if val and val not in [opt[0] for opt in options]:
                    options.append((val, prob))
        
        # If no options found, use top1
        if not options and "top1" in pred and pred["top1"]:
            options = [(pred["top1"], 1.0)]
        
        if options:
            field_options[target_col] = options
    
    if not field_options:
        return []
    
    # Generate combinations: prioritize top predictions
    # Strategy: Create n combinations by varying which fields use top1, top2, top3, etc.
    combinations = []
    
    # Combination 1: All top1 (highest confidence)
    combo1 = {}
    total_conf1 = 0.0
    for target_col in target_cols:
        if target_col in field_options and len(field_options[target_col]) > 0:
            val, prob = field_options[target_col][0]
            combo1[target_col] = val
            total_conf1 += prob
    if combo1:
        combinations.append((combo1, total_conf1))
    
    # Find fields with multiple options and good probability for top2
    variable_fields = []
    for target_col in target_cols:
        if target_col in field_options and len(field_options[target_col]) > 1:
            top1_prob = field_options[target_col][0][1]
            top2_prob = field_options[target_col][1][1] if len(field_options[target_col]) > 1 else 0
            # Only consider if top2 has reasonable probability
            if top2_prob > 0.1:
                variable_fields.append((target_col, top1_prob, top2_prob))
    
    # Sort by how close top2 is to top1 (closer = more likely to be good alternative)
    variable_fields.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
    
    # Generate more combinations by systematically varying fields
    combo_idx = 1
    
    # Strategy 1: Vary single fields one at a time (top2)
    for var_field in variable_fields[:min(n-1, len(variable_fields))]:
        if combo_idx >= n:
            break
        target_col = var_field[0]
        if len(field_options[target_col]) > 1:
            combo = combo1.copy()
            total_conf = total_conf1
            val, prob = field_options[target_col][1]  # top2
            combo[target_col] = val
            total_conf = total_conf - field_options[target_col][0][1] + prob
            combinations.append((combo, total_conf))
            combo_idx += 1
    
    # Strategy 2: Vary single fields to top3
    for target_col in target_cols:
        if combo_idx >= n:
            break
        if target_col in field_options and len(field_options[target_col]) > 2:
            combo = combo1.copy()
            total_conf = total_conf1
            val, prob = field_options[target_col][2]  # top3
            combo[target_col] = val
            total_conf = total_conf - field_options[target_col][0][1] + prob
            combinations.append((combo, total_conf))
            combo_idx += 1
    
    # Strategy 3: Vary 2 fields at a time (top2)
    if combo_idx < n and len(variable_fields) >= 2:
        for i in range(min(len(variable_fields), n - combo_idx)):
            if combo_idx >= n:
                break
            combo = combo1.copy()
            total_conf = total_conf1
            # Vary first field
            target_col1 = variable_fields[i][0]
            if len(field_options[target_col1]) > 1:
                val, prob = field_options[target_col1][1]
                combo[target_col1] = val
                total_conf = total_conf - field_options[target_col1][0][1] + prob
            # Vary second field
            if i + 1 < len(variable_fields):
                target_col2 = variable_fields[i + 1][0]
                if len(field_options[target_col2]) > 1:
                    val, prob = field_options[target_col2][1]
                    combo[target_col2] = val
                    total_conf = total_conf - field_options[target_col2][0][1] + prob
            combinations.append((combo, total_conf))
            combo_idx += 1
    
    # Strategy 4: Vary fields to top4, top5 if available
    for option_rank in [3, 4]:  # top4, top5
        if combo_idx >= n:
            break
        for target_col in target_cols:
            if combo_idx >= n:
                break
            if target_col in field_options and len(field_options[target_col]) > option_rank:
                combo = combo1.copy()
                total_conf = total_conf1
                val, prob = field_options[target_col][option_rank]
                combo[target_col] = val
                total_conf = total_conf - field_options[target_col][0][1] + prob
                combinations.append((combo, total_conf))
                combo_idx += 1
    
    # Sort by total confidence (descending) and return top 5
    combinations.sort(key=lambda x: x[1], reverse=True)
    
    # Remove duplicates (same combination of values)
    seen = set()
    unique_combos = []
    for combo, conf in combinations:
        combo_key = tuple(sorted(combo.items()))
        if combo_key not in seen:
            seen.add(combo_key)
            unique_combos.append(combo)
            if len(unique_combos) >= 5:
                break
    
    # Fill to n if needed
    if len(unique_combos) == 0:
        return []
    
    # If we have less than n, create variations by changing different fields
    while len(unique_combos) < n:
        # Try to create a new variation by changing one more field
        base_combo = unique_combos[0].copy()
        changed = False
        
        # Find a field we haven't varied yet
        for target_col in target_cols:
            if target_col in field_options and len(field_options[target_col]) > 1:
                # Check if this field is already at top1 in base_combo
                if target_col in base_combo and base_combo[target_col] == field_options[target_col][0][0]:
                    # Try top2
                    if len(field_options[target_col]) > 1:
                        base_combo[target_col] = field_options[target_col][1][0]
                        combo_key = tuple(sorted(base_combo.items()))
                        if combo_key not in seen:
                            seen.add(combo_key)
                            unique_combos.append(base_combo)
                            changed = True
                            break
                    # Try top3
                    elif len(field_options[target_col]) > 2:
                        base_combo[target_col] = field_options[target_col][2][0]
                        combo_key = tuple(sorted(base_combo.items()))
                        if combo_key not in seen:
                            seen.add(combo_key)
                            unique_combos.append(base_combo)
                            changed = True
                            break
        
        # If we can't create more variations, duplicate the best one
        if not changed:
            unique_combos.append(unique_combos[0].copy())
    
    return unique_combos[:n]


def _preprocess_input(X_df: pd.DataFrame, input_cols: List[str] = None) -> pd.DataFrame:
    """Preprocess input using imputer."""
    if input_cols is None:
        input_cols = BUNDLE.input_cols
    
    # Create a full dataframe with all columns, fill missing with NaN
    full_df = pd.DataFrame(index=X_df.index)
    for col in BUNDLE.all_cols:
        if col in X_df.columns:
            full_df[col] = X_df[col]
        else:
            full_df[col] = np.nan
    
    # Apply imputer
    imputed = BUNDLE.imputer.transform(full_df)
    
    # Extract only input columns and fillna with UNK
    # Only use columns that exist in both input_cols and imputed dataframe
    available_cols = [c for c in input_cols if c in imputed.columns]
    X_processed = imputed[available_cols].astype(str).fillna(UNK)
    
    # Fill missing columns with UNK
    for col in input_cols:
        if col not in X_processed.columns:
            X_processed[col] = UNK
    
    return X_processed[input_cols]


def _predict_target(target_col: str, X_df: pd.DataFrame, input_cols: List[str] = None) -> Dict[str, Union[str, List[str]]]:
    """Predict for a specific target column."""
    if BUNDLE is None:
        raise HTTPException(status_code=500, detail="Bundle not loaded")
    
    if input_cols is None:
        input_cols = BUNDLE.input_cols
    
    # Check for auto-fill rule first (highest priority)
    auto_fill_value = _apply_auto_fill_rules(target_col)
    if auto_fill_value is not None:
        return {
            "target": target_col,
            "top1": auto_fill_value,
            "top3": [auto_fill_value] * 3,
            "top5": [auto_fill_value] * 5,
            "proba": [1.0, 0.0, 0.0, 0.0, 0.0],
            "source": "auto_fill_rule"
        }
    
    if target_col not in BUNDLE.best_pack:
        # Fallback: use training stats mode if available (align with notebook behavior)
        stats_info = (BUNDLE.stats or {}).get(target_col, {})
        mode_val = stats_info.get("mode_train")
        if mode_val is not None:
            return {
                "target": target_col,
                "top1": str(mode_val),
                "top3": [str(mode_val)] * 3,
                "top5": [str(mode_val)] * 5,
                "proba": [1.0, 0.0, 0.0, 0.0, 0.0],
                "source": "stats_mode_fallback"
            }
        raise HTTPException(status_code=404, detail=f"Model for target '{target_col}' not found")
    
    model_info = BUNDLE.best_pack[target_col]
    kind = model_info.get("kind", "UNKNOWN")
    
    # Handle AUTO_FILL case
    if kind == "AUTO_FILL":
        fill_value = model_info.get("fill_value", UNK)
        return {
            "target": target_col,
            "top1": str(fill_value),
            "top3": [str(fill_value)] * 3,
            "top5": [str(fill_value)] * 5,
            "proba": [1.0, 0.0, 0.0, 0.0, 0.0]
        }
    
    # Prepare X_df with only the columns needed by the model
    # For models trained with BUNDLE.input_cols, we need to ensure X_df has those columns
    model_input_cols = model_info.get("input_cols", BUNDLE.input_cols)
    
    # Create X_df_model with model's expected input columns
    X_df_model = pd.DataFrame(index=X_df.index)
    for col in model_input_cols:
        if col in X_df.columns:
            X_df_model[col] = X_df[col]
        else:
            X_df_model[col] = UNK  # Fill missing with UNK
    
    # Handle TransitionTable
    if kind == "TransitionTable":
        model = model_info["model"]
        # TransitionTable can handle subset of input_cols due to backoff strategy
        topk = model.predict_topk(X_df_model, k=5)
        proba = model.predict_proba(X_df_model)[0]
        # Get top5 probabilities
        top5_idx = np.argsort(-proba)[:5]
        top5_proba = proba[top5_idx].tolist()
        top5_labels = topk[0][:5] if topk else []
        return {
            "target": target_col,
            "top1": top5_labels[0] if top5_labels else None,
            "top3": top5_labels[:3] if top5_labels else [],
            "top5": top5_labels[:5] if top5_labels else [],
            "proba": top5_proba[:5]
        }
    
    # Handle CatBoost
    if kind == "CatBoost":
        model = model_info["model"]
        # CatBoost requires exact input columns
        if set(model_input_cols) != set(X_df_model.columns):
            # Try to use only available columns (may not work perfectly)
            available_cols = [c for c in model_input_cols if c in X_df_model.columns]
            if len(available_cols) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot predict {target_col}: input columns mismatch. Model expects {model_input_cols}, got {list(X_df.columns)}"
                )
            X_df_model = X_df_model[available_cols]
        
        topk_labels, topk_proba = catboost_topk(model, X_df_model, k=5)
        top5_labels = topk_labels[0][:5] if topk_labels else []
        top5_proba = topk_proba[0][:5] if topk_proba else []
        return {
            "target": target_col,
            "top1": top5_labels[0] if top5_labels else None,
            "top3": top5_labels[:3] if top5_labels else [],
            "top5": top5_labels[:5] if top5_labels else [],
            "proba": top5_proba
        }
    
    # Handle TabTransformer
    if kind == "TabTransformer":
        model = model_info["model"]
        extra = model_info.get("extra", {})
        x_enc = extra.get("x_enc", {})
        y_enc = extra.get("y_enc")
        
        if y_enc is None:
            raise HTTPException(status_code=500, detail="TabTransformer model missing y_enc")
        
        topk_labels, topk_proba = tabtransformer_topk(model, x_enc, y_enc, X_df_model, model_input_cols, k=5)
        top5_labels = topk_labels[0][:5] if topk_labels else []
        top5_proba = topk_proba[0][:5] if topk_proba else []
        
        return {
            "target": target_col,
            "top1": top5_labels[0] if top5_labels else None,
            "top3": top5_labels[:3] if top5_labels else [],
            "top5": top5_labels[:5] if top5_labels else [],
            "proba": top5_proba
        }
    
    raise HTTPException(status_code=500, detail=f"Unsupported model kind: {kind}")


# Load bundle on startup
try:
    _load_bundle()
except Exception as exc:
    print(f"[ERROR] Failed to load bundle: {exc}")
    import traceback
    traceback.print_exc()
    BUNDLE = None


@app.get("/health")
def health():
    """Health check endpoint."""
    if BUNDLE is None:
        return {
            "status": "error",
            "bundle_loaded": False,
            "models_loaded": 0,
            "available_targets": []
        }
    return {
        "status": "ok",
        "bundle_loaded": True,
        "models_loaded": len(BUNDLE.best_pack),
        "available_targets": list(BUNDLE.best_pack.keys())
    }


@app.get("/targets")
def list_targets():
    """List all available target columns."""
    if BUNDLE is None:
        raise HTTPException(status_code=500, detail="Bundle not loaded")
    
    targets_info = []
    for target in BUNDLE.target_cols:
        if target in BUNDLE.best_pack:
            model_info = BUNDLE.best_pack[target]
            targets_info.append({
                "target": target,
                "kind": model_info.get("kind", "UNKNOWN"),
                "nunique_train": model_info.get("nunique_train", "N/A")
            })
    
    return {"targets": targets_info}


@app.post("/predict/all")
def predict_all(req: PredictRequest):
    """Predict for all target columns."""
    if BUNDLE is None:
        raise HTTPException(status_code=500, detail="Bundle not loaded")
    
    # Validate input first
    _validate_input(req)
    
    # Build and preprocess input
    X_df = _build_input_df(req)
    X_processed = _preprocess_input(X_df)
    
    # Predict for all targets
    results = {}
    for target_col in BUNDLE.target_cols:
        try:
            results[target_col] = _predict_target(target_col, X_processed)
        except Exception as e:
            results[target_col] = {
                "target": target_col,
                "error": str(e)
            }
    
    return {"predictions": results}


@app.post("/predict/{target_col}")
def predict_target_specific(target_col: str, req: PredictRequest):
    """Predict for a specific target column."""
    if BUNDLE is None:
        raise HTTPException(status_code=500, detail="Bundle not loaded")
    
    # Validate input first
    _validate_input(req)
    
    if target_col not in BUNDLE.target_cols:
        raise HTTPException(status_code=404, detail=f"Target '{target_col}' not in available targets")
    
    # Build and preprocess input
    X_df = _build_input_df(req)
    X_processed = _preprocess_input(X_df)
    
    # Predict
    result = _predict_target(target_col, X_processed)
    return result


@app.post("/autofill/dynamic")
def predict_dynamic(
    input_data: Dict[str, Any] = Body(
        ...,
        example={
            "CATEGORYID": "0110D00_MC"
        },
        description="Dynamic input fields. Can be any combination of fields from ALL_COLS. Example: {\"CATEGORYID\": \"0110D00_MC\"} or {\"CATEGORYID\": \"0110D00_MC\", \"U_TT\": \"TBI_CT_MC_U_TT_02\"}"
    ),
    rows: int = Query(5, ge=1, le=20, description="Number of result rows to return (1-20, default: 5)")
):
    """
    Predict all missing fields based on dynamic input.
    
    Accepts any fields as input and automatically predicts the remaining fields.
    
    Args:
        input_data: Dictionary with input fields
        rows: Number of result rows to return (default: 5, max: 20)
    
    Example request:
    {
        "CATEGORYID": "0110D00_MC"
    }
    
    Or with multiple fields:
    {
        "CATEGORYID": "0110D00_MC",
        "U_TT": "TBI_CT_MC_U_TT_02"
    }
    
    This will predict all other fields in ALL_COLS that are not in the input.
    Returns 'rows' number of combinations sorted by confidence (descending).
    """
    if BUNDLE is None:
        raise HTTPException(status_code=500, detail="Bundle not loaded")
    
    # Validate rows parameter
    if rows < 1:
        raise HTTPException(status_code=400, detail="rows must be at least 1")
    if rows > 20:
        raise HTTPException(status_code=400, detail="rows cannot exceed 20")
    
    # input_data is already a dict from Body(...)
    
    # Validate input fields
    if not input_data:
        raise HTTPException(status_code=400, detail="Input data cannot be empty")
    
    # Normalize input keys (uppercase)
    input_dict = {k.upper(): str(v).strip() if v is not None else UNK for k, v in input_data.items()}
    
    # Validate that all input fields are valid columns
    invalid_cols = [k for k in input_dict.keys() if k not in BUNDLE.all_cols]
    if invalid_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input columns: {invalid_cols}. Available columns: {BUNDLE.all_cols}"
        )
    
    # Validate NATIONALFACT if present
    if "NATIONALFACT" in input_dict:
        nationalfact = input_dict["NATIONALFACT"].strip()
        if nationalfact == INVALID_NATIONALFACT:
            raise HTTPException(
                status_code=400,
                detail="Không có quốc gia Việt Nam hãy nhập lại nhãn này"
            )
    
    # Determine input columns and target columns
    input_cols = list(input_dict.keys())
    target_cols = [c for c in BUNDLE.all_cols if c not in input_cols]
    
    if not target_cols:
        raise HTTPException(
            status_code=400,
            detail=f"All fields are provided. No fields to predict. Available fields: {BUNDLE.all_cols}"
        )
    
    # Build input dataframe
    input_df = pd.DataFrame([input_dict])
    
    # Preprocess input
    full_df = pd.DataFrame(index=input_df.index)
    for col in BUNDLE.all_cols:
        if col in input_df.columns:
            full_df[col] = input_df[col]
        else:
            full_df[col] = np.nan
    
    # Apply imputer
    imputed = BUNDLE.imputer.transform(full_df)
    
    # Extract input columns for prediction
    X_processed = imputed[input_cols].astype(str).fillna(UNK)
    
    # Predict for all target columns
    predictions = {}
    predicted_values = {}
    
    for target_col in target_cols:
        try:
            # Check auto-fill rules first
            if target_col in AUTO_FILL_RULES:
                fill_value = AUTO_FILL_RULES[target_col]
                predictions[target_col] = {
                    "target": target_col,
                    "top1": fill_value,
                    "top3": [fill_value] * 3,
                    "top5": [fill_value] * 5,
                    "proba": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "source": "auto_fill_rule"
                }
                predicted_values[target_col] = fill_value
                continue
            
            if target_col not in BUNDLE.best_pack:
                # Fallback: use training stats mode if available (align with notebook behavior)
                stats_info = (BUNDLE.stats or {}).get(target_col, {})
                mode_val = stats_info.get("mode_train")
                if mode_val is not None:
                    predictions[target_col] = {
                        "target": target_col,
                        "top1": str(mode_val),
                        "top3": [str(mode_val)] * 3,
                        "top5": [str(mode_val)] * 5,
                        "proba": [1.0, 0.0, 0.0, 0.0, 0.0],
                        "source": "stats_mode_fallback"
                    }
                    predicted_values[target_col] = str(mode_val)
                    continue
                else:
                    predictions[target_col] = {
                        "target": target_col,
                        "error": "Model not found"
                    }
                    continue
            
            model_info = BUNDLE.best_pack[target_col]
            kind = model_info.get("kind", "UNKNOWN")
            
            # Handle AUTO_FILL case
            if kind == "AUTO_FILL":
                fill_value = model_info.get("fill_value", UNK)
                predictions[target_col] = {
                    "target": target_col,
                    "top1": str(fill_value),
                    "top3": [str(fill_value)] * 3,
                    "top5": [str(fill_value)] * 5,
                    "proba": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "source": "auto_fill"
                }
                predicted_values[target_col] = str(fill_value)
                continue
            
            # Get model's expected input columns
            model_input_cols = model_info.get("input_cols", BUNDLE.input_cols)
            
            # Prepare X_df with model's expected columns
            X_df_model = pd.DataFrame(index=X_processed.index)
            for col in model_input_cols:
                if col in X_processed.columns:
                    X_df_model[col] = X_processed[col]
                else:
                    X_df_model[col] = UNK
            
            # Handle TransitionTable
            if kind == "TransitionTable":
                model = model_info["model"]
                topk = model.predict_topk(X_df_model, k=5)
                proba = model.predict_proba(X_df_model)[0]
                top5_idx = np.argsort(-proba)[:5]
                top5_proba = proba[top5_idx].tolist()
                top5_labels = topk[0][:5] if topk else []
                
                predictions[target_col] = {
                    "target": target_col,
                    "top1": top5_labels[0] if top5_labels else None,
                    "top3": top5_labels[:3] if top5_labels else [],
                    "top5": top5_labels[:5] if top5_labels else [],
                    "proba": top5_proba,
                    "source": "TransitionTable"
                }
                predicted_values[target_col] = top5_labels[0] if top5_labels else None
                continue
            
            # Handle CatBoost
            if kind == "CatBoost":
                model = model_info["model"]
                # Check if we have all required columns
                if set(model_input_cols) != set(X_df_model.columns):
                    available_cols = [c for c in model_input_cols if c in X_df_model.columns]
                    if len(available_cols) == 0:
                        predictions[target_col] = {
                            "target": target_col,
                            "error": f"Input columns mismatch. Model expects {model_input_cols}, got {list(X_processed.columns)}"
                        }
                        continue
                    X_df_model = X_df_model[available_cols]
                
                # Use helper to get string classes and top-k labels with probabilities
                topk_labels, topk_proba = catboost_topk(model, X_df_model, k=5)
                top5_labels = topk_labels[0][:5] if topk_labels else []
                top5_proba = topk_proba[0][:5] if topk_proba else []
                
                predictions[target_col] = {
                    "target": target_col,
                    "top1": top5_labels[0] if top5_labels else None,
                    "top3": top5_labels[:3] if top5_labels else [],
                    "top5": top5_labels[:5] if top5_labels else [],
                    "proba": top5_proba,
                    "source": "CatBoost"
                }
                predicted_values[target_col] = top5_labels[0] if top5_labels else None
                continue
            
            # Handle TabTransformer
            if kind == "TabTransformer":
                model = model_info["model"]
                extra = model_info.get("extra", {})
                x_enc = extra.get("x_enc", {})
                y_enc = extra.get("y_enc")
                
                if y_enc is None:
                    predictions[target_col] = {
                        "target": target_col,
                        "error": "TabTransformer model missing y_enc"
                    }
                    continue
                
                # Use helper to get top-k labels and probabilities
                topk_labels, topk_proba = tabtransformer_topk(model, x_enc, y_enc, X_df_model, model_input_cols, k=5)
                top5_labels = topk_labels[0][:5] if topk_labels else []
                top5_proba = topk_proba[0][:5] if topk_proba else []
                
                predictions[target_col] = {
                    "target": target_col,
                    "top1": top5_labels[0] if top5_labels else None,
                    "top3": top5_labels[:3] if top5_labels else [],
                    "top5": top5_labels[:5] if top5_labels else [],
                    "proba": top5_proba,
                    "source": "TabTransformer"
                }
                predicted_values[target_col] = top5_labels[0] if top5_labels else None
                continue
            
            predictions[target_col] = {
                "target": target_col,
                "error": f"Unsupported model kind: {kind}"
            }
            
        except Exception as e:
            predictions[target_col] = {
                "target": target_col,
                "error": str(e)
            }
    
    # Generate top-n combinations sorted by confidence
    topn_combinations = _generate_topn_combinations(predictions, target_cols, n=rows)
    
    # If no combinations generated, fallback to single result with top1
    if not topn_combinations:
        result_object = {}
        for target_col in target_cols:
            if target_col in predictions:
                pred = predictions[target_col]
                if "error" not in pred and "top1" in pred:
                    result_object[target_col] = pred["top1"]
        topn_combinations = [result_object] if result_object else []
    
    # Return top-n combinations (already sorted by confidence descending)
    return topn_combinations


@app.post("/autofill/dynamic/accuracy")
def predict_dynamic_with_accuracy(
    input_data: Dict[str, Any] = Body(
        ...,
        example={
            "CATEGORYID": "0110D00_MC"
        },
        description="Dynamic input fields. Can be any combination of fields from ALL_COLS. Example: {\"CATEGORYID\": \"0110D00_MC\"}"
    )
):
    """
    Predict all missing fields with accuracy/confidence scores for each field.
    
    This endpoint returns detailed accuracy information for each predicted field,
    including confidence scores (probabilities) for top predictions.
    
    Args:
        input_data: Dictionary with input fields
    
    Returns:
        Dictionary containing:
        - input_fields: List of input field names
        - predicted_fields: List of predicted field names
        - accuracy_details: Dictionary with accuracy info for each predicted field
            - field_name: {
                - top1: Top prediction value
                - top1_confidence: Confidence score (0-1) for top1
                - top3: List of top 3 predictions with confidence scores
                - top5: List of top 5 predictions with confidence scores
                - model_source: Source of prediction (TransitionTable, CatBoost, TabTransformer, etc.)
                - overall_confidence: Overall confidence score (top1 probability)
            }
    
    Example response:
    {
        "input_fields": ["CATEGORYID"],
        "predicted_fields": ["P_MANUFACTURERID", "DATEMANUFACTURE", ...],
        "accuracy_details": {
            "P_MANUFACTURERID": {
                "top1": "HSX.00311",
                "top1_confidence": 0.95,
                "top3": [
                    {"value": "HSX.00311", "confidence": 0.95},
                    {"value": "HSX.00312", "confidence": 0.03},
                    {"value": "HSX.00313", "confidence": 0.02}
                ],
                "top5": [...],
                "model_source": "CatBoost",
                "overall_confidence": 0.95
            },
            ...
        }
    }
    """
    if BUNDLE is None:
        raise HTTPException(status_code=500, detail="Bundle not loaded")
    
    # Validate input fields
    if not input_data:
        raise HTTPException(status_code=400, detail="Input data cannot be empty")
    
    # Normalize input keys (uppercase)
    input_dict = {k.upper(): str(v).strip() if v is not None else UNK for k, v in input_data.items()}
    
    # Validate that all input fields are valid columns
    invalid_cols = [k for k in input_dict.keys() if k not in BUNDLE.all_cols]
    if invalid_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input columns: {invalid_cols}. Available columns: {BUNDLE.all_cols}"
        )
    
    # Validate NATIONALFACT if present
    if "NATIONALFACT" in input_dict:
        nationalfact = input_dict["NATIONALFACT"].strip()
        if nationalfact == INVALID_NATIONALFACT:
            raise HTTPException(
                status_code=400,
                detail="Không có quốc gia Việt Nam hãy nhập lại nhãn này"
            )
    
    # Determine input columns and target columns
    input_cols = list(input_dict.keys())
    target_cols = [c for c in BUNDLE.all_cols if c not in input_cols]
    
    if not target_cols:
        raise HTTPException(
            status_code=400,
            detail=f"All fields are provided. No fields to predict. Available fields: {BUNDLE.all_cols}"
        )
    
    # Build input dataframe
    input_df = pd.DataFrame([input_dict])
    
    # Preprocess input
    full_df = pd.DataFrame(index=input_df.index)
    for col in BUNDLE.all_cols:
        if col in input_df.columns:
            full_df[col] = input_df[col]
        else:
            full_df[col] = np.nan
    
    # Apply imputer
    imputed = BUNDLE.imputer.transform(full_df)
    
    # Extract input columns for prediction
    X_processed = imputed[input_cols].astype(str).fillna(UNK)
    
    # Predict for all target columns (same logic as predict_dynamic)
    predictions = {}
    
    for target_col in target_cols:
        try:
            # Check auto-fill rules first
            if target_col in AUTO_FILL_RULES:
                fill_value = AUTO_FILL_RULES[target_col]
                predictions[target_col] = {
                    "target": target_col,
                    "top1": fill_value,
                    "top3": [fill_value] * 3,
                    "top5": [fill_value] * 5,
                    "proba": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "source": "auto_fill_rule"
                }
                continue
            
            if target_col not in BUNDLE.best_pack:
                # Fallback: use training stats mode if available
                stats_info = (BUNDLE.stats or {}).get(target_col, {})
                mode_val = stats_info.get("mode_train")
                if mode_val is not None:
                    predictions[target_col] = {
                        "target": target_col,
                        "top1": str(mode_val),
                        "top3": [str(mode_val)] * 3,
                        "top5": [str(mode_val)] * 5,
                        "proba": [1.0, 0.0, 0.0, 0.0, 0.0],
                        "source": "stats_mode_fallback"
                    }
                    continue
                else:
                    predictions[target_col] = {
                        "target": target_col,
                        "error": "Model not found"
                    }
                    continue
            
            model_info = BUNDLE.best_pack[target_col]
            kind = model_info.get("kind", "UNKNOWN")
            
            # Handle AUTO_FILL case
            if kind == "AUTO_FILL":
                fill_value = model_info.get("fill_value", UNK)
                predictions[target_col] = {
                    "target": target_col,
                    "top1": str(fill_value),
                    "top3": [str(fill_value)] * 3,
                    "top5": [str(fill_value)] * 5,
                    "proba": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "source": "auto_fill"
                }
                continue
            
            # Get model's expected input columns
            model_input_cols = model_info.get("input_cols", BUNDLE.input_cols)
            
            # Prepare X_df with model's expected columns
            X_df_model = pd.DataFrame(index=X_processed.index)
            for col in model_input_cols:
                if col in X_processed.columns:
                    X_df_model[col] = X_processed[col]
                else:
                    X_df_model[col] = UNK
            
            # Handle TransitionTable
            if kind == "TransitionTable":
                model = model_info["model"]
                topk = model.predict_topk(X_df_model, k=5)
                proba = model.predict_proba(X_df_model)[0]
                top5_idx = np.argsort(-proba)[:5]
                top5_proba = proba[top5_idx].tolist()
                top5_labels = topk[0][:5] if topk else []
                
                predictions[target_col] = {
                    "target": target_col,
                    "top1": top5_labels[0] if top5_labels else None,
                    "top3": top5_labels[:3] if top5_labels else [],
                    "top5": top5_labels[:5] if top5_labels else [],
                    "proba": top5_proba,
                    "source": "TransitionTable"
                }
                continue
            
            # Handle CatBoost
            if kind == "CatBoost":
                model = model_info["model"]
                if set(model_input_cols) != set(X_df_model.columns):
                    available_cols = [c for c in model_input_cols if c in X_df_model.columns]
                    if len(available_cols) == 0:
                        predictions[target_col] = {
                            "target": target_col,
                            "error": f"Input columns mismatch. Model expects {model_input_cols}, got {list(X_processed.columns)}"
                        }
                        continue
                
                labels, probas = catboost_topk(model, X_df_model[model_input_cols], k=5)
                
                predictions[target_col] = {
                    "target": target_col,
                    "top1": labels[0] if labels else None,
                    "top3": labels[:3] if labels else [],
                    "top5": labels[:5] if labels else [],
                    "proba": probas[:5] if probas else [],
                    "source": "CatBoost"
                }
                continue
            
            # Handle TabTransformer
            if kind == "TabTransformer":
                model = model_info["model"]
                label_encoder = model_info.get("label_encoder")
                cardinals = model_info.get("cardinals", [])
                
                if set(model_input_cols) != set(X_df_model.columns):
                    available_cols = [c for c in model_input_cols if c in X_df_model.columns]
                    if len(available_cols) == 0:
                        predictions[target_col] = {
                            "target": target_col,
                            "error": f"Input columns mismatch. Model expects {model_input_cols}, got {list(X_processed.columns)}"
                        }
                        continue
                
                labels, probas = tabtransformer_topk(model, label_encoder, X_df_model[model_input_cols], cardinals, k=5)
                
                predictions[target_col] = {
                    "target": target_col,
                    "top1": labels[0] if labels else None,
                    "top3": labels[:3] if labels else [],
                    "top5": labels[:5] if labels else [],
                    "proba": probas[:5] if probas else [],
                    "source": "TabTransformer"
                }
                continue
            
            # Unknown model type
            predictions[target_col] = {
                "target": target_col,
                "error": f"Unknown model kind: {kind}"
            }
            
        except Exception as e:
            predictions[target_col] = {
                "target": target_col,
                "error": str(e)
            }
    
    # Build accuracy details response
    accuracy_details = {}
    
    for target_col in target_cols:
        if target_col not in predictions:
            continue
        
        pred = predictions[target_col]
        
        if "error" in pred:
            accuracy_details[target_col] = {
                "top1": None,
                "top1_confidence": 0.0,
                "top3": [],
                "top5": [],
                "model_source": pred.get("source", "unknown"),
                "overall_confidence": 0.0,
                "error": pred["error"]
            }
            continue
        
        # Extract top predictions and probabilities
        top1 = pred.get("top1")
        top1_conf = pred.get("proba", [0.0])[0] if pred.get("proba") else 0.0
        
        top3_values = pred.get("top3", [])
        top3_probas = pred.get("proba", [])[:3] if pred.get("proba") else [0.0] * len(top3_values)
        top3_list = [
            {"value": val, "confidence": proba}
            for val, proba in zip(top3_values[:3], top3_probas[:3])
        ]
        
        top5_values = pred.get("top5", [])
        top5_probas = pred.get("proba", [])[:5] if pred.get("proba") else [0.0] * len(top5_values)
        top5_list = [
            {"value": val, "confidence": proba}
            for val, proba in zip(top5_values[:5], top5_probas[:5])
        ]
        
        accuracy_details[target_col] = {
            "top1": top1,
            "top1_confidence": round(float(top1_conf), 4),
            "top3": top3_list,
            "top5": top5_list,
            "model_source": pred.get("source", "unknown"),
            "overall_confidence": round(float(top1_conf), 4)
        }
    
    return {
        "input_fields": input_cols,
        "predicted_fields": target_cols,
        "accuracy_details": accuracy_details
    }
