"""
PMIS Device Suggestion API

API cung cấp các endpoint gợi ý thiết bị điện cho hệ thống PMIS.

Endpoints:
- POST /api/v1/suggest - Gợi ý thiết bị (không hiển thị score)
- POST /api/v1/suggest-with-score - Gợi ý thiết bị (có score)
- POST /api/v1/suggest-from-ocr - Gợi ý từ text OCR
- GET /api/v1/health - Health check

Author: PMIS Team
Version: 1.0.0
"""

import os
import sys
import re
import json
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from functools import wraps
import time

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# ML imports
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Create directories
os.makedirs(LOG_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'api.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_TOP_N = 5
MAX_TOP_N = 10

# Normalization rules (bắt buộc)
NORMALIZATION_RULES = {
    'PHA': 'EVN.PHA_3P',
    'KIEU_MC': 'TBI_CT_MC_KIEU_MC_01',
    'KIEU_DAPHQ': 'TBI_TT_MC_KIEU_DAPHQ.00001',
    'KIEU_CD': 'TBI_CT_MC_CC_CD.00001',
    'U_TT': 'TBI_CT_MC_U_TT_02',
}
FORBIDDEN_NATIONALFACT = 'TB040.00023'

# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================

class DeviceInput(BaseModel):
    """Input model cho API (không có _DESC)"""
    P_MANUFACTURERID: Optional[str] = Field(None, description="Mã nhà sản xuất")
    DATEMANUFACTURE: Optional[int] = Field(None, description="Năm sản xuất")
    NATIONALFACT: Optional[str] = Field(None, description="Mã quốc gia sản xuất")
    OWNER: Optional[str] = Field(None, description="Mã chủ sở hữu")
    LOAI: Optional[str] = Field(None, description="Mã loại")
    U_TT: Optional[str] = Field(None, description="Mã điện áp thao tác")
    KIEU_DAPHQ: Optional[str] = Field(None, description="Mã kiểu đập hồ quang")
    I_DM: Optional[str] = Field(None, description="Mã dòng điện định mức")
    U_DM: Optional[str] = Field(None, description="Mã điện áp định mức")
    KIEU_CD: Optional[str] = Field(None, description="Mã kiểu cơ cấu đóng")
    TG_CATNM: Optional[str] = Field(None, description="Mã thời gian cắt ngắn mạch")
    PHA: Optional[str] = Field(None, description="Mã pha")
    KIEU_MC: Optional[str] = Field(None, description="Mã kiểu máy cắt")
    KNCDNMDM: Optional[str] = Field(None, description="Mã khả năng cắt dòng ngắn mạch")
    CT_DC: Optional[str] = Field(None, description="Mã chu trình đóng cắt")
    top_n: Optional[int] = Field(DEFAULT_TOP_N, description="Số kết quả trả về", ge=1, le=MAX_TOP_N)


class OCRInput(BaseModel):
    """Input model cho API từ OCR (có _DESC)"""
    ASSETDESC: Optional[str] = Field(None, description="Mô tả thiết bị từ OCR")
    P_MANUFACTURERID_DESC: Optional[str] = Field(None, description="Tên nhà sản xuất từ OCR")
    FIELDDESC: Optional[str] = Field(None, description="Mô tả quốc gia từ OCR")
    OWNER_DESC: Optional[str] = Field(None, description="Tên chủ sở hữu từ OCR")
    LOAI_DESC: Optional[str] = Field(None, description="Mô tả loại từ OCR")
    KIEU_MC_DESC: Optional[str] = Field(None, description="Mô tả kiểu máy cắt từ OCR")
    KIEU_DAPHQ_DESC: Optional[str] = Field(None, description="Mô tả kiểu đập hồ quang từ OCR")
    top_n: Optional[int] = Field(DEFAULT_TOP_N, description="Số kết quả trả về", ge=1, le=MAX_TOP_N)


class DeviceSuggestion(BaseModel):
    """Model cho một gợi ý thiết bị - đầy đủ tất cả các trường"""
    ASSETID: str = Field(..., description="Mã tài sản")
    ASSETDESC: Optional[str] = Field(None, description="Mô tả thiết bị")
    CATEGORYID: Optional[str] = Field(None, description="Mã danh mục")
    P_MANUFACTURERID: Optional[str] = Field(None, description="Mã nhà sản xuất")
    P_MANUFACTURERID_DESC: Optional[str] = Field(None, description="Tên nhà sản xuất")
    DATEMANUFACTURE: Optional[int] = Field(None, description="Năm sản xuất")
    NATIONALFACT: Optional[str] = Field(None, description="Mã quốc gia")
    FIELDDESC: Optional[str] = Field(None, description="Tên quốc gia")
    OWNER: Optional[str] = Field(None, description="Mã chủ sở hữu")
    OWNER_DESC: Optional[str] = Field(None, description="Tên chủ sở hữu")
    LOAI: Optional[str] = Field(None, description="Mã loại máy cắt")
    LOAI_DESC: Optional[str] = Field(None, description="Tên loại máy cắt")
    U_TT: Optional[str] = Field(None, description="Mã điện áp thao tác")
    U_TT_DESC: Optional[str] = Field(None, description="Điện áp thao tác")
    KIEU_DAPHQ: Optional[str] = Field(None, description="Mã kiểu đập hồ quang")
    KIEU_DAPHQ_DESC: Optional[str] = Field(None, description="Kiểu đập hồ quang")
    I_DM: Optional[str] = Field(None, description="Mã dòng điện định mức")
    I_DM_DESC: Optional[str] = Field(None, description="Dòng điện định mức")
    U_DM: Optional[str] = Field(None, description="Mã điện áp định mức")
    U_DM_DESC: Optional[str] = Field(None, description="Điện áp định mức")
    KIEU_CD: Optional[str] = Field(None, description="Mã kiểu cơ cấu đóng")
    KIEU_CD_DESC: Optional[str] = Field(None, description="Kiểu cơ cấu đóng")
    TG_CATNM: Optional[str] = Field(None, description="Mã thời gian cắt ngắn mạch")
    TG_CATNM_DESC: Optional[str] = Field(None, description="Thời gian cắt ngắn mạch")
    PHA: Optional[str] = Field(None, description="Mã pha")
    PHA_DESC: Optional[str] = Field(None, description="Số pha")
    KIEU_MC: Optional[str] = Field(None, description="Mã kiểu máy cắt")
    KIEU_MC_DESC: Optional[str] = Field(None, description="Kiểu máy cắt")
    KNCDNMDM: Optional[str] = Field(None, description="Mã khả năng cắt dòng ngắn mạch")
    KNCDNMDM_DESC: Optional[str] = Field(None, description="Khả năng cắt dòng ngắn mạch")
    CT_DC: Optional[str] = Field(None, description="Mã chu trình đóng cắt")
    CT_DC_DESC: Optional[str] = Field(None, description="Chu trình đóng cắt")


class DeviceSuggestionWithScore(DeviceSuggestion):
    """Model cho gợi ý có score"""
    confidence_score: float = Field(..., description="Độ tin cậy (0-1)")


class SuggestResponse(BaseModel):
    """Response cho API không có score"""
    success: bool = True
    suggestions: List[DeviceSuggestion]
    total: int
    request_id: str
    processing_time_ms: float


class SuggestWithScoreResponse(BaseModel):
    """Response cho API có score"""
    success: bool = True
    suggestions: List[DeviceSuggestionWithScore]
    total: int
    request_id: str
    processing_time_ms: float


class ThuocTinhThietBi(BaseModel):
    """Thuộc tính thiết bị (dùng trong gợi ý)"""
    ASSETID: Optional[str] = None
    ASSETDESC: Optional[str] = None
    CATEGORYID: Optional[str] = None
    P_MANUFACTURERID: Optional[str] = None
    P_MANUFACTURERID_DESC: Optional[str] = None
    DATEMANUFACTURE: Optional[int] = None
    NATIONALFACT: Optional[str] = None
    FIELDDESC: Optional[str] = None
    OWNER: Optional[str] = None
    OWNER_DESC: Optional[str] = None
    LOAI: Optional[str] = None
    LOAI_DESC: Optional[str] = None
    U_TT: Optional[str] = None
    U_TT_DESC: Optional[str] = None
    KIEU_DAPHQ: Optional[str] = None
    KIEU_DAPHQ_DESC: Optional[str] = None
    I_DM: Optional[str] = None
    I_DM_DESC: Optional[str] = None
    U_DM: Optional[str] = None
    U_DM_DESC: Optional[str] = None
    KIEU_CD: Optional[str] = None
    KIEU_CD_DESC: Optional[str] = None
    TG_CATNM: Optional[str] = None
    TG_CATNM_DESC: Optional[str] = None
    PHA: Optional[str] = None
    PHA_DESC: Optional[str] = None
    KIEU_MC: Optional[str] = None
    KIEU_MC_DESC: Optional[str] = None
    KNCDNMDM: Optional[str] = None
    KNCDNMDM_DESC: Optional[str] = None
    CT_DC: Optional[str] = None
    CT_DC_DESC: Optional[str] = None


class GoiYThietBiItem(BaseModel):
    """Một phần tử trong danh sách gợi ý thiết bị"""
    thuTu: int = Field(..., description="Thứ tự (1, 2, 3, ...)")
    thuocTinh: ThuocTinhThietBi = Field(..., description="Thuộc tính thiết bị")
    doChinhXac: str = Field(..., description="Độ chính xác (%)")


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    error_code: str
    request_id: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    model_loaded: bool
    timestamp: str


# Multi-target artifact names (từ notebook 02_train_model.ipynb)
MULTI_TARGET_NAMES = [
    'loai_classifier', 'loai_encoder',
    'p_manufacturerid_classifier', 'p_manufacturerid_encoder',
    'feature_encoders'
]
MULTI_TARGET_CONFIG_PREFIX = 'multi_target_config_'


# ==============================================================================
# MODEL LOADER
# ==============================================================================

class ModelLoader:
    """Load và quản lý các model ML (hỗ trợ multi-target từ notebook 02_train_model)"""
    
    def __init__(self):
        self.models = {}           # target -> classifier (LOAI, P_MANUFACTURERID)
        self.target_encoders = {}  # target -> LabelEncoder
        self.label_encoders = None
        self.tfidf_vectorizers = None
        self.config = None
        self.device_data = None
        self.loaded = False
        self._feature_timestamp = None  # timestamp dùng để load cùng bộ artifacts
    
    def _find_latest_multi_target_timestamp(self) -> Optional[str]:
        """Tìm timestamp mới nhất có đủ bộ multi-target artifacts."""
        if not os.path.isdir(MODEL_DIR):
            return None
        files = os.listdir(MODEL_DIR)
        # Extract timestamps from filenames: *_YYYYMMDD_HHMMSS.pkl
        pattern = re.compile(r'_(\d{8}_\d{6})\.pkl$')
        timestamps = set()
        for f in files:
            m = pattern.search(f)
            if m:
                timestamps.add(m.group(1))
        for ts in sorted(timestamps, reverse=True):
            present = {f for f in files if ts in f}
            if all(any(prefix in f for f in present) for prefix in MULTI_TARGET_NAMES):
                return ts
        return None
    
    def load(self):
        """Load tất cả model artifacts (ưu tiên multi-target từ notebook)."""
        try:
            ts = self._find_latest_multi_target_timestamp()
            if ts:
                self._load_multi_target(ts)
            else:
                logger.warning("No multi-target artifacts found; trying legacy single model")
                self._load_legacy()
            
            # Load device data
            data_path = os.path.join(DATA_DIR, 'devicesPMISMayCat_cleaned.csv')
            if not os.path.exists(data_path):
                data_path = os.path.join(DATA_DIR, 'devicesPMISMayCat.csv')
            
            if os.path.exists(data_path):
                self.device_data = pd.read_csv(data_path, delimiter=';', encoding='utf-8')
                self.device_data = self.device_data.replace('NULL', np.nan)
                original_count = len(self.device_data)
                self.device_data = self.device_data[
                    self.device_data['CATEGORYID'].astype(str).str.startswith('0')
                ]
                filtered_count = original_count - len(self.device_data)
                if filtered_count > 0:
                    logger.warning(f"Filtered out {filtered_count} invalid rows from device data")
                logger.info(f"Loaded device data: {len(self.device_data)} records")
            
            self.loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            self.loaded = False
    
    def _load_multi_target(self, timestamp: str):
        """Load bộ artifacts multi-target (cùng timestamp)."""
        self._feature_timestamp = timestamp
        
        # Config
        config_name = f"{MULTI_TARGET_CONFIG_PREFIX}{timestamp}.json"
        config_path = os.path.join(CONFIG_DIR, config_name)
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
        
        # Feature encoders (label_encoders + tfidf_vectorizers)
        fe_path = os.path.join(MODEL_DIR, f'feature_encoders_{timestamp}.pkl')
        if os.path.exists(fe_path):
            with open(fe_path, 'rb') as f:
                data = pickle.load(f)
            self.label_encoders = data.get('label_encoders')
            self.tfidf_vectorizers = data.get('tfidf_vectorizers')
            logger.info("Loaded feature encoders (label + TF-IDF)")
        
        # Models và target encoders cho từng target
        target_map = [
            ('loai', 'LOAI'),
            ('p_manufacturerid', 'P_MANUFACTURERID'),
        ]
        for file_prefix, target_key in target_map:
            model_path = os.path.join(MODEL_DIR, f'{file_prefix}_classifier_{timestamp}.pkl')
            enc_path = os.path.join(MODEL_DIR, f'{file_prefix}_encoder_{timestamp}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[target_key] = pickle.load(f)
                logger.info(f"Loaded model for {target_key}")
            if os.path.exists(enc_path):
                with open(enc_path, 'rb') as f:
                    self.target_encoders[target_key] = pickle.load(f)
                logger.info(f"Loaded encoder for {target_key}")
    
    def _load_legacy(self):
        """Fallback: load theo tên file cũ (single target)."""
        config_path = os.path.join(CONFIG_DIR, 'model_config_latest.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
        
        model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('device_classifier') and f.endswith('.pkl')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            with open(os.path.join(MODEL_DIR, latest_model), 'rb') as f:
                self.models['LOAI'] = pickle.load(f)
            logger.info(f"Loaded legacy model from {latest_model}")
        
        encoder_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('label_encoders') and f.endswith('.pkl')]
        if encoder_files:
            latest = sorted(encoder_files)[-1]
            with open(os.path.join(MODEL_DIR, latest), 'rb') as f:
                self.label_encoders = pickle.load(f)
            logger.info("Loaded label encoders")
        
        tfidf_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('tfidf') and f.endswith('.pkl')]
        if tfidf_files:
            latest = sorted(tfidf_files)[-1]
            with open(os.path.join(MODEL_DIR, latest), 'rb') as f:
                self.tfidf_vectorizers = pickle.load(f)
            logger.info("Loaded TF-IDF vectorizers")
        
        target_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('target_encoder') and f.endswith('.pkl')]
        if target_files:
            latest = sorted(target_files)[-1]
            with open(os.path.join(MODEL_DIR, latest), 'rb') as f:
                self.target_encoders['LOAI'] = pickle.load(f)
            logger.info("Loaded target encoder")
    
    def is_loaded(self):
        return self.loaded
    
    def has_ml_models(self) -> bool:
        """True nếu có đủ models + encoders để predict (multi-target hoặc legacy)."""
        return bool(self.models and self.target_encoders and self.label_encoders is not None)


# Global model loader
model_loader = ModelLoader()


# ==============================================================================
# FEATURE BUILDING (đồng bộ với notebook create_features)
# ==============================================================================

def create_features_from_input(
    input_data: Dict[str, Any],
    label_encoders: Dict,
    tfidf_vectorizers: Dict,
    date_median: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Tạo vector đặc trưng từ một input (dict) để predict bằng model.
    Trả về X shape (1, n_features) hoặc None nếu thiếu encoders.
    """
    if not label_encoders or not tfidf_vectorizers:
        return None
    features_list = []
    # Categorical
    for col, le in label_encoders.items():
        val = input_data.get(col)
        if pd.isna(val) or val is None or val == '':
            val = '_MISSING_'
        val = str(val).strip()
        if val in le.classes_:
            encoded = le.transform([val])[0]
        else:
            encoded = -1
        features_list.append(np.array([[encoded]]))
    # Text (TF-IDF)
    for col, tfidf in tfidf_vectorizers.items():
        text = input_data.get(col)
        if pd.isna(text) or text is None:
            text = ''
        text = str(text).strip()
        tfidf_features = tfidf.transform([text]).toarray()
        features_list.append(tfidf_features)
    # Numeric DATEMANUFACTURE
    date_val = input_data.get('DATEMANUFACTURE')
    if date_val is None or (isinstance(date_val, float) and np.isnan(date_val)):
        date_val = date_median if date_median is not None else 0
    try:
        date_val = float(date_val)
    except (TypeError, ValueError):
        date_val = date_median if date_median is not None else 0
    features_list.append(np.array([[date_val]]))
    X = np.hstack(features_list)
    return X


# ==============================================================================
# TEXT NORMALIZER
# ==============================================================================

class TextNormalizer:
    """Chuẩn hóa text input từ OCR"""
    
    @staticmethod
    def normalize(text: str) -> str:
        """Chuẩn hóa một chuỗi text"""
        if not text or pd.isna(text):
            return ''
        
        text = str(text).strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Upper case for consistency
        text = text.upper()
        
        return text


text_normalizer = TextNormalizer()


# ==============================================================================
# SUGGESTION ENGINE
# ==============================================================================

class SuggestionEngine:
    """Engine gợi ý thiết bị"""
    
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
    
    def suggest_from_features(
        self, 
        input_data: Dict[str, Any], 
        top_n: int = DEFAULT_TOP_N,
        include_score: bool = False
    ) -> List[Dict]:
        """
        Gợi ý thiết bị dựa trên features (ID không _DESC).
        Nếu có ML models (multi-target): predict LOAI & P_MANUFACTURERID rồi rank theo match.
        Ngược lại: similarity matching theo từng trường.
        """
        if not self.model_loader.is_loaded() or self.model_loader.device_data is None:
            return self._fallback_suggest(input_data, top_n, include_score)
        
        df = self.model_loader.device_data.copy()
        loader = self.model_loader
        
        # Bước 1: Lọc trước theo input để đảm bảo kết quả đúng với yêu cầu
        df_filtered = df.copy()
        input_match_mask = np.ones(len(df), dtype=bool)
        for field, value in input_data.items():
            if value is not None and field in df.columns:
                match = (df[field].astype(str) == str(value))
                input_match_mask = input_match_mask & match
        
        # Nếu có ít nhất 1 trường input khớp, chỉ lấy các dòng khớp tất cả input
        if input_match_mask.sum() > 0:
            df_filtered = df[input_match_mask].copy()
            logger.debug(f"Filtered to {len(df_filtered)} rows matching all input fields")
        else:
            # Không có dòng nào khớp tất cả input: trả về empty hoặc nới lỏng filter
            logger.warning(f"No rows match all input fields, using original dataset")
            df_filtered = df.copy()
        
        if loader.has_ml_models():
            
            # Bước 2: Dùng ML để predict và tính điểm (chỉ trên df_filtered)
            date_median = None
            if loader.device_data is not None and 'DATEMANUFACTURE' in loader.device_data.columns:
                date_median = float(loader.device_data['DATEMANUFACTURE'].median())
            X = create_features_from_input(
                input_data, loader.label_encoders, loader.tfidf_vectorizers, date_median
            )
            if X is not None and loader.models and loader.target_encoders:
                predicted = {}
                for target_key, model in loader.models.items():
                    enc = loader.target_encoders.get(target_key)
                    if enc is not None:
                        try:
                            y_pred = model.predict(X)
                            pred_label = enc.inverse_transform(y_pred)[0]
                            predicted[target_key] = pred_label
                        except Exception:
                            pass
                
                # Score: ƯU TIÊN match với input người dùng (×20), sau đó match với predicted (×5)
                # Base score = 0 để có thể phân biệt rõ ràng
                scores = np.zeros(len(df_filtered))
                
                # Match với input: trọng số cao (×20) để ưu tiên
                input_match_count = 0
                for field, value in input_data.items():
                    if value is not None and field in df_filtered.columns:
                        match = (df_filtered[field].astype(str) == str(value))
                        scores += match.astype(float) * 20
                        input_match_count += 1
                
                # Match với predicted: trọng số thấp hơn (×5) để phân biệt ranking
                for field, value in predicted.items():
                    if field in df_filtered.columns:
                        match = (df_filtered[field].astype(str) == str(value))
                        scores += match.astype(float) * 5
                
                # Nếu tất cả scores bằng nhau (đều khớp input nhưng không có predicted khác biệt),
                # thêm điểm dựa trên thứ tự để phân biệt (giảm dần)
                if len(np.unique(scores)) == 1 and len(scores) > 1:
                    # Tất cả đều có cùng điểm → thêm điểm nhỏ giảm dần theo thứ tự
                    scores = scores + np.linspace(0.1, 0, len(scores))
                
                top_indices_filtered = np.argsort(scores)[::-1][:top_n]
                # Map lại indices về df gốc
                if len(df_filtered) < len(df):
                    top_indices = df_filtered.iloc[top_indices_filtered].index.tolist()
                else:
                    top_indices = top_indices_filtered.tolist()
            else:
                # Không có model: chỉ dùng input matching
                scores = np.ones(len(df_filtered))
                for field, value in input_data.items():
                    if value is not None and field in df_filtered.columns:
                        match = (df_filtered[field].astype(str) == str(value))
                        scores += match.astype(float) * 20
                top_indices_filtered = np.argsort(scores)[::-1][:top_n]
                if len(df_filtered) < len(df):
                    top_indices = df_filtered.iloc[top_indices_filtered].index.tolist()
                else:
                    top_indices = top_indices_filtered.tolist()
        else:
            # Fallback: similarity matching theo từng trường (chỉ trên df_filtered)
            scores = np.zeros(len(df_filtered))
            for field, value in input_data.items():
                if value is not None and field in df_filtered.columns:
                    match = (df_filtered[field].astype(str) == str(value))
                    scores += match.astype(float) * 20
            
            # Nếu tất cả scores bằng nhau, thêm điểm giảm dần
            if len(np.unique(scores)) == 1 and len(scores) > 1:
                scores = scores + np.linspace(0.1, 0, len(scores))
            
            top_indices_filtered = np.argsort(scores)[::-1][:top_n]
            if len(df_filtered) < len(df):
                top_indices = df_filtered.iloc[top_indices_filtered].index.tolist()
            else:
                top_indices = top_indices_filtered.tolist()
        
        results = []
        # Tính số trường input để tính doChinhXac
        input_fields_count = sum(1 for v in input_data.values() if v is not None)
        
        for idx in top_indices:
            row = df.iloc[idx]
            suggestion = self._build_suggestion_dict(row)
            if include_score:
                # Tính số trường khớp với input
                matched_fields = 0
                for field, value in input_data.items():
                    if value is not None and field in df.columns:
                        if str(row.get(field, '')) == str(value):
                            matched_fields += 1
                
                # Tính confidence_score: tỉ lệ trường khớp + điểm ranking
                if input_fields_count > 0:
                    match_ratio = matched_fields / input_fields_count
                else:
                    match_ratio = 1.0
                
                # Kết hợp: 50% từ match ratio, 50% từ ranking để phân biệt tốt hơn
                rank_score = 1.0
                if hasattr(scores, '__len__') and len(scores) > 0:
                    # Tìm score tương ứng với idx này
                    try:
                        if len(df_filtered) < len(df):
                            # idx là index trong df gốc, cần tìm trong df_filtered
                            filtered_idx = df_filtered.index.get_loc(idx)
                            score_val = scores[filtered_idx] if filtered_idx < len(scores) else 0
                        else:
                            score_val = scores[idx] if idx < len(scores) else 0
                        max_score = max(scores.max(), 1e-9)
                        min_score = scores.min()
                        # Normalize về [0, 1] với min=0 nếu có thể
                        if max_score > min_score:
                            rank_score = (score_val - min_score) / (max_score - min_score)
                        else:
                            rank_score = 1.0
                    except (KeyError, IndexError, TypeError):
                        rank_score = 1.0
                
                # Kết hợp: 50% match ratio + 50% rank score để phân biệt tốt hơn
                confidence = 0.5 * match_ratio + 0.5 * rank_score
                suggestion['confidence_score'] = round(confidence, 4)
            results.append(suggestion)
        return results
    
    def _build_suggestion_dict(self, row) -> Dict:
        """Tạo dict gợi ý với đầy đủ các trường"""
        def get_str(field):
            val = row.get(field)
            return str(val) if pd.notna(val) else None
        
        def get_int(field):
            val = row.get(field)
            return int(val) if pd.notna(val) else None
        
        return {
            'ASSETID': str(row.get('ASSETID', '')),
            'ASSETDESC': get_str('ASSETDESC'),
            'CATEGORYID': get_str('CATEGORYID'),
            'P_MANUFACTURERID': get_str('P_MANUFACTURERID'),
            'P_MANUFACTURERID_DESC': get_str('P_MANUFACTURERID_DESC'),
            'DATEMANUFACTURE': get_int('DATEMANUFACTURE'),
            'NATIONALFACT': get_str('NATIONALFACT'),
            'FIELDDESC': get_str('FIELDDESC'),
            'OWNER': get_str('OWNER'),
            'OWNER_DESC': get_str('OWNER_DESC'),
            'LOAI': get_str('LOAI'),
            'LOAI_DESC': get_str('LOAI_DESC'),
            'U_TT': get_str('U_TT'),
            'U_TT_DESC': get_str('U_TT_DESC'),
            'KIEU_DAPHQ': get_str('KIEU_DAPHQ'),
            'KIEU_DAPHQ_DESC': get_str('KIEU_DAPHQ_DESC'),
            'I_DM': get_str('I_DM'),
            'I_DM_DESC': get_str('I_DM_DESC'),
            'U_DM': get_str('U_DM'),
            'U_DM_DESC': get_str('U_DM_DESC'),
            'KIEU_CD': get_str('KIEU_CD'),
            'KIEU_CD_DESC': get_str('KIEU_CD_DESC'),
            'TG_CATNM': get_str('TG_CATNM'),
            'TG_CATNM_DESC': get_str('TG_CATNM_DESC'),
            'PHA': get_str('PHA'),
            'PHA_DESC': get_str('PHA_DESC'),
            'KIEU_MC': get_str('KIEU_MC'),
            'KIEU_MC_DESC': get_str('KIEU_MC_DESC'),
            'KNCDNMDM': get_str('KNCDNMDM'),
            'KNCDNMDM_DESC': get_str('KNCDNMDM_DESC'),
            'CT_DC': get_str('CT_DC'),
            'CT_DC_DESC': get_str('CT_DC_DESC'),
        }
    
    def suggest_from_ocr(
        self, 
        input_data: Dict[str, Any], 
        top_n: int = DEFAULT_TOP_N,
        include_score: bool = False
    ) -> List[Dict]:
        """
        Gợi ý thiết bị dựa trên text OCR (_DESC fields)
        """
        if not self.model_loader.is_loaded() or self.model_loader.device_data is None:
            return self._fallback_suggest(input_data, top_n, include_score)
        
        df = self.model_loader.device_data.copy()
        scores = np.zeros(len(df))
        
        # Normalize input text
        normalized_input = {}
        for field, value in input_data.items():
            if value is not None:
                normalized_input[field] = text_normalizer.normalize(value)
        
        # Text similarity matching
        for field, value in normalized_input.items():
            if value and field in df.columns:
                df_values = df[field].fillna('').astype(str).str.upper()
                
                # Exact match
                exact_match = df_values == value
                scores += exact_match.astype(float) * 10
                
                # Partial match (contains)
                partial_match = df_values.str.contains(value, case=False, na=False)
                scores += partial_match.astype(float) * 5
                
                # Partial match (input contains df value)
                for i, dv in enumerate(df_values):
                    if dv and value and dv in value:
                        scores[i] += 3
        
        # Get top N
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        results = []
        max_score = scores.max()
        
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                row = df.iloc[idx]
                suggestion = self._build_suggestion_dict(row)
                
                if include_score:
                    suggestion['confidence_score'] = round(scores[idx] / max_score, 4) if max_score > 0 else 0.0
                
                results.append(suggestion)
        
        return results
    
    def _fallback_suggest(
        self, 
        input_data: Dict[str, Any], 
        top_n: int,
        include_score: bool
    ) -> List[Dict]:
        """Fallback khi model chưa load"""
        return []


# Global suggestion engine
suggestion_engine = SuggestionEngine(model_loader)


# ==============================================================================
# FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="PMIS Device Suggestion API",
    description="API gợi ý thiết bị điện cho hệ thống PMIS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# MIDDLEWARE & UTILITIES
# ==============================================================================

def generate_request_id() -> str:
    """Generate unique request ID"""
    import uuid
    return str(uuid.uuid4())[:8]


def get_request_id() -> str:
    """Dependency để generate request_id"""
    return generate_request_id()


def build_goi_y_list(suggestions: List[Dict]) -> List[Dict]:
    """
    Chuyển danh sách suggestions sang format [{ thuTu, thuocTinh, doChinhXac }].
    Độ chính xác dựa trên confidence_score (đã tính từ số trường khớp + ranking).
    Nếu không có confidence_score thì giảm dần theo thứ hạng.
    """
    if not suggestions:
        return []
    
    result = []
    scores = [s.get('confidence_score') for s in suggestions]
    has_scores = all(x is not None for x in scores)
    
    if has_scores:
        # Dùng confidence_score để tính doChinhXac
        raw = [float(x) for x in scores]
        for i, s in enumerate(suggestions):
            thuoc_tinh = {k: v for k, v in s.items() if k != 'confidence_score'}
            pct = raw[i] * 100
            do_chinh_xac = "100%" if pct >= 99.995 else f"{pct:.2f}%"
            result.append({
                "thuTu": i + 1,
                "thuocTinh": thuoc_tinh,
                "doChinhXac": do_chinh_xac,
            })
    else:
        # Fallback: giảm dần theo thứ hạng
        n = len(suggestions)
        if n <= 1:
            pcts = [100.0]
        else:
            step = 20.0 / (n - 1)
            pcts = [100.0 - i * step for i in range(n)]
        for i, s in enumerate(suggestions):
            thuoc_tinh = {k: v for k, v in s.items() if k != 'confidence_score'}
            pct = pcts[i]
            do_chinh_xac = "100%" if pct >= 99.995 else f"{pct:.2f}%"
            result.append({
                "thuTu": i + 1,
                "thuocTinh": thuoc_tinh,
                "doChinhXac": do_chinh_xac,
            })
    return result


# ==============================================================================
# STARTUP EVENT
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting PMIS Device Suggestion API...")
    model_loader.load()
    logger.info("API ready!")


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model_loader.is_loaded(),
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/suggest", response_model=List[GoiYThietBiItem])
async def suggest_devices(input_data: DeviceInput, request_id: str = Depends(get_request_id)):
    """
    API A: Gợi ý thiết bị.
    
    Input: Các trường ID (không có _DESC)
    Output: Mảng [{ thuTu, thuocTinh, doChinhXac }]
    """
    start_time = time.time()
    logger.info(f"[{request_id}] Request: suggest_devices")
    
    try:
        if input_data.NATIONALFACT == FORBIDDEN_NATIONALFACT:
            raise HTTPException(
                status_code=400,
                detail=f"NATIONALFACT không được là {FORBIDDEN_NATIONALFACT}"
            )
        
        input_dict = {k: v for k, v in input_data.dict().items() if v is not None and k != 'top_n'}
        
        suggestions = suggestion_engine.suggest_from_features(
            input_dict,
            top_n=input_data.top_n or DEFAULT_TOP_N,
            include_score=True,
        )
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"[{request_id}] Response: success, {len(suggestions)} items, {processing_time:.2f}ms")
        
        return build_goi_y_list(suggestions)
    except HTTPException:
        raise
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Error: {str(e)}, {elapsed:.2f}ms")
        raise


@app.post("/api/v1/suggest-with-score", response_model=List[GoiYThietBiItem])
async def suggest_devices_with_score(input_data: DeviceInput, request_id: str = Depends(get_request_id)):
    """
    API B: Gợi ý thiết bị (có độ chính xác %).
    
    Input: Các trường ID (không có _DESC)
    Output: Mảng [{ thuTu, thuocTinh, doChinhXac }]
    """
    start_time = time.time()
    logger.info(f"[{request_id}] Request: suggest_devices_with_score")
    
    try:
        if input_data.NATIONALFACT == FORBIDDEN_NATIONALFACT:
            raise HTTPException(
                status_code=400,
                detail=f"NATIONALFACT không được là {FORBIDDEN_NATIONALFACT}"
            )
        
        input_dict = {k: v for k, v in input_data.dict().items() if v is not None and k != 'top_n'}
        
        suggestions = suggestion_engine.suggest_from_features(
            input_dict,
            top_n=input_data.top_n or DEFAULT_TOP_N,
            include_score=True,
        )
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"[{request_id}] Response: success, {len(suggestions)} items, {processing_time:.2f}ms")
        
        return build_goi_y_list(suggestions)
    except HTTPException:
        raise
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Error: {str(e)}, {elapsed:.2f}ms")
        raise


@app.post("/api/v1/suggest-from-ocr", response_model=List[GoiYThietBiItem])
async def suggest_from_ocr(input_data: OCRInput, request_id: str = Depends(get_request_id)):
    """
    API C: Gợi ý thiết bị từ OCR.
    
    Input: Các trường _DESC từ OCR
    Output: Mảng [{ thuTu, thuocTinh, doChinhXac }]
    """
    start_time = time.time()
    logger.info(f"[{request_id}] Request: suggest_from_ocr")
    
    try:
        input_dict = {k: v for k, v in input_data.dict().items() if v is not None and k != 'top_n'}
        
        suggestions = suggestion_engine.suggest_from_ocr(
            input_dict,
            top_n=input_data.top_n or DEFAULT_TOP_N,
            include_score=True,
        )
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"[{request_id}] Response: success, {len(suggestions)} items, {processing_time:.2f}ms")
        
        return build_goi_y_list(suggestions)
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Error: {str(e)}, {elapsed:.2f}ms")
        raise


# ==============================================================================
# ERROR HANDLERS
# ==============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "request_id": generate_request_id()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "request_id": generate_request_id()
        }
    )


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
