"""
PMIS Configuration Settings

Cấu hình cho hệ thống PMIS Device Suggestion
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Directory paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, 'notebooks')

# File paths
RAW_DATA_FILE = os.path.join(DATA_DIR, 'devicesPMISMayCat.csv')
CLEANED_DATA_FILE = os.path.join(DATA_DIR, 'devicesPMISMayCat_cleaned.csv')


# ==============================================================================
# NORMALIZATION RULES (BẮT BUỘC)
# ==============================================================================

NORMALIZATION_RULES: Dict[str, str] = {
    'PHA': 'EVN.PHA_3P',                          # Pha = 3 pha ABC
    'KIEU_MC': 'TBI_CT_MC_KIEU_MC_01',            # Kiểu máy cắt = AIS
    'KIEU_DAPHQ': 'TBI_TT_MC_KIEU_DAPHQ.00001',   # Kiểu đập hồ quang = SF6
    'KIEU_CD': 'TBI_CT_MC_CC_CD.00001',           # Kiểu cơ cấu đóng = Lò xo
    'U_TT': 'TBI_CT_MC_U_TT_02',                  # Điện áp thao tác = 110VDC
}

# Giá trị bị cấm
FORBIDDEN_NATIONALFACT = 'TB040.00023'


# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================

@dataclass
class DataConfig:
    """Cấu hình xử lý dữ liệu"""
    
    # CSV settings
    delimiter: str = ';'
    encoding: str = 'utf-8'
    
    # Column lists
    id_columns: tuple = (
        'P_MANUFACTURERID', 'DATEMANUFACTURE', 'NATIONALFACT', 'OWNER',
        'LOAI', 'U_TT', 'KIEU_DAPHQ', 'I_DM', 'U_DM', 'KIEU_CD',
        'TG_CATNM', 'PHA', 'KIEU_MC', 'KNCDNMDM', 'CT_DC'
    )
    
    desc_columns: tuple = (
        'ASSETDESC', 'P_MANUFACTURERID_DESC', 'FIELDDESC', 'OWNER_DESC',
        'LOAI_DESC', 'KIEU_MC_DESC', 'KIEU_DAPHQ_DESC', 'I_DM_DESC',
        'U_DM_DESC', 'KIEU_CD_DESC', 'TG_CATNM_DESC', 'PHA_DESC',
        'KNCDNMDM_DESC', 'CT_DC_DESC'
    )
    
    critical_columns: tuple = ('ASSETID', 'ASSETDESC', 'CATEGORYID')
    
    # Technical columns (không nên điền giá trị mặc định)
    technical_columns: tuple = (
        'LOAI', 'U_TT', 'KIEU_DAPHQ', 'I_DM', 'U_DM',
        'KIEU_CD', 'TG_CATNM', 'PHA', 'KIEU_MC', 'KNCDNMDM', 'CT_DC'
    )
    
    # Fillable columns (có thể điền giá trị mặc định)
    fillable_columns: tuple = ('OWNER', 'CATEGORYID')


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

@dataclass
class ModelConfig:
    """Cấu hình model ML"""
    
    # Train/val/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    
    # TF-IDF settings
    tfidf_max_features: int = 100
    tfidf_ngram_range: tuple = (1, 2)
    
    # Random Forest settings
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    
    # Gradient Boosting settings
    gb_n_estimators: int = 100
    gb_max_depth: int = 5
    
    # KNN settings
    knn_n_neighbors: int = 5
    knn_metric: str = 'cosine'
    
    # Isolation Forest (anomaly detection)
    iso_n_estimators: int = 100
    iso_contamination: float = 0.05


# ==============================================================================
# API CONFIGURATION
# ==============================================================================

@dataclass
class APIConfig:
    """Cấu hình API"""
    
    # Server settings
    host: str = '0.0.0.0'
    port: int = 8000
    reload: bool = True
    log_level: str = 'info'
    
    # Response settings
    default_top_n: int = 5
    max_top_n: int = 10
    
    # Cors settings
    allow_origins: list = field(default_factory=lambda: ['*'])
    allow_methods: list = field(default_factory=lambda: ['*'])
    allow_headers: list = field(default_factory=lambda: ['*'])


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': os.path.join(LOG_DIR, 'pmis.log'),
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}


# ==============================================================================
# INSTANCES
# ==============================================================================

data_config = DataConfig()
model_config = ModelConfig()
api_config = APIConfig()
