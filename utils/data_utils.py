"""
Data Utilities for PMIS

Các hàm tiện ích xử lý dữ liệu
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_device_data(
    file_path: str,
    delimiter: str = ';',
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    """
    Đọc file dữ liệu thiết bị
    
    Args:
        file_path: Đường dẫn file CSV
        delimiter: Ký tự phân cách
        encoding: Mã hóa file
        
    Returns:
        DataFrame chứa dữ liệu
    """
    df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
    
    # Replace 'NULL' string with NaN
    df = df.replace('NULL', np.nan)
    
    return df


def save_device_data(
    df: pd.DataFrame,
    file_path: str,
    delimiter: str = ';',
    encoding: str = 'utf-8'
) -> None:
    """
    Lưu dữ liệu thiết bị ra file CSV
    
    Args:
        df: DataFrame cần lưu
        file_path: Đường dẫn file đầu ra
        delimiter: Ký tự phân cách
        encoding: Mã hóa file
    """
    df.to_csv(file_path, sep=delimiter, index=False, encoding=encoding)


# ==============================================================================
# DATA CLEANING
# ==============================================================================

def clean_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cắt khoảng trắng thừa ở tất cả cột string
    
    Args:
        df: DataFrame cần xử lý
        
    Returns:
        DataFrame đã được làm sạch
    """
    df_cleaned = df.copy()
    
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].apply(
                lambda x: str(x).strip() if pd.notna(x) else x
            )
    
    return df_cleaned


def remove_duplicates(
    df: pd.DataFrame,
    subset: List[str] = None,
    keep_most_complete: bool = True
) -> pd.DataFrame:
    """
    Loại bỏ dòng trùng lặp, giữ lại bản ghi đầy đủ nhất
    
    Args:
        df: DataFrame cần xử lý
        subset: Danh sách cột để kiểm tra trùng lặp
        keep_most_complete: Nếu True, giữ bản ghi có nhiều giá trị nhất
        
    Returns:
        DataFrame không có dòng trùng lặp
    """
    df_cleaned = df.copy()
    
    if keep_most_complete:
        # Tính số cột không null cho mỗi dòng
        df_cleaned['_non_null_count'] = df_cleaned.notna().sum(axis=1)
        
        # Sắp xếp theo subset và số cột không null
        sort_cols = (subset if subset else [df_cleaned.columns[0]]) + ['_non_null_count']
        df_cleaned = df_cleaned.sort_values(sort_cols, ascending=[True] * len(subset or [df_cleaned.columns[0]]) + [False])
        
        # Giữ bản ghi đầu tiên
        df_cleaned = df_cleaned.drop_duplicates(subset=subset, keep='first')
        
        # Xóa cột tạm
        df_cleaned = df_cleaned.drop('_non_null_count', axis=1)
    else:
        df_cleaned = df_cleaned.drop_duplicates(subset=subset, keep='first')
    
    return df_cleaned


def normalize_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa kiểu dữ liệu các cột
    
    Args:
        df: DataFrame cần xử lý
        
    Returns:
        DataFrame với kiểu dữ liệu đã chuẩn hóa
    """
    df_normalized = df.copy()
    
    # DATEMANUFACTURE -> Int64 (nullable integer)
    if 'DATEMANUFACTURE' in df_normalized.columns:
        df_normalized['DATEMANUFACTURE'] = pd.to_numeric(
            df_normalized['DATEMANUFACTURE'], errors='coerce'
        ).astype('Int64')
    
    return df_normalized


# ==============================================================================
# DATA VALIDATION
# ==============================================================================

def validate_normalization_rules(
    df: pd.DataFrame,
    rules: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Kiểm tra dữ liệu theo các quy tắc chuẩn hóa
    
    Args:
        df: DataFrame cần kiểm tra
        rules: Dict các quy tắc {column: expected_value}
        
    Returns:
        Dict thống kê vi phạm cho mỗi cột
    """
    results = {}
    
    for col, expected_value in rules.items():
        if col in df.columns:
            mask_not_null = df[col].notna()
            mask_not_standard = mask_not_null & (df[col] != expected_value)
            
            non_standard_count = mask_not_standard.sum()
            non_standard_values = df[mask_not_standard][col].value_counts().head(10).to_dict()
            
            results[col] = {
                'expected_value': expected_value,
                'non_standard_count': non_standard_count,
                'non_standard_values': non_standard_values,
                'compliance_rate': 1 - (non_standard_count / len(df)) if len(df) > 0 else 1.0
            }
    
    return results


def check_forbidden_values(
    df: pd.DataFrame,
    forbidden: Dict[str, Any]
) -> Dict[str, int]:
    """
    Kiểm tra các giá trị bị cấm
    
    Args:
        df: DataFrame cần kiểm tra
        forbidden: Dict {column: forbidden_value}
        
    Returns:
        Dict số lượng vi phạm cho mỗi cột
    """
    results = {}
    
    for col, forbidden_value in forbidden.items():
        if col in df.columns:
            results[col] = (df[col] == forbidden_value).sum()
    
    return results


def get_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thống kê dữ liệu thiếu theo cột
    
    Args:
        df: DataFrame cần thống kê
        
    Returns:
        DataFrame thống kê
    """
    stats = []
    
    for col in df.columns:
        nan_count = df[col].isna().sum()
        stats.append({
            'column': col,
            'missing_count': nan_count,
            'missing_pct': round(nan_count / len(df) * 100, 2) if len(df) > 0 else 0,
            'dtype': str(df[col].dtype)
        })
    
    return pd.DataFrame(stats).sort_values('missing_count', ascending=False)


# ==============================================================================
# DATA TRANSFORMATION
# ==============================================================================

def fill_with_mode(
    df: pd.DataFrame,
    columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Điền giá trị thiếu bằng mode (giá trị phổ biến nhất)
    
    Args:
        df: DataFrame cần xử lý
        columns: Danh sách cột cần điền
        
    Returns:
        Tuple (DataFrame đã điền, Dict giá trị đã điền)
    """
    df_filled = df.copy()
    filled_values = {}
    
    for col in columns:
        if col in df_filled.columns:
            null_count = df_filled[col].isna().sum()
            if null_count > 0:
                mode = df_filled[col].mode()
                if len(mode) > 0:
                    fill_value = mode[0]
                    df_filled[col] = df_filled[col].fillna(fill_value)
                    filled_values[col] = {
                        'value': fill_value,
                        'count': null_count
                    }
    
    return df_filled, filled_values


def add_missing_flags(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """
    Thêm cột flag đánh dấu giá trị thiếu
    
    Args:
        df: DataFrame cần xử lý
        columns: Danh sách cột cần đánh dấu
        
    Returns:
        DataFrame với các cột flag mới
    """
    df_flagged = df.copy()
    
    for col in columns:
        if col in df_flagged.columns:
            flag_col = f'_FLAG_{col}_MISSING'
            df_flagged[flag_col] = df_flagged[col].isna()
    
    return df_flagged


# ==============================================================================
# TEXT PROCESSING
# ==============================================================================

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa text (cho OCR input)
    
    Args:
        text: Text cần chuẩn hóa
        
    Returns:
        Text đã chuẩn hóa
    """
    if pd.isna(text) or text is None:
        return ''
    
    text = str(text).strip()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Upper case for consistency
    text = text.upper()
    
    return text


def extract_year_from_text(text: str) -> Optional[int]:
    """
    Trích xuất năm từ text
    
    Args:
        text: Text chứa năm
        
    Returns:
        Năm (int) hoặc None
    """
    import re
    
    if pd.isna(text):
        return None
    
    # Tìm năm 4 chữ số (19xx hoặc 20xx)
    matches = re.findall(r'\b(19\d{2}|20\d{2})\b', str(text))
    
    if matches:
        # Lấy năm gần nhất với hiện tại
        current_year = datetime.now().year
        valid_years = [int(y) for y in matches if int(y) <= current_year + 1]
        if valid_years:
            return max(valid_years)
    
    return None


# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def generate_cleaning_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Tạo báo cáo so sánh trước/sau làm sạch
    
    Args:
        df_before: DataFrame trước khi làm sạch
        df_after: DataFrame sau khi làm sạch
        output_path: Đường dẫn file báo cáo (optional)
        
    Returns:
        Dict chứa thông tin báo cáo
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'before': {
            'total_rows': len(df_before),
            'total_columns': len(df_before.columns),
            'total_missing': df_before.isna().sum().sum()
        },
        'after': {
            'total_rows': len(df_after),
            'total_columns': len(df_after.columns),
            'total_missing': df_after.isna().sum().sum()
        },
        'changes': {
            'rows_removed': len(df_before) - len(df_after),
            'missing_values_changed': df_before.isna().sum().sum() - df_after.isna().sum().sum()
        }
    }
    
    if output_path:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report
