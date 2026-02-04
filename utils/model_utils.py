"""
Model Utilities for PMIS

Các hàm tiện ích cho model ML
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)


# ==============================================================================
# MODEL SAVING/LOADING
# ==============================================================================

def save_model(model: Any, file_path: str) -> None:
    """
    Lưu model vào file pickle
    
    Args:
        model: Model cần lưu
        file_path: Đường dẫn file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(file_path: str) -> Any:
    """
    Load model từ file pickle
    
    Args:
        file_path: Đường dẫn file
        
    Returns:
        Model đã load
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_config(config: Dict, file_path: str) -> None:
    """
    Lưu config vào file JSON
    
    Args:
        config: Dict config
        file_path: Đường dẫn file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_config(file_path: str) -> Dict:
    """
    Load config từ file JSON
    
    Args:
        file_path: Đường dẫn file
        
    Returns:
        Dict config
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def create_feature_matrix(
    df: pd.DataFrame,
    categorical_cols: List[str],
    text_cols: List[str],
    numeric_cols: List[str],
    label_encoders: Dict = None,
    tfidf_vectorizers: Dict = None
) -> Tuple[np.ndarray, List[str], Dict, Dict]:
    """
    Tạo feature matrix từ DataFrame
    
    Args:
        df: DataFrame đầu vào
        categorical_cols: Danh sách cột categorical
        text_cols: Danh sách cột text
        numeric_cols: Danh sách cột numeric
        label_encoders: Dict label encoders (nếu đã có)
        tfidf_vectorizers: Dict TF-IDF vectorizers (nếu đã có)
        
    Returns:
        Tuple (feature_matrix, feature_names, label_encoders, tfidf_vectorizers)
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    features_list = []
    feature_names = []
    
    # Initialize encoders if not provided
    if label_encoders is None:
        label_encoders = {}
    if tfidf_vectorizers is None:
        tfidf_vectorizers = {}
    
    # Categorical features
    for col in categorical_cols:
        if col in df.columns:
            if col not in label_encoders:
                le = LabelEncoder()
                all_values = df[col].fillna('_MISSING_').astype(str).unique()
                le.fit(all_values)
                label_encoders[col] = le
            
            le = label_encoders[col]
            values = df[col].fillna('_MISSING_').astype(str)
            
            # Handle unseen labels
            encoded = []
            for v in values:
                if v in le.classes_:
                    encoded.append(le.transform([v])[0])
                else:
                    encoded.append(-1)
            
            features_list.append(np.array(encoded).reshape(-1, 1))
            feature_names.append(col)
    
    # Text features (TF-IDF)
    for col in text_cols:
        if col in df.columns:
            if col not in tfidf_vectorizers:
                tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
                text_data = df[col].fillna('').astype(str)
                tfidf.fit(text_data)
                tfidf_vectorizers[col] = tfidf
            
            tfidf = tfidf_vectorizers[col]
            text_data = df[col].fillna('').astype(str)
            tfidf_features = tfidf.transform(text_data).toarray()
            
            features_list.append(tfidf_features)
            feature_names.extend([f"{col}_tfidf_{i}" for i in range(tfidf_features.shape[1])])
    
    # Numeric features
    for col in numeric_cols:
        if col in df.columns:
            values = df[col].fillna(df[col].median()).values.reshape(-1, 1)
            features_list.append(values)
            feature_names.append(col)
    
    # Concatenate
    X = np.hstack(features_list) if features_list else np.array([])
    
    return X, feature_names, label_encoders, tfidf_vectorizers


# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Đánh giá classifier
    
    Args:
        y_true: Nhãn thực
        y_pred: Nhãn dự đoán
        y_proba: Xác suất dự đoán (optional)
        average: Phương thức tính trung bình
        
    Returns:
        Dict các metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        top_k_accuracy_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Top-k accuracy if probabilities provided
    if y_proba is not None:
        for k in [1, 3, 5]:
            if k <= y_proba.shape[1]:
                metrics[f'top_{k}_accuracy'] = top_k_accuracy_score(y_true, y_proba, k=k)
    
    return metrics


def measure_latency(
    model: Any,
    X: np.ndarray,
    n_iterations: int = 100
) -> Dict[str, float]:
    """
    Đo latency của model
    
    Args:
        model: Model cần đo
        X: Feature matrix
        n_iterations: Số lần đo
        
    Returns:
        Dict thống kê latency
    """
    import time
    
    latencies = []
    
    for i in range(n_iterations):
        sample = X[i % len(X):i % len(X) + 1]
        start = time.time()
        _ = model.predict(sample)
        latencies.append((time.time() - start) * 1000)  # ms
    
    return {
        'avg_ms': np.mean(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies)
    }


# ==============================================================================
# PREDICTION UTILITIES
# ==============================================================================

def get_top_k_predictions(
    model: Any,
    X: np.ndarray,
    label_encoder: Any,
    k: int = 5
) -> List[List[Dict]]:
    """
    Lấy top-k predictions với confidence score
    
    Args:
        model: Model đã train
        X: Feature matrix
        label_encoder: Label encoder cho target
        k: Số kết quả trả về
        
    Returns:
        List các predictions cho mỗi sample
    """
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X)
    else:
        # Fallback cho model không có predict_proba
        predictions = model.predict(X)
        results = []
        for pred in predictions:
            label = label_encoder.inverse_transform([pred])[0]
            results.append([{'label': label, 'confidence': 1.0}])
        return results
    
    results = []
    
    for proba in probas:
        top_indices = np.argsort(proba)[::-1][:k]
        top_predictions = []
        
        for idx in top_indices:
            label = label_encoder.inverse_transform([idx])[0]
            confidence = proba[idx]
            top_predictions.append({
                'label': label,
                'confidence': float(confidence)
            })
        
        results.append(top_predictions)
    
    return results


# ==============================================================================
# ANOMALY DETECTION
# ==============================================================================

def detect_anomalies(
    X: np.ndarray,
    contamination: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phát hiện anomalies sử dụng Isolation Forest
    
    Args:
        X: Feature matrix
        contamination: Tỷ lệ anomalies ước tính
        
    Returns:
        Tuple (anomaly_labels, anomaly_scores)
    """
    from sklearn.ensemble import IsolationForest
    
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    
    anomaly_labels = iso_forest.fit_predict(X)
    anomaly_scores = iso_forest.decision_function(X)
    
    return anomaly_labels, anomaly_scores


# ==============================================================================
# VALUE SUGGESTION
# ==============================================================================

class ValueSuggester:
    """Class đề xuất giá trị cho các trường thiếu/lỗi"""
    
    def __init__(self, df: pd.DataFrame, rules: Dict[str, str]):
        """
        Args:
            df: DataFrame huấn luyện
            rules: Dict quy tắc chuẩn hóa
        """
        self.rules = rules
        self.mode_values = {}
        self.value_counts = {}
        
        # Tính mode và value counts cho mỗi cột
        for col in df.columns:
            if df[col].dtype == 'object':
                mode = df[col].mode()
                if len(mode) > 0:
                    self.mode_values[col] = mode[0]
                self.value_counts[col] = df[col].value_counts().to_dict()
    
    def suggest(
        self, 
        column: str, 
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Đề xuất giá trị cho một cột
        
        Args:
            column: Tên cột
            context: Dict các giá trị context (optional)
            
        Returns:
            List các đề xuất với confidence
        """
        suggestions = []
        
        # Rule-based suggestion
        if column in self.rules:
            suggestions.append({
                'value': self.rules[column],
                'source': 'rule',
                'confidence': 1.0
            })
        
        # Mode-based suggestion
        if column in self.mode_values:
            suggestions.append({
                'value': self.mode_values[column],
                'source': 'mode',
                'confidence': 0.8
            })
        
        # Top values suggestion
        if column in self.value_counts:
            total = sum(self.value_counts[column].values())
            for value, count in list(self.value_counts[column].items())[:3]:
                if value not in [s['value'] for s in suggestions]:
                    suggestions.append({
                        'value': value,
                        'source': 'frequency',
                        'confidence': count / total if total > 0 else 0
                    })
        
        return suggestions
