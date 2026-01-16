import numpy as np
import pandas as pd
from loguru import logger
from app.core.config import settings

class DataSanitizer:
    """Enterprise Data Cleaning Pipeline"""
    
    def clean(self, raw_values: list) -> np.ndarray:
        try:
            # 1. Type Enforcement (Alles muss Float sein)
            arr = np.array(pd.to_numeric(raw_values, errors='coerce'), dtype=float)
            
            # 2. Dimension Fix (Falls Sensor ausfällt und Array zu kurz ist)
            target = settings.FEATURE_COUNT
            if len(arr) != target:
                logger.warning(f"⚠️ Data Mismatch: Input {len(arr)} vs Target {target}. Resizing.")
                arr = np.resize(arr, target)
            
            # 3. Imputation (NaN/Null Werte reparieren)
            if np.isnan(arr).any():
                # Wir füllen Lücken mit 0.0 (Neutralwert bei Normierung)
                arr = np.nan_to_num(arr, nan=0.0)
                
            return arr
        except Exception as e:
            logger.error(f"❌ FATAL SANITIZATION ERROR: {e}")
            # Notfall-Fallback: Leeres Signal senden, damit Pipeline nicht crasht
            return np.zeros(settings.FEATURE_COUNT)