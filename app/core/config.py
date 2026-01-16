import os
import torch
import torch_directml
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Any

# Wir berechnen den Pfad hier temporär
_base_path = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    API_KEY: str = os.getenv("NEXUS_API_KEY", "sk-enterprise-secure-key-2024")
    PROJECT_NAME: str = "Quantum Nexus (AMD DirectML)"
    
    FEATURE_COUNT: int = 8
    SEQUENCE_LENGTH: int = 10 
    
    # --- FIX: BASE_DIR muss TEIL der Klasse sein ---
    BASE_DIR: str = str(_base_path)
    
    # Pfade
    DATA_DIR: str = str(_base_path / "data")
    MODEL_PATH: str = str(_base_path / "data" / "brain_weights.pth")
    CHROMA_PATH: str = str(_base_path / "data" / "chroma_db")

    # DirectML Device
    DEVICE: Any = torch_directml.device()

settings = Settings()
