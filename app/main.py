from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import numpy as np
import os
import asyncio
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any
from loguru import logger

from app.core.config import settings
from app.logic.brain import QuantumBrain
from app.logic.memory import VectorMemory

# --- SARTORIUS PRECISION CONFIG ---
VIBRATION_WEIGHT = 12.0  # Fokus auf mechanische Stabilität
PHYSICS_WEIGHT = 15.0    # Fokus auf Energieerhaltung (Power/DeltaTemp)
HISTORY_SIZE = 5         # Glättungs-Fenster
THRESHOLD_CRITICAL = 6.5 # Basierend auf Benchmark-Kalibrierung

# --- INTERNER SARTORIUS SCALER ---
class SmartScaler:
    """Sorgt dafür, dass Rohdaten immer im KI-Wohlfühlbereich landen."""
    # Industrie-Grenzwerte (Min/Max)
    MINS = np.array([295.0, 305.0, 1300.0, 10.0, 0.0, 0.0, 0.0, 10000.0])
    MAXS = np.array([305.0, 315.0, 2800.0, 75.0, 250.0, 1.0, 15.0, 80000.0])

    @classmethod
    def transform(cls, values: List[float]) -> List[float]:
        val_arr = np.array(values)
        # Normalisierung (0..1)
        norm = (val_arr - cls.MINS) / (cls.MAXS - cls.MINS + 1e-6)
        # Skalierung (0.5..1.0) für VAE-Input
        scaled = (norm * 0.5) + 0.5
        return np.clip(scaled, 0.0, 1.2).tolist()

class SessionManager:
    def __init__(self):
        self._buffers = defaultdict(lambda: deque(maxlen=settings.SEQUENCE_LENGTH))
        self._locks = defaultdict(asyncio.Lock)

    async def add_data(self, sid, data):
        async with self._locks[sid]:
            self._buffers[sid].append(data)

    async def get_tensor(self, sid):
        async with self._locks[sid]:
            if len(self._buffers[sid]) < settings.SEQUENCE_LENGTH:
                return None
            return torch.tensor([list(self._buffers[sid])], dtype=torch.float32).to(settings.DEVICE)

# --- GLOBALE INSTANZEN ---
session_manager = SessionManager()
brain, memory = None, None
score_history = {}
model_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain, memory
    memory = VectorMemory()
    brain = QuantumBrain().to(settings.DEVICE)
    if os.path.exists(settings.MODEL_PATH):
        try:
            brain.load_state_dict(torch.load(settings.MODEL_PATH, map_location=settings.DEVICE))
            brain.eval()
            logger.success("🚀 SARTORIUS-CORE ONLINE: Gehirn geladen.")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Gewichte: {e}")
    yield

app = FastAPI(title="Quantum Nexus - Sartorius Edition", lifespan=lifespan)

# CORS für Dashboards
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class InputPayload(BaseModel):
    source_id: str
    values: List[float]

@app.post("/api/v1/analyze")
async def analyze(payload: InputPayload, x_api_key: str = Header(...)):
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Ungültiger API-Key")

    # 1. Daten-Normalisierung (Padding & Scaling)
    raw_vals = (payload.values + [0.0]*8)[:8]
    scaled_vals = SmartScaler.transform(raw_vals)
    
    await session_manager.add_data(payload.source_id, scaled_vals)
    tensor_in = await session_manager.get_tensor(payload.source_id)
    
    if tensor_in is None:
        return {
            "source_id": payload.source_id,
            "status": "CALIBRATION",
            "risk_score": 0.0,
            "diagnosis": "Initialisierung läuft..."
        }

    # 2. KI-Inferenz & Physik-Analyse
    async with model_lock:
        with torch.no_grad():
            # A) Vibration (Abweichung im Torque-Kanal)
            vibration = torch.std(tensor_in[0, :, 3]).item()
            
            # B) Rekonstruktion (VAE Fehler-Check)
            recon, mu, _ = brain(tensor_in)
            # Fokus auf Kanäle 6 (DeltaTemp) und 7 (Power)
            physics_mse = torch.mean((recon[0, :, 6:] - tensor_in[0, :, 6:])**2).item()
            
            # C) Hybrid Score Berechnung
            raw_mse = (physics_mse * PHYSICS_WEIGHT) + (vibration * VIBRATION_WEIGHT)

    # 3. Glättung des Ergebnisses
    sid = payload.source_id
    if sid not in score_history:
        score_history[sid] = deque(maxlen=HISTORY_SIZE)
    score_history[sid].append(raw_mse)
    smooth_score = sum(score_history[sid]) / len(score_history[sid])

    # 4. Experten-Diagnose
    status = "OK"
    insight = "System stabil."

    if smooth_score > THRESHOLD_CRITICAL:
        status = "CRITICAL"
        insight = "🚨 Anomalie in der physikalischen Signatur erkannt!"
    elif smooth_score > THRESHOLD_CRITICAL * 0.7:
        status = "WARNING"
        insight = "⚠️ Erhöhte mechanische Instabilität festgestellt."

    return {
        "source_id": sid,
        "status": status,
        "risk_score": round(smooth_score, 4),
        "diagnosis": insight,
        "metrics": {
            "vibration": round(vibration, 4),
            "physics_gap": round(physics_mse, 4)
        }
    }