import torch
import pandas as pd
import numpy as np
import os
import sys
import zipfile
import gc
import glob

# --- SETUP ---
try:
    import torch_directml
    device = torch_directml.device()
    print(f"🚀 DirectML Device aktiv: {device}")
except:
    device = torch.device("cpu")
    print("⚠️ Fallback auf CPU")

from app.logic.brain import QuantumBrain
from app.core.config import settings

# --- KONFIGURATION ---
DATA_DIR = os.path.join(settings.BASE_DIR, "data")
MODEL_PATH = os.path.join(DATA_DIR, "brain_weights.pth")
ROOT_DATA_DIR = os.path.join(settings.BASE_DIR, "Datensets")

LEARNING_RATE = 0.0001 # Etwas niedriger für höhere Stabilität
MAX_GRAD_NORM = 1.0    # GRADIENT CLIPPING (Wichtig gegen NaN)

def robust_prepare_tensor(df):
    try:
        # 1. Nur numerische Daten
        df = df.select_dtypes(include=[np.number])
        if df.empty: return None
        
        # 2. NaN/Inf im Datensatz sofort killen
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
        df = df.fillna(0)

        # 3. Features auf Modell-Größe anpassen (z.B. 6 oder 8)
        target = settings.FEATURE_COUNT
        if len(df.columns) < target:
            # Padding
            for i in range(target - len(df.columns)):
                df[f"pad_{i}"] = 0
        data = df.iloc[:, :target].values.astype(np.float32)

        # 4. Robuste Normalisierung
        d_min, d_max = data.min(axis=0), data.max(axis=0)
        norm = (data - d_min) / (d_max - d_min + 1e-8)
        
        # 5. Reshape zu Sequenzen
        seq_len = settings.SEQUENCE_LENGTH
        num = len(norm) // seq_len
        if num < 1: return None
        
        tensor = torch.tensor(norm[:num*seq_len].reshape(num, seq_len, target))
        return tensor.to(device)
    except: return None

def train_enterprise():
    print("🚀 STARTE ROBUSTES ENTERPRISE TRAINING...")
    
    brain = QuantumBrain().to(device)
    # LÖSCHE ALTE PTH DATEI VORHER MANUELL!
    if os.path.exists(MODEL_PATH):
        try: brain.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except: print("✨ Starte Training bei Null.")

    optimizer = torch.optim.AdamW(brain.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    files = glob.glob(os.path.join(ROOT_DATA_DIR, "**", "*.zip"), recursive=True)
    
    for i, path in enumerate(files):
        try:
            with zipfile.ZipFile(path, "r") as z:
                csv_name = [n for n in z.namelist() if n.endswith('.csv')][0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, nrows=5000, encoding='latin1')
            
            X = robust_prepare_tensor(df)
            if X is None: continue

            brain.train()
            optimizer.zero_grad()
            
            recon, mu, logvar = brain(X)
            loss, mse = brain.get_loss(X, recon, mu, logvar)

            # --- NaN SCHUTZ ---
            if torch.isnan(loss):
                print(f"⚠️ Warnung: {os.path.basename(path)} hat NaN erzeugt. Überspringe...")
                continue

            loss.backward()
            
            # --- GRADIENT CLIPPING ---
            torch.nn.utils.clip_grad_norm_(brain.parameters(), MAX_GRAD_NORM)
            
            optimizer.step()

            if i % 5 == 0:
                torch.save(brain.state_dict(), MODEL_PATH)
            
            print(f"✅ [{i+1}/{len(files)}] {os.path.basename(path)[:15]} | MSE: {mse:.6f}")
            
            del X, recon, mu, logvar
            gc.collect()

        except Exception as e:
            continue

    torch.save(brain.state_dict(), MODEL_PATH)
    print("✨ TRAINING FINALISIERT.")

if __name__ == "__main__":
    train_enterprise()