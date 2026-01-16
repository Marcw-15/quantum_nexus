import pandas as pd
import requests
import time
import os
import sys
import numpy as np
import zipfile
import random
from sklearn.metrics import accuracy_score, recall_score
from pathlib import Path

# Pfad-Hack für Config-Zugriff
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.core.config import settings

# --- KONFIGURATION ---
URL = "http://localhost:8000/api/v1/analyze"
API_KEY = settings.API_KEY
HEADERS = {"x-api-key": API_KEY}

# WICHTIG: Burst-Größe
# Wir senden 15 Pakete pro Maschine, damit der "Rolling Average" (Größe 5) warm wird.
BURST_SIZE = 15 

# Scaling (Muss zum Training passen!)
DATA_OFFSET = 0.5
DATA_SCALE = 0.5
ALARM_THRESHOLD = 0.25  # Deine optimierte Grenze

# Nimm den Pfad aus der Config
BASE_DIR = settings.DATA_DIR

print(f"📂 Arbeits-Pfad: {BASE_DIR}")
TECH_FILE = None

# 1. Suche nach Daten
search_dirs = [BASE_DIR, os.path.join(settings.BASE_DIR, "Datensets")]
print(f"🔍 Suche nach 'ai4i...csv' oder 'ai4i...zip'...")
for d in search_dirs:
    if not os.path.exists(d): continue
    for root, dirs, files in os.walk(d):
        for f in files:
            if (f.endswith(".csv") or f.endswith(".zip")) and "ai4i" in f.lower():
                TECH_FILE = os.path.join(root, f)
                print(f"   ✅ Gefunden: {TECH_FILE}")
                break
        if TECH_FILE: break
    if TECH_FILE: break

# 2. Helper Funktionen
def get_dataframe(path):
    if path.endswith(".zip"):
        print(f"📦 Entpacke ZIP im Speicher...")
        with zipfile.ZipFile(path, 'r') as z:
            csv_list = [n for n in z.namelist() if n.endswith('.csv')]
            if not csv_list: return None
            with z.open(csv_list[0]) as f:
                return pd.read_csv(f)
    else:
        return pd.read_csv(path)

def smart_normalize_row(row, df_stats):
    """
    Berechnet die 8 Features (inkl. Physik) für den Live-Test
    """
    try:
        # Versuch, die richtigen Spalten zu finden
        air = row[[c for c in row.index if 'air' in c.lower()][0]]
        proc = row[[c for c in row.index if 'process' in c.lower()][0]]
        speed = row[[c for c in row.index if 'speed' in c.lower()][0]]
        torque = row[[c for c in row.index if 'torque' in c.lower()][0]]
        
        # Tool wear als 5. Feature (oft wichtig)
        tool = 0.0
        try: tool = row[[c for c in row.index if 'tool' in c.lower()][0]]
        except: pass

    except:
        # Fallback: Nimm einfach die ersten Spalten
        vals = list(row.values)
        air, proc, speed, torque = vals[0], vals[1], vals[2], vals[3]
        tool = vals[4] if len(vals) > 4 else 0.0

    # Physik Features
    delta_temp = proc - air
    power = speed * torque
    
    # Die 8 Features zusammenbauen
    # [Air, Proc, Speed, Torque, Tool, 0.0(Pad), Delta, Power]
    final_feats = [air, proc, speed, torque, tool, 0.0, delta_temp, power]
    
    # Normalisieren & Skalieren
    norm_row = []
    
    for i, val in enumerate(final_feats):
        # Wir simulieren hier die Normalisierung vom Training
        # Da wir im Benchmark keine globalen Stats laden, nutzen wir
        # grobe, typische Industriewerte für die Skalierung.
        
        if i == 6: # Delta Temp (meist 8-15)
            norm_val = (val - 8.0) / (15.0 - 8.0)
        elif i == 7: # Power (meist 0-10000)
            norm_val = val / 10000.0
        elif i == 2: # Speed (RPM)
            norm_val = val / 3000.0
        elif i == 3: # Torque (Nm)
            norm_val = val / 100.0
        elif i == 0 or i == 1: # Temps (K)
            norm_val = (val - 290) / (350 - 290)
        else:
            norm_val = val / 250.0 # Tool wear etc
            
        # Clipping (damit nichts explodiert)
        norm_val = max(0.0, min(1.0, norm_val))
        
        # Scaling für Brain
        norm_val = (norm_val * DATA_SCALE) + DATA_OFFSET
        norm_row.append(norm_val)
        
    return norm_row

# 3. Main Test
def run_benchmark():
    print(f"⚖️ STARTE PROFI-BENCHMARK (Burst Mode: {BURST_SIZE} Pings/Maschine)")
    print(f"   Grenzwert: {ALARM_THRESHOLD}")
    
    if TECH_FILE:
        df = get_dataframe(TECH_FILE)
    else:
        print("❌ Keine Daten gefunden.")
        return

    # Label finden
    label_col = next((c for c in df.columns if 'fail' in c.lower()), None)
    if not label_col: return

    # Test-Set: 50 Gesunde, 50 Kaputte
    try:
        df_broken = df[df[label_col] == 1]
        df_healthy = df[df[label_col] == 0]
        sample_size = min(50, len(df_broken))
        
        test_set = pd.concat([
            df_healthy.sample(sample_size), 
            df_broken.sample(sample_size)
        ]).sample(frac=1) # Mischen
    except:
        test_set = df.sample(100)

    print(f"🧪 Test-Größe: {len(test_set)} Fälle")
    print("🚀 Sende Daten-Salven an API...")
    
    y_true = []
    y_pred = []
    
    scores_healthy = []
    scores_broken = []
    
    counter = 0
    for idx, row in test_set.iterrows():
        counter += 1
        true_state = row[label_col]
        
        # Smart Features berechnen
        norm_values = smart_normalize_row(row, None)
        
        final_risk = 0.0
        
        try:
            # --- BURST MODE ---
            # Wir senden die gleichen Daten X mal hintereinander,
            # um den Server-Puffer zu füllen und den stabilen Wert zu kriegen.
            for i in range(BURST_SIZE):
                resp = requests.post(
                    URL, 
                    json={"source_id": f"Bench-{idx}", "values": norm_values}, 
                    headers=HEADERS
                )
                if i == BURST_SIZE - 1: # Nur das letzte Ergebnis zählt (stabilisiert)
                    res_json = resp.json()
                    final_risk = res_json['risk_score']
            
            # Auswertung
            if true_state == 0: scores_healthy.append(final_risk)
            else: scores_broken.append(final_risk)
            
            # Entscheidung basierend auf Grenzwert
            ai_sees_broken = 1 if final_risk > ALARM_THRESHOLD else 0
            
            y_true.append(true_state)
            y_pred.append(ai_sees_broken)
            
            icon = "✅" if ai_sees_broken == true_state else "❌"
            # Live-Log
            print(f"\rFall {counter}/{len(test_set)} {icon} (Real: {true_state} | AI: {final_risk:.4f})", end="")
                
        except Exception as e:
            print(f"\nFehler: {e}")

    print("\n\n📊 --- ERGEBNISSE (Geglättet) ---")
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred) * 100
        rec = recall_score(y_true, y_pred, zero_division=0) * 100
        
        print(f"Genauigkeit: {acc:.1f}%")
        print(f"Recall:      {rec:.1f}%")
        
        avg_h = np.mean(scores_healthy) if scores_healthy else 0
        avg_b = np.mean(scores_broken) if scores_broken else 0
        
        print("\n🧠 INDUSTRIE-ANALYSE:")
        print(f"   Score Gesund: {avg_h:.4f}")
        print(f"   Score Defekt: {avg_b:.4f}")
        print(f"   Abstand:      {avg_b - avg_h:.4f}")
    else:
        print("Keine Daten.")

if __name__ == "__main__":
    run_benchmark()