import requests
import time
import numpy as np

# Auth & URL
API_KEY = "sk-enterprise-secure-key-2024" 
URL = "http://localhost:8000/api/v1/analyze"
HEADERS = {"x-api-key": API_KEY}

def scale_value(val, min_val, max_val):
    # Bringt den Wert auf 0.0 - 1.0 und dann auf 0.5 - 1.0 (wie im Training)
    denom = (max_val - min_val) if max_val != min_val else 1.0
    norm = (val - min_val) / denom
    return (norm * 0.5) + 0.5

def get_scaled_payload(raw_values):
    # Typische Min/Max Werte aus dem AI4I Datensatz zur Skalierung
    # [Air, Proc, Speed, Torque, Tool, Pad, Delta, Power]
    mins = [295.0, 305.0, 1300.0, 10.0, 0.0, 0.0, 0.0, 10000.0]
    maxs = [305.0, 315.0, 2800.0, 75.0, 250.0, 1.0, 15.0, 80000.0]
    
    scaled = []
    for i in range(len(raw_values)):
        scaled.append(scale_value(raw_values[i], mins[i], maxs[i]))
    return scaled

def send_data(name, raw_values):
    # WICHTIG: Wir senden die skalierten Werte!
    payload = {
        "source_id": "TEST_UNIT_01", 
        "values": get_scaled_payload(raw_values)
    }
    try:
        res = requests.post(URL, json=payload, headers=HEADERS)
        if res.status_code == 200:
            data = res.json()
            score = data.get('risk_score', 0)
            status = data.get('status')
            diag = data.get('diagnosis')
            print(f"[{name}] Score: {score:.4f} | Status: {status} | {diag}")
        else:
            print(f"[{name}] Fehler: {res.status_code}")
    except Exception as e:
        print(f"Fehler: {e}")

def run_test():
    print("🚀 Starte skalierten Stabilitäts-Test...")
    
    # Basisdaten (Rohwerte)
    # Air, Proc, Speed, Torque, Tool, Pad, Delta, Power
    base = [298.1, 308.6, 1500, 40, 0, 0, 10.5, 60000]
    
    # 1. KALIBRIERUNG (Sequenz füllen)
    for _ in range(5):
        send_data("KALIBRIERUNG", base)
        time.sleep(0.05)

    # 2. GESUND (Ruhiger Lauf)
    print("\n--- TEST: NORMALBETRIEB ---")
    for _ in range(5):
        val = [v + np.random.normal(0, 0.02) for v in base]
        send_data("GESUND", val)

    # 3. VIBRATION (Mechanisches Zittern im Torque)
    print("\n--- TEST: VIBRATION (Mechanisch) ---")
    for _ in range(5):
        val = base.copy()
        val[3] += np.random.normal(0, 12.0) # Starkes Zittern
        send_data("MECHANISCH", val)

    # 4. ENERGIE-ANOMALIE (Power-Sprung)
    print("\n--- TEST: POWER-ANOMALIE (Thermisch) ---")
    for _ in range(5):
        val = base.copy()
        val[7] += 15000 # Massiver Power-Anstieg
        send_data("ANOMALIE", val)

if __name__ == "__main__":
    run_test()