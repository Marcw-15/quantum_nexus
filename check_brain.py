import torch
import os
import sys
import numpy as np
import warnings

# DirectML/Numpy-Fix für das Laden alter Gewichte
torch.serialization.add_safe_globals([torch._utils._rebuild_device_tensor_from_numpy])
warnings.filterwarnings("ignore")

sys.path.append(os.getcwd())
from app.logic.brain import QuantumBrain
from app.core.config import settings

def check():
    print("🧠 DIAGNOSE-MODUS: Gehirn-Check (Safe-Device-Mode)")
    model_path = os.path.join(settings.BASE_DIR, "data", "brain_weights.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ FEHLER: Datei nicht gefunden: {model_path}")
        return

    # Wir erzwingen CPU für maximale Kompatibilität beim Testen
    device = torch.device("cpu")
    brain = QuantumBrain().to(device)
    
    try:
        # Laden der Gewichte direkt auf die CPU
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        brain.load_state_dict(state_dict)
        print(f"✅ Gewichte geladen. System bereit.")
    except Exception as e:
        print(f"❌ Ladefehler: {e}")
        return

    brain.eval()
    
    def run_test(val_array):
        # 1. Input auf CPU erstellen
        input_tensor = torch.tensor([[val_array]*10], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            # 2. Vorhersage generieren
            recon, mu, logvar = brain(input_tensor)
            
            # --- WICHTIGSTER FIX: Beide Tensoren auf CPU erzwingen ---
            # Wir berechnen den Fehler auf der CPU, um Device-Mismatch zu vermeiden
            diff = input_tensor.cpu() - recon.cpu()
            mse = torch.mean(diff**2).item()
            
        return mse

    print("\n🧪 VALIDIERUNG DER ANOMALIE-ERKENNUNG:")
    
    # [Air, Proc, Speed, Torque, Tool, Pad, Delta, Power]
    # Normal (Alles im Mittelmaß)
    normal_vals = [0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5]
    score_normal = run_test(normal_vals)
    print(f"🟢 Normalbetrieb Score: {score_normal:.6f}")

    # Kritisch (Massive Hitze und Vibration)
    anomaly_vals = [0.9, 0.1, 0.9, 0.1, 0.9, 0.0, 0.9, 0.1]
    score_crit = run_test(anomaly_vals)
    print(f"🔴 Kritisch Score:     {score_crit:.6f}")

    print("\n--- BEWERTUNG ---")
    if score_normal == 0: score_normal = 1e-9 # Div durch Null Schutz
    diff_factor = score_crit / score_normal
    
    print(f"Anomalie-Erkennungs-Faktor: {diff_factor:.2f}x")

    if diff_factor > 1.1:
        print("✨ ERGEBNIS: Das Gehirn ist einsatzbereit! Es unterscheidet zwischen Normal und Fehler.")
    else:
        print("⚠️ HINWEIS: Die Unterscheidung ist noch schwach. Evtl. länger trainieren.")

if __name__ == "__main__":
    check()