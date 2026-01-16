⚛️ Quantum Nexus Enterprise (AMD Edition)
Predictive Maintenance AI mit Hybrider Quanten-Architektur

Dieses System nutzt DirectML, um High-End AMD-Grafikkarten (wie die 7900 XTX) unter WSL2 für KI-Training zu nutzen. Es erkennt Maschinenausfälle, bevor sie passieren.




### 1. Installation (Einmalig)
Führe diese Befehle in deinem WSL2 Ubuntu Terminal aus: wsl -d Ubuntu

Bash

# 1. Projektordner betreten
cd ~/quantum_nexus
# 2. Virtuelle Umgebung erstellen & aktivieren
python3 -m venv venv
source venv/bin/activate
# 3. WICHTIG: AMD-Treiber (DirectML) zuerst installieren
pip install torch-directml
# 4. Restliche Abhängigkeiten installieren
pip install -r requirements.txt



### 2. Training 
Bevor der Agent starten kann, muss er deine Maschinen kennenlernen.
Daten ablegen: Kopiere deine CSV- oder ZIP-Dateien (z.B. AI4I 2020) in den Ordner: ~/quantum_nexus/Datensets
Training starten:

Bash

wsl -d Ubuntu
cd ~/quantum_nexus
python3 -m venv venv
source venv/bin/activate
python3 train.py
Das Skript nutzt automatisch die GPU (privateuseone:0). Warte, bis "TRAINING ABGESCHLOSSEN" erscheint.



### 3. Visueller Check (Lifti)
Prüfe, ob der Agent logisch denkt (Langzeit-Verschleiß-Analyse):

Bash

python3 test/test.py



### 4. Live-Server Starten (API)
Startet die Schnittstelle für echte Maschinendaten:

Bash

wsl -d Ubuntu
cd ~/quantum_nexus
python3 -m venv venv
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
Der Server läuft jetzt. Lass dieses Fenster offen!



### 5. Benchmarking (Qualitätstest)
Prüfe in einem zweiten Terminal-Fenster, wie genau der Agent ist:

Bash

source venv/bin/activate
python test/benchmark.py


Zielwerte: Genauigkeit > 80%, Recall > 90%.

## Worauf du achten musst 
Schwellenwerte (Thresholds): Dein Modell ist sehr präzise.

Ändern in: app/main.py (Suche nach risk_score >)
DirectML Besonderheit: Der Code nutzt torch_directml.device(). Wenn du Pakete installierst, vermeide pip install torch (das installiert die Nvidia-Version). Nutze immer torch-directml.

Ordner-Struktur: Der Code nutzt relative Pfade. Du kannst den Ordner quantum_nexus verschieben, aber die interne Struktur (app/, data/, test/) muss gleich bleiben.