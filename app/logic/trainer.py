import torch
import torch.nn as nn
import numpy as np
from loguru import logger
import copy
import os
from app.logic.brain import QuantumBrain
from app.core.config import settings

class AutonomousTrainer:
    def __init__(self):
        self.new_data_buffer = []
        self.gold_standard_buffer = None
        self.is_training = False
        self.batch_size = 32
        
        # Versuche, den "Gold Standard" (Ur-Wissen) zu laden
        self._load_gold_standard()

    def _load_gold_standard(self):
        """Lädt die perfekten Daten, die wir beim Bootcamp generiert haben."""
        path = os.path.join(settings.DATA_DIR, "gold_standard.pt")
        if os.path.exists(path):
            try:
                self.gold_standard_buffer = torch.load(path)
                logger.info(f"🏆 AutonomousTrainer: {len(self.gold_standard_buffer)} Gold-Standard-Sequenzen geladen.")
            except Exception as e:
                logger.error(f"❌ Konnte Gold-Standard nicht laden: {e}")
        else:
            logger.warning("⚠️ Kein Gold-Standard gefunden! Risiko von 'Catastrophic Forgetting'.")

    def add_observation(self, features: list):
        """Sammelt neue Live-Daten."""
        self.new_data_buffer.append(features)
        # Limit auf 2000, damit wir nicht den RAM sprengen
        if len(self.new_data_buffer) > 2000:
            self.new_data_buffer.pop(0)

    def _prepare_sequences(self, raw_data_list):
        """Wandelt flache Listen in Zeitreihen-Sequenzen (Tensors) um."""
        if len(raw_data_list) < settings.SEQUENCE_LENGTH + 1:
            return None, None
            
        data_tensor = torch.tensor(raw_data_list, dtype=torch.float32)
        input_seq = []
        target_seq = []
        
        # Sliding Window
        for i in range(len(data_tensor) - settings.SEQUENCE_LENGTH):
            input_seq.append(data_tensor[i : i + settings.SEQUENCE_LENGTH])
            target_seq.append(data_tensor[i + settings.SEQUENCE_LENGTH]) # Next step prediction
            
        if not input_seq: return None, None
        
        return torch.stack(input_seq), torch.stack(target_seq)

    def retrain_model(self, current_brain: QuantumBrain):
        """
        Der sichere Selbstverbesserungs-Zyklus (Experience Replay + Validation).
        """
        if len(self.new_data_buffer) < 200:
            return current_brain

        logger.info("♻️ START AUTONOMOUS RETRAINING...")
        self.is_training = True
        
        try:
            # 1. Daten Vorbereiten: NEUE DATEN
            X_new, y_new = self._prepare_sequences(self.new_data_buffer)
            if X_new is None: 
                self.is_training = False
                return current_brain

            # 2. Daten Vorbereiten: EXPERIENCE REPLAY (Gold Standard beimischen)
            # Wir mischen 50% neue Daten mit 50% alten perfekten Daten
            # Das "verankert" das Modell im gesunden Zustand.
            X_train = X_new
            y_train = y_new
            
            if self.gold_standard_buffer is not None:
                # Nimm zufällig so viele Gold-Daten wie wir neue Daten haben
                indices = torch.randperm(len(self.gold_standard_buffer))[:len(X_new)]
                X_gold = self.gold_standard_buffer[indices]
                # Bei Autoencodern ist Input = Target (meistens), 
                # hier haben wir aber Next-Step Prediction im Trainer Logic, 
                # also vereinfachen wir: Gold Standard nutzt Input als Target (Rekonstruktion)
                # oder wir müssten Gold-Sequenzen anders speichern. 
                # HIER: Wir nutzen X_gold auch als Target, da perfekte Daten stabil bleiben.
                
                X_train = torch.cat([X_new, X_gold])
                y_train = torch.cat([y_new, X_gold[:, -1, :]]) # Letzter Schritt als Target
                
                logger.info(f"   ⚗️ Mix: {len(X_new)} Neue + {len(indices)} Gold-Standard Samples")

            # 3. Training auf der Kopie
            candidate_brain = copy.deepcopy(current_brain)
            candidate_brain.train()
            optimizer = torch.optim.Adam(candidate_brain.parameters(), lr=0.001) # Low LR für Feintuning
            
            # Shuffling
            perm = torch.randperm(X_train.size(0))
            X_train = X_train[perm]
            y_train = y_train[perm]

            total_loss = 0
            epochs = 5
            
            for e in range(epochs):
                optimizer.zero_grad()
                # Forward durchs Brain (wir ignorieren mu/logvar hier für einfaches Training)
                recon, _, _ = candidate_brain(X_train)
                
                # Wir vergleichen nur den letzten Zeitschritt der Rekonstruktion mit dem Target
                # (Da y_train shape (Batch, Features) ist und recon (Batch, Seq, Features))
                recon_last_step = recon[:, -1, :]
                
                loss = nn.functional.mse_loss(recon_last_step, y_train)
                loss.backward()
                optimizer.step()
                total_loss = loss.item()

            # 4. SAFETY CHECK (Validation Gate)
            # Bevor wir das neue Gehirn freigeben, prüfen wir:
            # "Versteht das neue Gehirn noch, was 'Perfekt' ist?"
            
            if self.gold_standard_buffer is not None:
                candidate_brain.eval()
                with torch.no_grad():
                    # Teste auf ALLEN Gold-Daten
                    g_recon, _, _ = candidate_brain(self.gold_standard_buffer)
                    # MSE berechnen
                    val_loss = nn.functional.mse_loss(g_recon, self.gold_standard_buffer).item()
                    
                logger.info(f"   🛡️ Safety Check (Gold Loss): {val_loss:.6f}")
                
                # THRESHOLD: Wenn der Fehler auf den perfekten Daten zu hoch ist,
                # hat das Modell "vergessen", wie ein gesunder Zustand aussieht.
                # Wir erlauben maximal 0.05 (oder ähnlich, je nach Skalierung).
                if val_loss > 0.05:
                    logger.warning("   ⛔ UPDATE ABGELEHNT: Modell hat den Normalzustand verlernt (Catastrophic Forgetting).")
                    self.new_data_buffer = [] # Buffer leeren, war wohl Müll drin
                    self.is_training = False
                    return current_brain

            # 5. Erfolg
            logger.success(f"   ✅ Update akzeptiert. Train Loss: {total_loss:.6f}")
            self.new_data_buffer = [] # Buffer leeren
            self.is_training = False
            candidate_brain.eval()
            return candidate_brain

        except Exception as e:
            logger.error(f"❌ Retraining Error: {e}")
            self.is_training = False
            return current_brain