import chromadb
import uuid
from datetime import datetime
from app.core.config import settings
from loguru import logger

class VectorMemory:
    def __init__(self):
        # Persistente Datenbank im Container
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.collection = self.client.get_or_create_collection("incident_history")

    def find_similar(self, embedding: list, threshold=0.3):
        """Sucht nach ähnlichen Fehlermustern in der Vergangenheit"""
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=1
            )
            
            if not results['documents'] or not results['documents'][0]:
                return None

            # Distanz prüfen (je kleiner desto ähnlicher)
            distance = results['distances'][0][0]
            if distance < threshold:
                return {
                    "description": results['documents'][0][0],
                    "meta": results['metadatas'][0][0],
                    "confidence": f"{(1-distance)*100:.1f}%"
                }
            return None
        except Exception as e:
            logger.error(f"Memory Error: {e}")
            return None

    def memorize(self, embedding: list, description: str, severity: str, solution: str = "Untersuchung ausstehend"):
        """Lernt aus einem Vorfall und speichert die Lösung"""
        try:
            self.collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                documents=[description], # Wir suchen nach der Beschreibung/Muster
                metadatas=[{
                    "timestamp": datetime.now().isoformat(),
                    "severity": severity,
                    "solution": solution  # <-- Das ist neu! Das Expertenwissen.
                }]
            )
            logger.info(f"🧠 Wissen gespeichert: {description} -> Lösung: {solution}")
        except Exception as e:
            logger.error(f"Memorize Failed: {e}")