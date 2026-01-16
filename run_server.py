import uvicorn
import os
import sys
from app.core.config import settings

# Notwendig für PyInstaller
if getattr(sys, 'frozen', False):
    sys.path.append(os.path.join(sys._MEIPASS, 'app'))

if __name__ == "__main__":
    print("🚀 QUANTUM NEXUS ENTERPRISE - Starting Binary...")
    print(f"🔒 License: VALID")
    print(f"📡 Interface: http://0.0.0.0:8000")
    
    # Server starten
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
