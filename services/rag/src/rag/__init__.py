import uvicorn
from rag.api import app
import os 

def main():
    uvicorn.run(
        'rag:app',
        host='0.0.0.0',
        port=int(os.getenv("PORT", 3010)),
        workers=int(os.getenv("WORKERS", 1)),
        reload=True,
    )
