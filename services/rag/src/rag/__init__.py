import uvicorn
from rag.api import app
import os 

def main():
    uvicorn.run(
        'rag:app',
        host='0.0.0.0',
        port=int(os.getenv("PORT")),
        workers=int(os.getenv("WORKERS")),
        reload=True,
    )
