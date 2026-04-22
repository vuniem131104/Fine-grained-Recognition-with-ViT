import uvicorn
from authentication.api import app
import os 

def main():
    uvicorn.run(
        'authentication:app',
        host='0.0.0.0',
        port=int(os.getenv("PORT", 8001)),
        workers=int(os.getenv("WORKERS", 1)),
        reload=True,
    )

if __name__ == "__main__":
    main()
