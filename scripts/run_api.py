import uvicorn
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    uvicorn.run(
        "serving.fraud_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )