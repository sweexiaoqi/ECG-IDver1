import uvicorn
import generate_samples

def bootstrap():
    print("[Bootstrap] Initializing ECG ID Biometric Recognition Platform...")
    
    # 1. Download or generate sample records for testing
    try:
        generate_samples.main()
    except Exception as e:
        print(f"[Bootstrap] Warning: failed to complete sample initialization: {e}")
        
    # 2. Launch FastAPI server
    print("[Bootstrap] Launching FastAPI Web Server on http://127.0.0.1:8000...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    bootstrap()
