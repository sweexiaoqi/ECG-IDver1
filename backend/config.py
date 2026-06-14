import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
SAMPLES_DIR = BASE_DIR / "samples"
MODEL_PATH = BASE_DIR / "tcn_model.pt"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Database configuration: defaults to local SQLite, but supports PostgreSQL DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/ecg_id.db")

# Security
DEV_PASSWORD = os.getenv("DEV_PASSWORD", "admin123")
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-ecg-id-key")
JWT_ALGORITHM = "HS256"

# Biometrics
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))
EMBEDDING_DIM = 128
HEARTBEAT_WINDOW_SIZE = 200  # Number of samples around R-peak

# Continual Learning
MAX_REPLAY_SAMPLES_PER_USER = 10  # Max heartbeats stored per user in the replay buffer
OCL_LR = 0.005                    # Learning rate for experience replay fine-tuning
OCL_EPOCHS = 10                   # Epochs per registration / manual calibration step
TRIPLET_MARGIN = 0.3              # Triplet loss margin
