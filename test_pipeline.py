import unittest
import numpy as np
import os
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set environment variable to use test database
os.environ["DATABASE_URL"] = "sqlite:///./test_ecg_id.db"

from backend.config import BASE_DIR, ACCURACY_THRESHOLD
from backend.database import Base, User, EcgTemplate, ReplaySample, AuthLog, get_db
from backend.ecg_processor import ECGProcessor, preprocess_signal, parse_ecg_file
from backend.main import app

# Create a local sessionmaker for testing
engine = create_engine("sqlite:///./test_ecg_id.db", connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override get_db dependency in FastAPI app
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

class TestECGIDSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup clean test database
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        
        # Initialize processor
        cls.processor = ECGProcessor()
        
        # Generate some synthetic ECG signals for testing
        cls.fs = 500.0
        cls.duration = 7.0  # 7 seconds is enough for tests
        cls.t = np.arange(0, cls.duration, 1.0 / cls.fs)
        
        # Simple template heartbeat
        cls.hb_template = np.sin(2 * np.pi * 5 * np.arange(0, 0.4, 1.0/cls.fs)) # 200 samples
        cls.hb_template = (cls.hb_template - np.mean(cls.hb_template)) / np.std(cls.hb_template)

    @classmethod
    def tearDownClass(cls):
        # Dispose SQLAlchemy engine to release file locks on Windows
        engine.dispose()
        
        # Cleanup test database file
        try:
            Base.metadata.drop_all(bind=engine)
            db_file = Path("test_ecg_id.db")
            if db_file.exists():
                db_file.unlink()
        except PermissionError:
            print("[Test] Database file locked on teardown. Skipping deletion (it will be reset on next run).")
        
        # Clear main app overrides
        app.dependency_overrides.clear()

    def setUp(self):
        # Clean database tables before each test
        db = TestingSessionLocal()
        db.query(AuthLog).delete()
        db.query(ReplaySample).delete()
        db.query(EcgTemplate).delete()
        db.query(User).delete()
        db.commit()
        db.close()

    def generate_synthetic_ecg(self, multiplier=1.0):
        """Helper to create a synthetic ECG recording array."""
        signal = np.random.normal(0, 0.05, len(self.t))
        # Insert 6 heartbeats spaced out
        peaks = [500, 1000, 1500, 2000, 2500, 3000]
        for p in peaks:
            signal[p-80:p+120] += self.hb_template * multiplier
        return signal

    def test_01_preprocessing(self):
        """Test that ECG signals are parsed, filtered, and segmented correctly."""
        print("\n[Test] Testing ECG signal preprocessing...")
        signal_data = self.generate_synthetic_ecg()
        
        # Run preprocessing
        heartbeats = preprocess_signal(signal_data, self.fs)
        
        # Verify segmented heartbeats
        self.assertGreaterEqual(len(heartbeats), 3)
        self.assertEqual(len(heartbeats[0]), 200)
        
        # Verify Z-score normalization
        self.assertAlmostEqual(np.mean(heartbeats[0]), 0.0, places=4)
        self.assertAlmostEqual(np.std(heartbeats[0]), 1.0, places=4)
        print("  -> ECG preprocessing passed: segmented and normalized heartbeats successfully.")

    def test_02_tcn_encoder(self):
        """Test TCN Encoder model output dimension and normalization."""
        print("\n[Test] Testing TCN Encoder model...")
        signal_data = self.generate_synthetic_ecg()
        heartbeats = preprocess_signal(signal_data, self.fs)
        
        # Extract embeddings
        embeddings = self.processor.extract_embeddings(heartbeats)
        
        # Verify embedding dimensions
        self.assertEqual(embeddings.shape, (len(heartbeats), 128))
        
        # Verify L2 normalization (norm should be 1.0 for each row)
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            self.assertAlmostEqual(norm, 1.0, places=5)
            
        print("  -> TCN Encoder passed: extracted L2-normalized 128-dimensional embeddings.")

    def test_03_continual_learning_replay(self):
        """Test Online Continual Learning with Replay Buffer and Triplet Loss."""
        print("\n[Test] Testing TCN-OCL Continual Learning / Experience Replay loop...")
        db = TestingSessionLocal()
        
        # Enroll first user: Alice
        alice = User(username="alice")
        db.add(alice)
        db.commit()
        db.refresh(alice)
        
        alice_signals = preprocess_signal(self.generate_synthetic_ecg(multiplier=1.2), self.fs)
        alice_template_emb = self.processor.get_template(alice_signals)
        
        alice_template = EcgTemplate(user_id=alice.id, embedding=alice_template_emb)
        db.add(alice_template)
        db.commit()
        
        # Run experience replay (first user should skip OCL loop training but save template)
        stats1 = self.processor.run_experience_replay(db, alice.id, alice_signals)
        self.assertEqual(stats1["status"], "skipped_single_user")
        
        # Enroll second user: Bob
        bob = User(username="bob")
        db.add(bob)
        db.commit()
        db.refresh(bob)
        
        bob_signals = preprocess_signal(self.generate_synthetic_ecg(multiplier=0.8), self.fs)
        bob_template_emb = self.processor.get_template(bob_signals)
        
        bob_template = EcgTemplate(user_id=bob.id, embedding=bob_template_emb)
        db.add(bob_template)
        db.commit()
        
        # Run experience replay for Bob (with Alice in buffer)
        stats2 = self.processor.run_experience_replay(db, bob.id, bob_signals)
        
        # Verification
        self.assertEqual(stats2["status"], "completed")
        self.assertGreater(stats2["replay_buffer_size"], 0)
        self.assertEqual(stats2["num_users"], 2)
        
        # Check that templates are updated in database
        alice_template_updated = db.query(EcgTemplate).filter(EcgTemplate.user_id == alice.id).first()
        bob_template_updated = db.query(EcgTemplate).filter(EcgTemplate.user_id == bob.id).first()
        
        self.assertIsNotNone(alice_template_updated)
        self.assertIsNotNone(bob_template_updated)
        
        # Verify calibration log was written
        calib_log = db.query(AuthLog).filter(AuthLog.event_type == "CALIBRATION").first()
        self.assertIsNotNone(calib_log)
        self.assertEqual(calib_log.status, "SUCCESS")
        
        db.close()
        print("  -> Experience Replay and template re-sync passed.")

    def test_04_api_endpoints(self):
        """Test API endpoints via FastAPI TestClient."""
        print("\n[Test] Testing API Endpoints via HTTP requests...")
        client = TestClient(app)
        
        # 1. Register Alice via API
        # Create temp file
        ecg_data = self.generate_synthetic_ecg(multiplier=1.1)
        temp_csv = Path("test_alice.csv")
        np.savetxt(temp_csv, ecg_data, fmt="%.6f")
        
        with open(temp_csv, "rb") as f:
            response = client.post(
                "/api/users/register",
                data={"username": "alice_api"},
                files=[("files", ("test_alice.csv", f, "text/csv"))]
            )
        
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json["success"])
        self.assertEqual(res_json["username"], "alice_api")
        
        # 2. Verify Alice (should approve)
        with open(temp_csv, "rb") as f:
            response = client.post(
                "/api/auth/verify",
                files=[("files", ("test_alice.csv", f, "text/csv"))]
            )
            
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json["verified"])
        self.assertEqual(res_json["username"], "alice_api")
        self.assertGreaterEqual(res_json["accuracy"], ACCURACY_THRESHOLD)
        
        # 3. Verify unregistered user/noisy ECG (should deny)
        noisy_data = np.random.normal(0, 0.8, len(self.t))
        temp_noisy_csv = Path("test_noisy.csv")
        np.savetxt(temp_noisy_csv, noisy_data, fmt="%.6f")
        
        with open(temp_noisy_csv, "rb") as f:
            response = client.post(
                "/api/auth/verify",
                files=[("files", ("test_noisy.csv", f, "text/csv"))]
            )
            
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertFalse(res_json["verified"])
        
        # 4. Dev Login
        response = client.post("/api/dev/login", data={"password": "admin123"})
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json["success"])
        token = res_json["token"]
        
        # 5. Fetch Logs (authenticated)
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/logs", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.json()), 0)
        
        # Cleanup temp CSV files
        if temp_csv.exists():
            temp_csv.unlink()
        if temp_noisy_csv.exists():
            temp_noisy_csv.unlink()
            
        print("  -> API endpoints (Register, Verify, Dev Login, Fetch Logs) passed.")

if __name__ == "__main__":
    unittest.main()
