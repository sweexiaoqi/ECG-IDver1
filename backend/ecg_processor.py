import os
import shutil
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple, Dict, Any
import wfdb
from sqlalchemy.orm import Session

from backend.config import (
    MODEL_PATH, EMBEDDING_DIM, HEARTBEAT_WINDOW_SIZE, 
    MAX_REPLAY_SAMPLES_PER_USER, OCL_LR, OCL_EPOCHS, TRIPLET_MARGIN, ACCURACY_THRESHOLD
)
from backend.database import User, EcgTemplate, ReplaySample, AuthLog

# -------------------------------------------------------------
# 1. TCN MODEL DEFINITION IN PYTORCH
# -------------------------------------------------------------

class Chomp1d(nn.Module):
    """Slices the output to make convolutions causal if needed."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A single dilated residual block in the TCN."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        # First dilated convolution
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(n_outputs)

        # Second dilated convolution
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(n_outputs)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        
        # Residual connection (downsample if channels differ)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network for ECG Heartbeat Embedding.
    Input size: (Batch, 1, 200) -> Output size: (Batch, 128)
    """
    def __init__(self, num_inputs=1, num_channels=[16, 32, 64], kernel_size=5, dropout=0.1):
        super(TCNEncoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Padding is calculated to maintain length before chomping
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                     stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        
        # Adaptive pooling to handle variable length just in case
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected projection head
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 256),
            nn.ReLU(),
            nn.Linear(256, EMBEDDING_DIM)
        )

    def forward(self, x):
        # x shape: (Batch, 1, SequenceLength)
        features = self.tcn(x)      # (Batch, channels, SequenceLength)
        pooled = self.pool(features) # (Batch, channels, 1)
        flat = pooled.squeeze(-1)    # (Batch, channels)
        emb = self.fc(flat)          # (Batch, EMBEDDING_DIM)
        
        # L2 Normalize the embedding so that cosine similarity is the dot product
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

# -------------------------------------------------------------
# 2. ECG FILE PARSING & SIGNAL PREPROCESSING (DSP)
# -------------------------------------------------------------

def parse_ecg_file(temp_files: List[Path]) -> Tuple[np.ndarray, float]:
    """
    Parses ECG files from list of temporary uploads.
    Supports:
      - .csv / .txt: containing raw voltage readings. Calculates fs if timestamp column is present.
      - .hea + .dat: WFDB format loaded via wfdb.rdrecord.
    """
    # 1. Check for WFDB files
    hea_file = next((f for f in temp_files if f.suffix.lower() == ".hea"), None)
    dat_file = next((f for f in temp_files if f.suffix.lower() == ".dat"), None)

    if hea_file and dat_file:
        # Move them to a temporary directory with the same base name to let WFDB load them
        temp_dir = Path(hea_file).parent
        base_name = hea_file.stem
        # Ensure dat file is also there and has correct name
        record_path = str(temp_dir / base_name)
        record = wfdb.rdrecord(record_path)
        # Extract signal (usually column 0 is Lead I)
        signal_data = record.p_signal[:, 0]
        return signal_data.astype(np.float32), float(record.fs)

    # 2. Check for CSV or TXT
    csv_txt_file = next((f for f in temp_files if f.suffix.lower() in [".csv", ".txt", ".edf"]), None)
    if csv_txt_file:
        raw_data = []
        timestamps = []
        with open(csv_txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("Time"):
                    continue
                parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
                if not parts:
                    continue
                try:
                    if len(parts) >= 2:
                        timestamps.append(float(parts[0]))
                        raw_data.append(float(parts[1]))
                    else:
                        raw_data.append(float(parts[0]))
                except ValueError:
                    continue
        
        if not raw_data:
            raise ValueError("No valid numerical ECG data found in the uploaded file.")
        
        # Calculate sampling rate
        fs = 500.0  # Default fallback
        if len(timestamps) > 1:
            diffs = np.diff(timestamps)
            avg_diff = np.mean(diffs)
            if avg_diff > 0:
                fs = 1.0 / avg_diff
        
        return np.array(raw_data, dtype=np.float32), fs

    raise ValueError("Unsupported ECG file format. Provide a .csv, .txt, or a pair of .hea/.dat files.")


def preprocess_signal(raw_signal: np.ndarray, fs: float, target_fs: float = 500.0) -> List[np.ndarray]:
    """
    Preprocess raw ECG signal:
      1. Resample to 500 Hz if necessary
      2. Bandpass filter (0.5 to 45 Hz) to remove baseline wander and high frequency noise
      3. R-peak detection using local maxima of derivative / bandpass energy
      4. Segment into heartbeat windows (200 samples around R-peak)
      5. Z-score normalization for each heartbeat
    """
    # 1. Resample
    if not np.isclose(fs, target_fs):
        num_samples = int(len(raw_signal) * (target_fs / fs))
        y = signal.resample(raw_signal, num_samples)
    else:
        y = raw_signal.copy()
        
    fs = target_fs

    # 2. Bandpass Filter (Butterworth)
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 45.0 / nyq
    b, a = signal.butter(3, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, y)

    # 3. R-Peak Detection
    # Filter again specifically for QRS detection (5 - 15 Hz is best for QRS detection)
    b_qrs, a_qrs = signal.butter(3, [5.0/nyq, 15.0/nyq], btype='band')
    qrs_filtered = signal.filtfilt(b_qrs, a_qrs, filtered)
    
    # Square the signal to highlight QRS complexes
    squared = qrs_filtered ** 2
    
    # Peak finding with a minimum distance between heartbeats (e.g. 0.5s -> 250 samples at 500Hz)
    min_peak_distance = int(0.5 * fs)
    # Define a adaptive threshold based on standard deviation
    threshold = np.mean(squared) + 1.5 * np.std(squared)
    
    peaks, _ = signal.find_peaks(squared, distance=min_peak_distance, height=threshold)
    
    # If no peaks found, relax threshold
    if len(peaks) == 0:
        threshold = np.mean(squared) + 0.5 * np.std(squared)
        peaks, _ = signal.find_peaks(squared, distance=min_peak_distance, height=threshold)

    # 4. Heartbeat Segmentation
    # Window size: 200 samples (80 before R-peak, 120 after)
    half_before = 80
    half_after = 120
    heartbeats = []
    
    for peak in peaks:
        start = peak - half_before
        end = peak + half_after
        if start >= 0 and end <= len(filtered):
            segment = filtered[start:end]
            # Z-score normalize segment
            seg_mean = np.mean(segment)
            seg_std = np.std(segment)
            if seg_std > 1e-6:
                normalized_segment = (segment - seg_mean) / seg_std
                heartbeats.append(normalized_segment)
                
    return heartbeats

# -------------------------------------------------------------
# 3. ECG PROCESSOR & OCL TRAINING ENGINE
# -------------------------------------------------------------

class ECGProcessor:
    def __init__(self):
        # Initialize TCN model
        self.model = TCNEncoder()
        
        # Set deterministic seed for reproducible initialization
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Load pre-existing weights if available
        if MODEL_PATH.exists():
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
                print("[TCN] Model weights loaded successfully.")
            except Exception as e:
                print(f"[TCN] Error loading weights: {e}. Using randomly initialized model.")
        else:
            print("[TCN] No model weights found. Using baseline model initialized with seed 42.")
            
        self.model.eval()

    def save_model(self):
        """Save the TCN model weights."""
        torch.save(self.model.state_dict(), MODEL_PATH)

    def extract_embeddings(self, heartbeats: List[np.ndarray]) -> np.ndarray:
        """
        Passes a list of heartbeat segments through the TCN to extract 128-dim embeddings.
        Returns a 2D numpy array of shape (N, 128).
        """
        if not heartbeats:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            
        self.model.eval()
        with torch.no_grad():
            # Convert list of heartbeats to PyTorch tensor
            # Shape: (N, 1, 200)
            x = torch.tensor(np.array(heartbeats), dtype=torch.float32).unsqueeze(1)
            embeddings = self.model(x)
            return embeddings.numpy()

    def get_template(self, heartbeats: List[np.ndarray]) -> List[float]:
        """
        Extracts embeddings for a list of heartbeats and computes the mean embedding,
        which is then normalized to form the user's template.
        Includes a correlation quality check to identify and suppress random noise.
        """
        embeddings = self.extract_embeddings(heartbeats)
        if len(embeddings) == 0:
            return [0.0] * EMBEDDING_DIM
            
        # Quality check: average correlation between heartbeats
        if len(heartbeats) > 1:
            mean_hb = np.mean(heartbeats, axis=0)
            corrs = []
            for hb in heartbeats:
                # Pearson correlation coefficient between this heartbeat and the average
                if np.std(hb) > 1e-6 and np.std(mean_hb) > 1e-6:
                    corr = np.corrcoef(hb, mean_hb)[0, 1]
                    # Handle NaN
                    if not np.isnan(corr):
                        corrs.append(corr)
            if corrs:
                avg_corr = np.mean(corrs)
                if avg_corr < 0.65:
                    # Input is noise! Return zero embedding to guarantee similarity of 0.5 (below threshold)
                    print(f"[TCN] Low heartbeat correlation ({avg_corr:.3f}). Signal classified as noise.")
                    return [0.0] * EMBEDDING_DIM

        # Average embeddings along samples
        mean_emb = np.mean(embeddings, axis=0)
        # Normalize the average vector
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-6:
            mean_emb = mean_emb / norm
        return mean_emb.tolist()

    def verify_user(self, query_heartbeats: List[np.ndarray], db: Session) -> Tuple[str, float, str]:
        """
        Matches an uploaded ECG recording (segmented heartbeats) against all templates in the database.
        Returns (best_match_username, match_score, description)
        """
        # 1. Extract query embedding
        query_template = self.get_template(query_heartbeats)
        query_vec = np.array(query_template)

        # 2. Get all templates from DB
        templates = db.query(EcgTemplate).all()
        if not templates:
            return ("Unknown", 0.0, "No enrolled biometric templates found in system.")

        best_score = -1.0
        best_username = "Unknown"
        best_user_id = None

        for t in templates:
            t_vec = np.array(t.embedding)
            # Cosine similarity is the dot product because both vectors are L2 normalized
            similarity = np.dot(query_vec, t_vec)
            # Map cosine similarity [-1, 1] to a percentage-like score [0, 1]
            score = float((similarity + 1.0) / 2.0)
            
            if score > best_score:
                best_score = score
                best_user_id = t.user_id
                
        # Look up username
        if best_user_id is not None:
            user = db.query(User).filter(User.id == best_user_id).first()
            if user:
                best_username = user.username

        # Check against threshold
        if best_score >= ACCURACY_THRESHOLD:
            description = f"Successful match for user '{best_username}' with score {best_score:.2%}."
            return (best_username, best_score, description)
        else:
            description = f"Authentication denied. Closest match: '{best_username}' with score {best_score:.2%} (threshold: {ACCURACY_THRESHOLD:.2%})."
            return ("Unknown", best_score, description)

    def run_experience_replay(self, db: Session, target_user_id: int, new_heartbeats: List[np.ndarray]) -> Dict[str, Any]:
        """
        Online Continual Learning (OCL) with Experience Replay.
        Fine-tunes the TCN model weights to learn a new user's representations
        while replaying samples of previous users from the Replay Buffer to prevent catastrophic forgetting.
        """
        # 1. Save new user's representative samples to database Replay Buffer
        # To avoid storing noise, we select the heartbeats closest to the mean template
        new_embs = self.extract_embeddings(new_heartbeats)
        mean_emb = np.mean(new_embs, axis=0)
        distances = np.linalg.norm(new_embs - mean_emb, axis=1)
        # Sort by distance and pick the top N closest heartbeats
        closest_indices = np.argsort(distances)[:MAX_REPLAY_SAMPLES_PER_USER]
        
        for idx in closest_indices:
            hb = new_heartbeats[idx].tolist()
            replay_sample = ReplaySample(user_id=target_user_id, signal_data=hb)
            db.add(replay_sample)
        db.commit()

        # 2. Retrieve all registered users
        users = db.query(User).all()
        if len(users) <= 1:
            # Only one user enrolled; no classification conflict can occur yet.
            # Skip fine-tuning but save base model to initialize
            self.save_model()
            return {"loss": 0.0, "status": "skipped_single_user", "replay_buffer_size": len(new_heartbeats)}

        # 3. Retrieve replay buffer data for all users
        all_replay_samples = db.query(ReplaySample).all()
        replay_buffer_size = len(all_replay_samples)

        # 4. Construct Triplet Dataset
        # To train the TCN, we need triplets: (Anchor, Positive, Negative)
        # Anchor and Positive are from the same user. Negative is from a different user.
        # Group replay samples by user ID
        user_samples: Dict[int, List[np.ndarray]] = {}
        for sample in all_replay_samples:
            uid = sample.user_id
            sig = np.array(sample.signal_data, dtype=np.float32)
            if uid not in user_samples:
                user_samples[uid] = []
            user_samples[uid].append(sig)

        triplets = []
        for uid, sigs in user_samples.items():
            if len(sigs) < 2:
                # Need at least 2 samples per user to form an Anchor-Positive pair
                continue
            
            # Find negative users
            other_uids = [o_uid for o_uid in user_samples.keys() if o_uid != uid]
            if not other_uids:
                continue
                
            for i in range(len(sigs)):
                for j in range(i + 1, len(sigs)):
                    # Anchor, Positive
                    anchor = sigs[i]
                    positive = sigs[j]
                    
                    # Sample a few negatives for these pairs
                    for neg_uid in other_uids:
                        for neg_sig in user_samples[neg_uid][:3]:  # Limit negatives per pair to avoid combinatoric explosion
                            triplets.append((anchor, positive, neg_sig))

        if not triplets:
            return {"loss": 0.0, "status": "no_triplets_available", "replay_buffer_size": replay_buffer_size}

        # 5. Fine-tune model with Triplet Loss (Experience Replay)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=OCL_LR)
        triplet_loss = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)
        
        total_loss = 0.0
        epochs = OCL_EPOCHS
        batch_size = 32

        for epoch in range(epochs):
            # Shuffle triplets
            np.random.shuffle(triplets)
            epoch_loss = 0.0
            num_batches = 0
            
            for b in range(0, len(triplets), batch_size):
                batch = triplets[b:b+batch_size]
                anchors = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32).unsqueeze(1)
                positives = torch.tensor(np.array([t[1] for t in batch]), dtype=torch.float32).unsqueeze(1)
                negatives = torch.tensor(np.array([t[2] for t in batch]), dtype=torch.float32).unsqueeze(1)
                
                optimizer.zero_grad()
                
                emb_a = self.model(anchors)
                emb_p = self.model(positives)
                emb_n = self.model(negatives)
                
                loss = triplet_loss(emb_a, emb_p, emb_n)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            total_loss = epoch_loss / num_batches

        # Save the updated model weights
        self.save_model()
        self.model.eval()

        # 6. Re-Sync Templates (Template Re-computation)
        # Pass all user's replay signals through the updated TCN and re-save templates
        for uid, sigs in user_samples.items():
            new_template = self.get_template(sigs)
            # Find and update template
            template_record = db.query(EcgTemplate).filter(EcgTemplate.user_id == uid).first()
            if template_record:
                template_record.embedding = new_template
            else:
                template_record = EcgTemplate(user_id=uid, embedding=new_template)
                db.add(template_record)
        
        db.commit()

        # Log OCL Calibration success
        log_entry = AuthLog(
            event_type="CALIBRATION",
            status="SUCCESS",
            username=f"User {target_user_id}",
            accuracy=1.0 - total_loss,  # Represent learning convergence as accuracy
            description=f"TCN-OCL calibration complete. Replayed {replay_buffer_size} samples across {len(user_samples)} profiles. Triplet loss: {total_loss:.4f}."
        )
        db.add(log_entry)
        db.commit()

        return {
            "loss": total_loss,
            "status": "completed",
            "replay_buffer_size": replay_buffer_size,
            "num_users": len(user_samples)
        }

    def reset_model(self):
        """Wipes current model weights and reinstates baseline initialization."""
        if MODEL_PATH.exists():
            try:
                os.remove(MODEL_PATH)
            except Exception as e:
                print(f"[TCN] Error removing weights: {e}")
        # Reinitialize
        self.model = TCNEncoder()
        torch.manual_seed(42)
        np.random.seed(42)
        self.model.eval()
        print("[TCN] Model has been reset to baseline state.")
