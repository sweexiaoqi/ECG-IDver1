import urllib.request
import numpy as np
from pathlib import Path
import os

# Target Directory
BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# PhysioNet ECG-ID Database URLs
PHYSIO_BASE = "https://physionet.org/files/ecgiddb/1.0.0/"
SUBJECTS = ["Person_01", "Person_02", "Person_03"]

def download_physionet_samples() -> bool:
    """Attempts to download real .hea/.dat files from PhysioNet."""
    print("[Downloader] Attempting to download real samples from PhysioNet...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        for subject in SUBJECTS:
            for ext in ["hea", "dat"]:
                file_url = f"{PHYSIO_BASE}{subject}/rec_1.{ext}"
                local_name = f"{subject}_rec_1.{ext}"
                local_path = SAMPLES_DIR / local_name
                
                # Skip if already exists
                if local_path.exists():
                    continue
                
                print(f"  Downloading {subject}/rec_1.{ext} -> {local_name}")
                req = urllib.request.Request(file_url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as response:
                    with open(local_path, 'wb') as out_file:
                        out_file.write(response.read())
                        
        print("[Downloader] Real PhysioNet samples downloaded successfully.")
        return True
    except Exception as e:
        print(f"[Downloader] Network download failed: {e}.")
        print("[Downloader] Falling back to physiological synthetic generator...")
        return False

def generate_synthetic_samples():
    """Generates synthetic CSV files representing different physiological signatures."""
    print("[Generator] Generating synthetic physiological ECG recordings...")
    
    fs = 500.0          # Hz
    duration = 20.0     # seconds
    t = np.arange(0, duration, 1.0 / fs)
    num_samples = len(t)
    
    # Heartbeat template generator (P-Q-R-S-T)
    # Different parameters for each person to create distinct biometric profiles
    def get_heartbeat_template(p_amp, r_amp, t_amp, qrs_width):
        hb_t = np.arange(-0.3, 0.5, 1.0 / fs) # 800 ms window
        hb = np.zeros_like(hb_t)
        
        # P-Wave (Gaussian)
        p_wave = p_amp * np.exp(-((hb_t + 0.15) / 0.035) ** 2)
        # QRS Complex (Narrow spike, Q is negative, R is positive, S is negative)
        q_wave = -0.15 * r_amp * np.exp(-((hb_t + 0.02) / 0.008) ** 2)
        r_wave = r_amp * np.exp(-(hb_t / (qrs_width)) ** 2)
        s_wave = -0.20 * r_amp * np.exp(-((hb_t - 0.02) / 0.010) ** 2)
        # T-Wave (Gaussian)
        t_wave = t_amp * np.exp(-((hb_t - 0.22) / 0.065) ** 2)
        
        hb = p_wave + q_wave + r_wave + s_wave + t_wave
        return hb, hb_t

    # ECG profiles
    profiles = {
        "Person_01": {"p": 0.12, "r": 1.20, "t": 0.28, "qrs": 0.011},
        "Person_02": {"p": 0.08, "r": 0.95, "t": 0.18, "qrs": 0.015},
        "Person_03": {"p": 0.18, "r": 1.45, "t": 0.35, "qrs": 0.009}
    }

    for name, params in profiles.items():
        csv_path = SAMPLES_DIR / f"{name}_rec_1.csv"
        
        # Skip if already exists
        if csv_path.exists():
            continue
            
        # Get individual heartbeat template
        template, hb_t = get_heartbeat_template(params["p"], params["r"], params["t"], params["qrs"])
        
        # Construct full signal with heart rate variability (around 72 bpm -> 0.83 seconds per beat)
        full_signal = np.zeros(num_samples)
        
        # Simulate baseline wander (low freq)
        baseline = 0.15 * np.sin(2 * np.pi * 0.15 * t) + 0.05 * np.sin(2 * np.pi * 0.02 * t)
        
        # Insert heartbeats at intervals
        curr_sample = 150
        while curr_sample < num_samples - 400:
            # Add beat
            start_idx = curr_sample - 150
            end_idx = curr_sample + 250
            
            # Blend template into full signal
            full_signal[start_idx:end_idx] += template[:400]
            
            # Next beat interval (BPM of ~72 with noise to simulate HRV)
            interval = int(fs * np.random.normal(0.83, 0.04))
            curr_sample += interval
            
        # Add high-frequency muscle noise
        noise = np.random.normal(0, 0.02, num_samples)
        
        ecg_signal = full_signal + baseline + noise
        
        # Save to CSV (single column)
        np.savetxt(csv_path, ecg_signal, fmt="%.6f", header="ECG_Voltage", comments="")
        print(f"  Generated synthetic CSV: {csv_path.name}")
        
        # Generate a mock WFDB header file to pair with it, so we can mock verify
        # or load CSV/TXT seamlessly. (The app supports CSV directly!)
        
    print("[Generator] Synthetic ECG sample files generated successfully.")

def main():
    # Attempt real download first, then fallback
    downloaded = download_physionet_samples()
    if not downloaded:
        generate_synthetic_samples()

if __name__ == "__main__":
    main()
