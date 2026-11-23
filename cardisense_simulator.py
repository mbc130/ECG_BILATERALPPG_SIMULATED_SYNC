"""
cardisense_simulator.py
-----------------------
Cardisense Standalone ECG + Bilateral PPG Acquisition Simulator & Synchronizer.

Features:
- Discovers and loads standalone ECG CSVs (ADS1292R) and bilateral PPG CSVs (AFE4490 dual).
- Precision filtering and fiducial point detection (ECG R-peaks, PPG systolic feet and peaks).
- Realistic beat-by-beat phase-locked synchronization with user-customizable PAT_Left, PAT_Right,
  and Bilateral dPTT transit differences.
- Preset disease simulation profiles (Healthy Normal, Arterial Stiffness / Hypertension,
  Unilateral PAD Left/Right Stenosis, Respiratory Sinus Arrhythmia).
- Generates high-resolution diagnostic waveform visual figures with PAT/PTT caliper annotations.
- Exports mode-specific beat-by-beat timing CSVs, compact waveform CSVs, and full continuous acquisition CSV.
"""

import os
import sys
import glob
import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Default repository paths
DEFAULT_ECG_DIR = r"C:\Users\Mubeen\Desktop\Cardisense\Prototype\ADS1292r\ecg_ads1292\stm32_ecg\ecg_ads1292_stm32f411_arduino\new raw sps"
DEFAULT_PPG_DIR = r"C:\Users\Mubeen\Desktop\Cardisense\Prototype\AFE4490\Code\afe4490_dual"
OUTPUT_DIR = r"C:\Users\Mubeen\Desktop\Cardisense\Simulation\output"


# ---------------------------------------------------------------------------
# Signal Processing Utilities
# ---------------------------------------------------------------------------

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """Zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    return signal.sosfiltfilt(sos, data)

def despike_and_interp(arr, sat_val=2096900):
    """Interpolate saturated or invalid ADC samples."""
    arr = arr.astype(float).copy()
    sat = (arr >= sat_val) | (arr <= -sat_val) | np.isnan(arr)
    if sat.any() and not sat.all():
        idx = np.arange(len(arr))
        arr[sat] = np.interp(idx[sat], idx[~sat], arr[~sat])
    return arr

def detect_ecg_r_peaks(ecg_signal, fs):
    """
    Pan-Tompkins inspired robust R-peak detector with parabolic peak refinement.
    """
    x = ecg_signal - np.median(ecg_signal)
    nyq = fs / 2.0
    sos_qrs = signal.butter(2, [max(0.01, 5.0 / nyq), min(0.99, 18.0 / nyq)], btype='bandpass', output='sos')
    x_qrs = signal.sosfiltfilt(sos_qrs, x)
    
    diff = np.gradient(x_qrs)
    energy = diff ** 2
    
    w_size = max(3, int(0.12 * fs))
    smooth_energy = np.convolve(energy, np.ones(w_size) / w_size, mode='same')
    
    min_dist = int(0.35 * fs)
    thresh = 0.20 * np.percentile(smooth_energy, 99)
    peaks, _ = signal.find_peaks(smooth_energy, distance=min_dist, height=thresh)
    
    sos_full = signal.butter(2, [max(0.01, 0.5 / nyq), min(0.99, 40.0 / nyq)], btype='bandpass', output='sos')
    ecg_filt = signal.sosfiltfilt(sos_full, x)
    
    if abs(np.percentile(ecg_filt, 1)) > abs(np.percentile(ecg_filt, 99)):
        ecg_filt = -ecg_filt
        
    search_half = int(0.08 * fs)
    refined_r = []
    for p in peaks:
        lo = max(0, p - search_half)
        hi = min(len(ecg_filt), p + search_half)
        if hi > lo:
            sub_idx = lo + np.argmax(ecg_filt[lo:hi])
            refined_r.append(sub_idx)
            
    refined_r = np.array(sorted(list(set(refined_r))), dtype=int)
    return refined_r, ecg_filt

def detect_ppg_fiducials(ppg_signal, fs):
    """
    Detects pulse foot (onset) and systolic peak for PPG pulses.
    """
    nyq = fs / 2.0
    sos_ppg = signal.butter(4, [max(0.01, 0.5 / nyq), min(0.99, 8.0 / nyq)], btype='bandpass', output='sos')
    filt = signal.sosfiltfilt(sos_ppg, ppg_signal - np.mean(ppg_signal))
    
    min_dist = int(0.40 * fs)
    amp_range = np.percentile(filt, 95) - np.percentile(filt, 5)
    peaks, _ = signal.find_peaks(filt, distance=min_dist, prominence=0.25 * amp_range)
    
    feet = []
    for pk in peaks:
        lookback = int(0.40 * fs)
        lo = max(0, pk - lookback)
        if pk > lo:
            seg_filt = filt[lo:pk]
            foot_idx = lo + np.argmin(seg_filt)
            feet.append(foot_idx)
            
    return np.array(feet, dtype=int), np.array(peaks, dtype=int), filt


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

def load_ecg_file(filepath):
    """Loads ECG CSV and returns (df, fs, raw_ecg, raw_bioz)."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    
    fs = 500.0
    if 'PC_Timestamp_ms' in df.columns and len(df) > 1:
        dt_ms = np.median(np.diff(df['PC_Timestamp_ms'].values))
        if dt_ms > 0:
            fs_est = 1000.0 / dt_ms
            for std_fs in [125, 250, 500, 1000, 2000]:
                if abs(fs_est - std_fs) < std_fs * 0.15:
                    fs = float(std_fs)
                    break
    m = re.search(r'(\d+)sps', os.path.basename(filepath))
    if m:
        fs = float(m.group(1))

    raw_ecg = df['CH1_ECG'].values.astype(float) if 'CH1_ECG' in df.columns else df.iloc[:, 1].values.astype(float)
    raw_bioz = df['CH2_BioZ'].values.astype(float) if 'CH2_BioZ' in df.columns else np.zeros_like(raw_ecg)
    
    return df, fs, raw_ecg, raw_bioz

def load_bilateral_ppg_file(filepath):
    """Loads bilateral PPG CSV and returns (p0_df, p1_df, fs, meta)."""
    meta = {}
    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('#'):
                parts = line.strip('# \r\n').split(':')
                if len(parts) == 2:
                    meta[parts[0].strip()] = parts[1].strip()
            else:
                break
                
    df = pd.read_csv(filepath, comment='#')
    df.columns = [c.strip().lower() for c in df.columns]
    
    fs = 500.0
    if 'Frequency' in meta:
        m = re.search(r'(\d+)', meta['Frequency'])
        if m:
            fs = float(m.group(1))
    elif '1000hz' in filepath.lower():
        fs = 1000.0
    elif '500hz' in filepath.lower():
        fs = 500.0

    p0 = df[df['ch'] == 0].sort_values('t_us').copy()
    p1 = df[df['ch'] == 1].sort_values('t_us').copy()
    
    return p0, p1, fs, meta


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

class CardisenseSimulator:
    def __init__(self, ecg_path=None, ppg_path=None):
        self.ecg_path = ecg_path
        self.ppg_path = ppg_path
        self.target_fs = 500.0
        self.pat_l_target_ms = 210.0
        self.pat_r_target_ms = 218.0
        self.jitter_ms = 2.0
        self.disease_profile = "Healthy Normal"
        self.duration_s = 30.0
        self.amp_ratio_l = 1.0
        self.amp_ratio_r = 1.0

    @staticmethod
    def get_mode_slug(profile_name):
        """Converts profile name to a clean filename slug."""
        name = profile_name.lower()
        if "normal" in name:
            return "normal_mode"
        elif "stiffness" in name or "hyper" in name:
            return "arterial_stiffness_mode"
        elif "left" in name:
            return "pad_left_stenosis_mode"
        elif "right" in name:
            return "pad_right_stenosis_mode"
        elif "rsa" in name or "arrhythmia" in name:
            return "rsa_arrhythmia_mode"
        else:
            return re.sub(r'[^a-z0-9]+', '_', name).strip('_')

    @staticmethod
    def get_available_files():
        """Scans workspace directories for ECG and PPG recordings."""
        ecg_files = glob.glob(os.path.join(DEFAULT_ECG_DIR, "*.csv"))
        ppg_files = [f for f in glob.glob(os.path.join(DEFAULT_PPG_DIR, "*.csv")) if "summary" not in f]
        return {
            "ecg": [os.path.abspath(f) for f in ecg_files],
            "ppg": [os.path.abspath(f) for f in ppg_files]
        }

    def set_disease_preset(self, preset_name):
        """Sets standard clinical presets for PAT/PTT disease simulation."""
        self.disease_profile = preset_name
        if preset_name == "Healthy Normal":
            self.pat_l_target_ms = 210.0
            self.pat_r_target_ms = 215.0  # dPTT = +5 ms
            self.amp_ratio_l = 1.0
            self.amp_ratio_r = 1.0
            self.jitter_ms = 2.0
        elif preset_name == "Arterial Stiffness / Hypertension":
            self.pat_l_target_ms = 148.0
            self.pat_r_target_ms = 152.0  # dPTT = +4 ms, short PAT
            self.amp_ratio_l = 1.15
            self.amp_ratio_r = 1.15
            self.jitter_ms = 1.5
        elif preset_name == "Unilateral PAD (Left Arm Stenosis)":
            self.pat_l_target_ms = 265.0
            self.pat_r_target_ms = 205.0  # dPTT = -60 ms
            self.amp_ratio_l = 0.65       # Damped left pulse
            self.amp_ratio_r = 1.0
            self.jitter_ms = 2.5
        elif preset_name == "Unilateral PAD (Right Arm Stenosis)":
            self.pat_l_target_ms = 205.0
            self.pat_r_target_ms = 265.0  # dPTT = +60 ms
            self.amp_ratio_l = 1.0
            self.amp_ratio_r = 0.65       # Damped right pulse
            self.jitter_ms = 2.5
        elif preset_name == "Respiratory Sinus Arrhythmia (RSA)":
            self.pat_l_target_ms = 212.0
            self.pat_r_target_ms = 219.0
            self.jitter_ms = 6.0          # Higher physiological variation
            self.amp_ratio_l = 1.0
            self.amp_ratio_r = 1.0

    def run_simulation(self, duration_s=30.0, target_fs=500.0,
                       pat_l_ms=None, pat_r_ms=None, jitter_ms=2.0,
                       disease_profile="Healthy Normal"):
        """
        Executes beat-by-beat phase-locked synchronization of real ECG and PPG waveforms.
        """
        self.target_fs = float(target_fs)
        self.duration_s = float(duration_s)
        self.jitter_ms = float(jitter_ms)
        self.disease_profile = disease_profile
        
        if pat_l_ms is not None:
            self.pat_l_target_ms = float(pat_l_ms)
        if pat_r_ms is not None:
            self.pat_r_target_ms = float(pat_r_ms)

        # Pick default files if not provided
        if not self.ecg_path or not os.path.exists(self.ecg_path):
            ecg_list = glob.glob(os.path.join(DEFAULT_ECG_DIR, "*500sps*.csv"))
            self.ecg_path = ecg_list[0] if ecg_list else glob.glob(os.path.join(DEFAULT_ECG_DIR, "*.csv"))[0]
            
        if not self.ppg_path or not os.path.exists(self.ppg_path):
            ppg_list = glob.glob(os.path.join(DEFAULT_PPG_DIR, "*LR_500Hz*.csv"))
            self.ppg_path = ppg_list[0] if ppg_list else glob.glob(os.path.join(DEFAULT_PPG_DIR, "*.csv"))[0]

        # 1. Load signals
        _, fs_ecg_raw, raw_ecg, raw_bioz = load_ecg_file(self.ecg_path)
        p0_df, p1_df, fs_ppg_raw, ppg_meta = load_bilateral_ppg_file(self.ppg_path)

        # Resample ECG to target_fs if needed
        if fs_ecg_raw != self.target_fs:
            n_ecg_target = int(len(raw_ecg) * (self.target_fs / fs_ecg_raw))
            raw_ecg = signal.resample(raw_ecg, n_ecg_target)
            raw_bioz = signal.resample(raw_bioz, n_ecg_target)
        
        # 2. Detect ECG R-peaks
        r_peaks_idx, ecg_filt = detect_ecg_r_peaks(raw_ecg, self.target_fs)
        
        # Truncate to desired duration
        max_samples = int(self.duration_s * self.target_fs)
        if len(raw_ecg) > max_samples:
            raw_ecg = raw_ecg[:max_samples]
            raw_bioz = raw_bioz[:max_samples]
            ecg_filt = ecg_filt[:max_samples]
            r_peaks_idx = r_peaks_idx[r_peaks_idx < max_samples]

        # 3. Process Bilateral PPG
        p0_ir = despike_and_interp(p0_df['ir'].values)
        p0_red = despike_and_interp(p0_df['red'].values)
        p1_ir = despike_and_interp(p1_df['ir'].values)
        p1_red = despike_and_interp(p1_df['red'].values)
        
        if fs_ppg_raw != self.target_fs:
            p0_ir = signal.resample(p0_ir, int(len(p0_ir) * (self.target_fs / fs_ppg_raw)))
            p0_red = signal.resample(p0_red, int(len(p0_red) * (self.target_fs / fs_ppg_raw)))
            p1_ir = signal.resample(p1_ir, int(len(p1_ir) * (self.target_fs / fs_ppg_raw)))
            p1_red = signal.resample(p1_red, int(len(p1_red) * (self.target_fs / fs_ppg_raw)))

        feet_0, peaks_0, filt_p0 = detect_ppg_fiducials(p0_ir, self.target_fs)
        feet_1, peaks_1, filt_p1 = detect_ppg_fiducials(p1_ir, self.target_fs)

        def extract_pulse_cycles(filt_sig, red_sig, ir_sig, feet, peaks):
            cycles = []
            for i in range(len(feet) - 1):
                f_cur = feet[i]
                f_next = feet[i+1]
                if f_next - f_cur > int(0.3 * self.target_fs) and f_next - f_cur < int(1.8 * self.target_fs):
                    cycles.append({
                        'filt': filt_sig[f_cur:f_next],
                        'red': red_sig[f_cur:f_next],
                        'ir': ir_sig[f_cur:f_next],
                        'len': f_next - f_cur
                    })
            return cycles

        cycles_0 = extract_pulse_cycles(filt_p0, p0_red, p0_ir, feet_0, peaks_0)
        cycles_1 = extract_pulse_cycles(filt_p1, p1_red, p1_ir, feet_1, peaks_1)

        if not cycles_0:
            cycles_0 = [{'filt': filt_p0[:int(self.target_fs)], 'red': p0_red[:int(self.target_fs)], 'ir': p0_ir[:int(self.target_fs)], 'len': int(self.target_fs)}]
        if not cycles_1:
            cycles_1 = [{'filt': filt_p1[:int(self.target_fs)], 'red': p1_red[:int(self.target_fs)], 'ir': p1_ir[:int(self.target_fs)], 'len': int(self.target_fs)}]

        # 4. Synthesize Synchronized Multi-Channel Acquisition
        total_len = len(raw_ecg)
        time_s = np.arange(total_len) / self.target_fs
        
        ppg_l_red = np.zeros(total_len)
        ppg_l_ir = np.zeros(total_len)
        ppg_l_filt = np.zeros(total_len)
        
        ppg_r_red = np.zeros(total_len)
        ppg_r_ir = np.zeros(total_len)
        ppg_r_filt = np.zeros(total_len)
        
        marker_r = np.zeros(total_len, dtype=int)
        marker_foot_l = np.zeros(total_len, dtype=int)
        marker_foot_r = np.zeros(total_len, dtype=int)

        dc_l_red = np.median(p0_red)
        dc_l_ir = np.median(p0_ir)
        dc_r_red = np.median(p1_red)
        dc_r_ir = np.median(p1_ir)

        amp_l = getattr(self, 'amp_ratio_l', 1.0)
        amp_r = getattr(self, 'amp_ratio_r', 1.0)

        beats_data = []
        np.random.seed(42)
        
        for b_idx in range(len(r_peaks_idx) - 1):
            r_curr = r_peaks_idx[b_idx]
            r_next = r_peaks_idx[b_idx + 1]
            rr_samples = r_next - r_curr
            rr_s = rr_samples / self.target_fs
            hr_bpm = 60.0 / rr_s

            marker_r[r_curr] = 1

            beat_jitter = np.random.normal(0, self.jitter_ms)
            inst_pat_l = self.pat_l_target_ms + beat_jitter
            inst_pat_r = self.pat_r_target_ms + beat_jitter + np.random.normal(0, 0.4)

            delay_l_samples = int((inst_pat_l / 1000.0) * self.target_fs)
            foot_l_idx = r_curr + delay_l_samples
            
            delay_r_samples = int((inst_pat_r / 1000.0) * self.target_fs)
            foot_r_idx = r_curr + delay_r_samples

            if foot_l_idx < total_len:
                marker_foot_l[foot_l_idx] = 1
            if foot_r_idx < total_len:
                marker_foot_r[foot_r_idx] = 1

            c0 = cycles_0[b_idx % len(cycles_0)]
            c1 = cycles_1[b_idx % len(cycles_1)]

            def warp_cycle(cyc, target_len, amp_scale=1.0):
                x_orig = np.linspace(0, 1, len(cyc['filt']))
                x_new = np.linspace(0, 1, target_len)
                f_interp = interp1d(x_orig, cyc['filt'], kind='cubic', fill_value='extrapolate')
                r_interp = interp1d(x_orig, cyc['red'] - np.mean(cyc['red']), kind='cubic', fill_value='extrapolate')
                i_interp = interp1d(x_orig, cyc['ir'] - np.mean(cyc['ir']), kind='cubic', fill_value='extrapolate')
                return (f_interp(x_new) * amp_scale,
                        r_interp(x_new) * amp_scale,
                        i_interp(x_new) * amp_scale)

            w_filt_l, w_red_l, w_ir_l = warp_cycle(c0, rr_samples, amp_scale=amp_l)
            end_l = min(total_len, foot_l_idx + rr_samples)
            n_copy_l = end_l - foot_l_idx
            if foot_l_idx >= 0 and n_copy_l > 0:
                ppg_l_filt[foot_l_idx:end_l] = w_filt_l[:n_copy_l]
                ppg_l_red[foot_l_idx:end_l] = dc_l_red + w_red_l[:n_copy_l]
                ppg_l_ir[foot_l_idx:end_l] = dc_l_ir + w_ir_l[:n_copy_l]

            w_filt_r, w_red_r, w_ir_r = warp_cycle(c1, rr_samples, amp_scale=amp_r)
            end_r = min(total_len, foot_r_idx + rr_samples)
            n_copy_r = end_r - foot_r_idx
            if foot_r_idx >= 0 and n_copy_r > 0:
                ppg_r_filt[foot_r_idx:end_r] = w_filt_r[:n_copy_r]
                ppg_r_red[foot_r_idx:end_r] = dc_r_red + w_red_r[:n_copy_r]
                ppg_r_ir[foot_r_idx:end_r] = dc_r_ir + w_ir_r[:n_copy_r]

            pk_l_idx = foot_l_idx + int(np.argmax(w_filt_l)) if foot_l_idx < total_len else 0
            pk_r_idx = foot_r_idx + int(np.argmax(w_filt_r)) if foot_r_idx < total_len else 0
            pat_peak_l = (pk_l_idx - r_curr) / self.target_fs * 1000.0
            pat_peak_r = (pk_r_idx - r_curr) / self.target_fs * 1000.0

            dptt_val = inst_pat_r - inst_pat_l
            if abs(dptt_val) > 35.0:
                status_tag = "Suspected Unilateral PAD / Vascular Asymmetry"
            elif inst_pat_l < 165.0:
                status_tag = "High Arterial Stiffness / Suspected Hypertension"
            else:
                status_tag = "Normal Physiological Transit"

            beats_data.append({
                'Beat_Number': b_idx + 1,
                'R_Peak_Time_s': round(r_curr / self.target_fs, 4),
                'RR_Interval_s': round(rr_s, 4),
                'Heart_Rate_BPM': round(hr_bpm, 1),
                'PAT_Left_ms': round(inst_pat_l, 2),
                'PAT_Right_ms': round(inst_pat_r, 2),
                'Bilateral_dPTT_ms': round(dptt_val, 2),
                'PAT_Peak_Left_ms': round(pat_peak_l, 2),
                'PAT_Peak_Right_ms': round(pat_peak_r, 2),
                'Selected_Mode': self.disease_profile,
                'Diagnostic_Status': status_tag
            })

        if len(r_peaks_idx) > 0:
            first_foot_l = r_peaks_idx[0] + int((self.pat_l_target_ms/1000.0)*self.target_fs)
            first_foot_r = r_peaks_idx[0] + int((self.pat_r_target_ms/1000.0)*self.target_fs)
            if first_foot_l > 0:
                ppg_l_red[:first_foot_l] = dc_l_red
                ppg_l_ir[:first_foot_l] = dc_l_ir
            if first_foot_r > 0:
                ppg_r_red[:first_foot_r] = dc_r_red
                ppg_r_ir[:first_foot_r] = dc_r_ir

        noise_ecg = np.random.normal(0, np.std(ecg_filt) * 0.02, total_len)
        noise_ppg = np.random.normal(0, np.std(ppg_l_filt) * 0.02, total_len)
        ppg_l_filt += noise_ppg
        ppg_r_filt += noise_ppg

        base_ms = 1791365717524
        timestamps_ms = base_ms + (time_s * 1000.0).astype(np.int64)
        t_us = (time_s * 1e6).astype(np.int64)

        df_combined = pd.DataFrame({
            'timestamp_ms': timestamps_ms,
            't_us': t_us,
            'time_s': np.round(time_s, 4),
            'ECG_raw': raw_ecg.astype(int),
            'ECG_filtered': np.round(ecg_filt + noise_ecg, 2),
            'CH2_BioZ': raw_bioz.astype(int),
            'PPG_L_RED': np.round(ppg_l_red, 1),
            'PPG_L_IR': np.round(ppg_l_ir, 1),
            'PPG_L_filtered': np.round(ppg_l_filt, 2),
            'PPG_R_RED': np.round(ppg_r_red, 1),
            'PPG_R_IR': np.round(ppg_r_ir, 1),
            'PPG_R_filtered': np.round(ppg_r_filt, 2),
            'R_peak_marker': marker_r,
            'PPG_L_foot_marker': marker_foot_l,
            'PPG_R_foot_marker': marker_foot_r
        })

        df_beats = pd.DataFrame(beats_data)
        mode_slug = self.get_mode_slug(self.disease_profile)
        
        return df_combined, df_beats, {
            'target_fs': self.target_fs,
            'duration_s': self.duration_s,
            'mean_hr': float(df_beats['Heart_Rate_BPM'].mean()) if len(df_beats) else 0.0,
            'mean_pat_l': float(df_beats['PAT_Left_ms'].mean()) if len(df_beats) else self.pat_l_target_ms,
            'mean_pat_r': float(df_beats['PAT_Right_ms'].mean()) if len(df_beats) else self.pat_r_target_ms,
            'mean_dptt': float(df_beats['Bilateral_dPTT_ms'].mean()) if len(df_beats) else (self.pat_r_target_ms - self.pat_l_target_ms),
            'num_beats': len(df_beats),
            'disease_profile': self.disease_profile,
            'mode_slug': mode_slug,
            'ecg_source': os.path.basename(self.ecg_path),
            'ppg_source': os.path.basename(self.ppg_path)
        }

    def generate_waveform_plot(self, df_combined, df_beats, summary, out_png_path, zoom_s=3.0):
        """
        Creates a diagnostic multi-panel visualization with PAT/PTT calipers.
        """
        fig = plt.figure(figsize=(15, 11), dpi=150)
        gs = fig.add_gridspec(4, 2, width_ratios=[2.2, 1.2], hspace=0.35, wspace=0.22)

        time_s = df_combined['time_s'].values

        if len(df_beats) >= 3:
            zoom_start = max(0, df_beats.iloc[1]['R_Peak_Time_s'] - 0.15)
        else:
            zoom_start = 1.0
        zoom_end = zoom_start + zoom_s
        zoom_mask = (time_s >= zoom_start) & (time_s <= zoom_end)
        z_t = time_s[zoom_mask]

        c_ecg = '#1f77b4'
        c_ppg_l = '#2ca02c'
        c_ppg_r = '#d62728'
        c_caliper = '#ff7f0e'
        c_grid = '#e0e0e0'

        # Panel 1: Full ECG
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(time_s, df_combined['ECG_filtered'], color=c_ecg, lw=1.0, label='ECG Lead II (0.5-40 Hz)')
        r_times = df_beats['R_Peak_Time_s'].values
        r_y = np.interp(r_times, time_s, df_combined['ECG_filtered'])
        ax0.plot(r_times, r_y, 'rv', ms=6, label='R-peaks')
        ax0.axvspan(zoom_start, zoom_end, color='gold', alpha=0.22, label='Zoom Detail Window')
        ax0.set_ylabel('ECG (counts)', fontsize=9, fontweight='bold')
        ax0.set_title(f"ECG Signal - ADS1292R [{summary['ecg_source']}]", fontsize=10, fontweight='bold', loc='left')
        ax0.grid(True, ls=':', color=c_grid)
        ax0.legend(loc='upper right', fontsize=8, framealpha=0.85)

        # Panel 2: Left PPG
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax1.plot(time_s, df_combined['PPG_L_filtered'], color=c_ppg_l, lw=1.1, label='PPG Left Foot/Peak (Ch 0)')
        foot_l_times = df_beats['R_Peak_Time_s'].values + (df_beats['PAT_Left_ms'].values / 1000.0)
        foot_l_y = np.interp(foot_l_times, time_s, df_combined['PPG_L_filtered'])
        ax1.plot(foot_l_times, foot_l_y, '^', color='darkgreen', ms=6, label=f"Left Foot (PAT_L ~ {summary['mean_pat_l']:.1f} ms)")
        ax1.axvspan(zoom_start, zoom_end, color='gold', alpha=0.22)
        ax1.set_ylabel('Left PPG AC', fontsize=9, fontweight='bold')
        ax1.set_title(f"Bilateral PPG Channel 0 (LEFT) - AFE4490 [{summary['ppg_source']}]", fontsize=10, fontweight='bold', loc='left')
        ax1.grid(True, ls=':', color=c_grid)
        ax1.legend(loc='upper right', fontsize=8, framealpha=0.85)

        # Panel 3: Right PPG
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        ax2.plot(time_s, df_combined['PPG_R_filtered'], color=c_ppg_r, lw=1.1, label='PPG Right Foot/Peak (Ch 1)')
        foot_r_times = df_beats['R_Peak_Time_s'].values + (df_beats['PAT_Right_ms'].values / 1000.0)
        foot_r_y = np.interp(foot_r_times, time_s, df_combined['PPG_R_filtered'])
        ax2.plot(foot_r_times, foot_r_y, '^', color='darkred', ms=6, label=f"Right Foot (PAT_R ~ {summary['mean_pat_r']:.1f} ms)")
        ax2.axvspan(zoom_start, zoom_end, color='gold', alpha=0.22)
        ax2.set_ylabel('Right PPG AC', fontsize=9, fontweight='bold')
        ax2.set_title(f"Bilateral PPG Channel 1 (RIGHT) - AFE4490 [{summary['ppg_source']}]", fontsize=10, fontweight='bold', loc='left')
        ax2.grid(True, ls=':', color=c_grid)
        ax2.legend(loc='upper right', fontsize=8, framealpha=0.85)

        # Panel 4: Superimposed Overlay
        ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
        norm_ecg = df_combined['ECG_filtered'] / (np.max(np.abs(df_combined['ECG_filtered'])) + 1e-6)
        norm_l = df_combined['PPG_L_filtered'] / (np.max(np.abs(df_combined['PPG_L_filtered'])) + 1e-6)
        norm_r = df_combined['PPG_R_filtered'] / (np.max(np.abs(df_combined['PPG_R_filtered'])) + 1e-6)
        ax3.plot(time_s, norm_ecg, color=c_ecg, lw=0.9, alpha=0.8, label='Normalized ECG')
        ax3.plot(time_s, norm_l, color=c_ppg_l, lw=1.0, alpha=0.9, label='Normalized Left PPG')
        ax3.plot(time_s, norm_r, color=c_ppg_r, lw=1.0, alpha=0.9, label='Normalized Right PPG')
        ax3.set_xlabel('Recording Time (seconds)', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Norm. Amp', fontsize=9, fontweight='bold')
        ax3.set_title(f"Superimposed View - {summary['disease_profile']}", fontsize=10, fontweight='bold', loc='left')
        ax3.grid(True, ls=':', color=c_grid)
        ax3.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.85)

        # Right Column: Zoomed Detail with Calipers
        ax_zoom = fig.add_subplot(gs[0:2, 1])
        ax_zoom.set_title("Synchronized Cardiac Cycle (PAT / PTT Calipers)", fontsize=11, fontweight='bold', pad=10)
        
        z_ecg = norm_ecg[zoom_mask]
        z_ppg_l = norm_l[zoom_mask]
        z_ppg_r = norm_r[zoom_mask]
        ax_zoom.plot(z_t, z_ecg, color=c_ecg, lw=1.5, label='ECG')
        ax_zoom.plot(z_t, z_ppg_l, color=c_ppg_l, lw=1.8, label='PPG Left')
        ax_zoom.plot(z_t, z_ppg_r, color=c_ppg_r, lw=1.8, ls='--', label='PPG Right')

        sub_beats = df_beats[(df_beats['R_Peak_Time_s'] >= zoom_start) & (df_beats['R_Peak_Time_s'] <= zoom_end - 0.5)]
        if len(sub_beats) > 0:
            target_beat = sub_beats.iloc[0]
            t_r = target_beat['R_Peak_Time_s']
            pat_l = target_beat['PAT_Left_ms']
            pat_r = target_beat['PAT_Right_ms']
            dptt = target_beat['Bilateral_dPTT_ms']
            t_fl = t_r + (pat_l / 1000.0)
            t_fr = t_r + (pat_r / 1000.0)

            ax_zoom.axvline(t_r, color=c_ecg, ls=':', lw=1.5)
            ax_zoom.axvline(t_fl, color=c_ppg_l, ls=':', lw=1.5)
            ax_zoom.axvline(t_fr, color=c_ppg_r, ls=':', lw=1.5)

            y_pat_l = 0.85
            y_pat_r = 0.65
            y_dptt = 0.40

            ax_zoom.annotate(f"PAT_Left = {pat_l:.1f} ms",
                             xy=(t_fl, y_pat_l), xytext=(t_r, y_pat_l),
                             xycoords='data', textcoords='data',
                             arrowprops=dict(arrowstyle='<->', color=c_ppg_l, lw=2.0),
                             fontsize=9, fontweight='bold', color='darkgreen',
                             ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=c_ppg_l, alpha=0.9))

            ax_zoom.annotate(f"PAT_Right = {pat_r:.1f} ms",
                             xy=(t_fr, y_pat_r), xytext=(t_r, y_pat_r),
                             xycoords='data', textcoords='data',
                             arrowprops=dict(arrowstyle='<->', color=c_ppg_r, lw=2.0),
                             fontsize=9, fontweight='bold', color='darkred',
                             ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=c_ppg_r, alpha=0.9))

            ax_zoom.annotate(f"dPTT = {abs(dptt):.1f} ms",
                             xy=(t_fr, y_dptt), xytext=(t_fl, y_dptt),
                             xycoords='data', textcoords='data',
                             arrowprops=dict(arrowstyle='<->', color=c_caliper, lw=2.2),
                             fontsize=9, fontweight='bold', color='chocolate',
                             ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=c_caliper, alpha=0.9))

        ax_zoom.set_xlim(zoom_start, zoom_end)
        ax_zoom.set_xlabel('Time (s)', fontsize=9, fontweight='bold')
        ax_zoom.set_ylabel('Normalized Amplitude', fontsize=9, fontweight='bold')
        ax_zoom.grid(True, ls=':', color=c_grid)
        ax_zoom.legend(loc='lower left', fontsize=8)

        # Panel 5: Beat-to-Beat PAT Dynamics
        ax_dyn = fig.add_subplot(gs[2, 1])
        b_nums = df_beats['Beat_Number'].values
        ax_dyn.plot(b_nums, df_beats['PAT_Left_ms'], 'o-', color=c_ppg_l, ms=4, lw=1.2, label='PAT Left')
        ax_dyn.plot(b_nums, df_beats['PAT_Right_ms'], 's-', color=c_ppg_r, ms=4, lw=1.2, label='PAT Right')
        ax_dyn.set_xlabel('Beat Index #', fontsize=9, fontweight='bold')
        ax_dyn.set_ylabel('PAT (ms)', fontsize=9, fontweight='bold')
        ax_dyn.set_title('Beat-to-Beat Pulse Arrival Time Tracking', fontsize=10, fontweight='bold', loc='left')
        ax_dyn.grid(True, ls=':', color=c_grid)
        ax_dyn.legend(loc='upper right', fontsize=8)

        # Panel 6: Diagnostic Summary Card
        ax_card = fig.add_subplot(gs[3, 1])
        ax_card.axis('off')

        dptt_val = abs(summary['mean_dptt'])
        if dptt_val > 35.0:
            diag_status = "CRITICAL / ASYMMETRIC (Suspected Unilateral PAD / Stenosis)"
            diag_color = '#d62728'
        elif summary['mean_pat_l'] < 165.0:
            diag_status = "ELEVATED ARTERIAL STIFFNESS (Short PAT / Suspected Hypertension)"
            diag_color = '#e65100'
        else:
            diag_status = "PHYSIOLOGICAL NORMAL (Symmetric Bilateral Transit)"
            diag_color = '#2e7d32'

        card_text = (
            f"DIAGNOSTIC SUMMARY & SIMULATION METRICS\n"
            f"--------------------------------------------------\n"
            f"• Target Sampling Frequency : {summary['target_fs']:.0f} Hz (dt = {1000.0/summary['target_fs']:.1f} ms)\n"
            f"• Selected Mode            : {summary['disease_profile']}\n"
            f"• Recording Duration       : {summary['duration_s']:.1f} s ({summary['num_beats']} synchronized beats)\n"
            f"• Mean Heart Rate          : {summary['mean_hr']:.1f} bpm\n"
            f"• Mean PAT (Left Limb)     : {summary['mean_pat_l']:.1f} ms\n"
            f"• Mean PAT (Right Limb)    : {summary['mean_pat_r']:.1f} ms\n"
            f"• Bilateral Transit (dPTT) : {summary['mean_dptt']:+.1f} ms (|Δ| = {dptt_val:.1f} ms)\n"
            f"--------------------------------------------------\n"
            f"ASSESSMENT:\n{diag_status}"
        )

        ax_card.text(0.02, 0.95, card_text, fontsize=9.5, va='top', family='monospace',
                     bbox=dict(boxstyle='square,pad=0.6', facecolor='#f8f9fa', edgecolor=diag_color, lw=2))

        fig.suptitle(f"Cardisense Synchronized ECG + Bilateral PPG Acquisition Simulation\n"
                     f"Mode: {summary['disease_profile']} | PAT_L: {summary['mean_pat_l']:.1f} ms | PAT_R: {summary['mean_pat_r']:.1f} ms | dPTT: {summary['mean_dptt']:+.1f} ms",
                     fontsize=12, fontweight='bold', y=0.99)

        Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png_path

    def export_csvs(self, df_combined, df_beats, summary, output_dir=None):
        """
        Exports:
        1. Mode-specific clean timings CSV (concise beat-by-beat PAT/PTT metrics).
        2. Mode-specific compact waveform CSV (3 to 5 cardiac cycles ~1500 samples).
        3. Full continuous raw acquisition stream CSV.
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        mode_slug = summary.get('mode_slug', self.get_mode_slug(summary['disease_profile']))
        
        # 1. Mode-Specific Timings CSV (The primary clean deliverable!)
        mode_timings_csv = os.path.join(output_dir, f"cardisense_{mode_slug}_timings.csv")
        with open(mode_timings_csv, 'w', encoding='utf-8', newline='') as f:
            f.write("# =========================================================================\n")
            f.write(f"# Cardisense Mode Timings & Clinical Metrics : {summary['disease_profile']}\n")
            f.write(f"# Mean Heart Rate       : {summary['mean_hr']:.1f} bpm\n")
            f.write(f"# Mean PAT Left (Ch 0)  : {summary['mean_pat_l']:.1f} ms\n")
            f.write(f"# Mean PAT Right (Ch 1) : {summary['mean_pat_r']:.1f} ms\n")
            f.write(f"# Mean Bilateral dPTT   : {summary['mean_dptt']:+.1f} ms\n")
            f.write(f"# Total Analyzed Beats  : {summary['num_beats']}\n")
            f.write("# Columns: Beat_Number, R_Peak_Time_s, RR_Interval_s, Heart_Rate_BPM,\n")
            f.write("#          PAT_Left_ms, PAT_Right_ms, Bilateral_dPTT_ms, PAT_Peak_Left_ms,\n")
            f.write("#          PAT_Peak_Right_ms, Selected_Mode, Diagnostic_Status\n")
            f.write("# =========================================================================\n")
            df_beats.to_csv(f, index=False, lineterminator='\n')

        # Also maintain standard filename for generic references
        generic_beats_csv = os.path.join(output_dir, "cardisense_timing_beats_pat_ptt.csv")
        df_beats.to_csv(generic_beats_csv, index=False, lineterminator='\n')

        # 2. Mode-Specific Compact Waveform CSV (clean window of ~3-4 cardiac cycles, ~1500 samples)
        mode_compact_csv = os.path.join(output_dir, f"cardisense_{mode_slug}_compact_waveform.csv")
        compact_samples = min(len(df_combined), int(3.5 * summary['target_fs']))
        df_compact = df_combined.iloc[:compact_samples][['time_s', 'ECG_filtered', 'PPG_L_filtered', 'PPG_R_filtered', 'R_peak_marker', 'PPG_L_foot_marker', 'PPG_R_foot_marker']]
        with open(mode_compact_csv, 'w', encoding='utf-8', newline='') as f:
            f.write("# =========================================================================\n")
            f.write(f"# Cardisense Compact Waveform Window (3-4 Cardiac Cycles) : {summary['disease_profile']}\n")
            f.write(f"# Sampling Frequency : {summary['target_fs']:.0f} Hz | Window: {compact_samples/summary['target_fs']:.2f} s ({compact_samples} samples)\n")
            f.write("# =========================================================================\n")
            df_compact.to_csv(f, index=False, lineterminator='\n')

        # 3. Full Continuous Acquisition CSV (complete multi-channel stream)
        comb_csv_path = os.path.join(output_dir, "cardisense_synchronized_acquisition.csv")
        with open(comb_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write("# =========================================================================\n")
            f.write("# Cardisense Multi-Modal Biometric Acquisition System - Simulation Record\n")
            f.write(f"# Sampling Frequency : {summary['target_fs']:.1f} Hz\n")
            f.write(f"# Simulated Profile  : {summary['disease_profile']}\n")
            f.write(f"# Mean Heart Rate    : {summary['mean_hr']:.1f} bpm\n")
            f.write(f"# Target PAT Left    : {summary['mean_pat_l']:.1f} ms\n")
            f.write(f"# Target PAT Right   : {summary['mean_pat_r']:.1f} ms\n")
            f.write(f"# Target Bilateral dPTT : {summary['mean_dptt']:+.1f} ms\n")
            f.write(f"# Source ECG File    : {summary['ecg_source']}\n")
            f.write(f"# Source PPG File    : {summary['ppg_source']}\n")
            f.write("# =========================================================================\n")
            df_combined.to_csv(f, index=False, lineterminator='\n')

        return {
            'mode_timings_csv': mode_timings_csv,
            'mode_compact_csv': mode_compact_csv,
            'full_acquisition_csv': comb_csv_path
        }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cardisense ECG & Bilateral PPG Simulator")
    parser.add_argument("--ecg", type=str, default=None, help="Path to ECG CSV file")
    parser.add_argument("--ppg", type=str, default=None, help="Path to Bilateral PPG CSV file")
    parser.add_argument("--pat_l", type=float, default=210.0, help="Target PAT Left in ms (default: 210)")
    parser.add_argument("--pat_r", type=float, default=218.0, help="Target PAT Right in ms (default: 218)")
    parser.add_argument("--fs", type=float, default=500.0, help="Simulation sampling frequency in Hz (default: 500)")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds (default: 30)")
    parser.add_argument("--jitter", type=float, default=2.0, help="Beat jitter in ms (default: 2.0)")
    parser.add_argument("--profile", type=str, default="Healthy Normal",
                        choices=["Healthy Normal", "Arterial Stiffness / Hypertension",
                                 "Unilateral PAD (Left Arm Stenosis)", "Unilateral PAD (Right Arm Stenosis)",
                                 "Respiratory Sinus Arrhythmia (RSA)"],
                        help="Clinical simulation profile preset")
    parser.add_argument("--outdir", type=str, default=OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()

    sim = CardisenseSimulator(ecg_path=args.ecg, ppg_path=args.ppg)
    if args.profile:
        sim.set_disease_preset(args.profile)
        
    print("=" * 70)
    print("CARDISENSE SYNCHRONIZATION & PAT/PTT SIMULATION ENGINE")
    print("=" * 70)
    print(f"Profile: {args.profile}")
    print(f"Target PAT_Left: {args.pat_l} ms | Target PAT_Right: {args.pat_r} ms")
    print(f"Duration: {args.duration} s | Target Fs: {args.fs} Hz")

    df_comb, df_beats, summary = sim.run_simulation(
        duration_s=args.duration,
        target_fs=args.fs,
        pat_l_ms=args.pat_l,
        pat_r_ms=args.pat_r,
        jitter_ms=args.jitter,
        disease_profile=args.profile
    )

    out_png = os.path.join(args.outdir, "cardisense_waveform_simulation.png")
    sim.generate_waveform_plot(df_comb, df_beats, summary, out_png)
    files = sim.export_csvs(df_comb, df_beats, summary, args.outdir)

    print("\nSimulation Completed Successfully!")
    print(f"-> Waveform Visual Preview : {out_png}")
    print(f"-> Mode Timings CSV        : {files['mode_timings_csv']}")
    print(f"-> Mode Compact Waveform   : {files['mode_compact_csv']}")
    print(f"-> Full Acquisition CSV    : {files['full_acquisition_csv']}")
    print(f"Synchronized {summary['num_beats']} beats. Mean HR: {summary['mean_hr']:.1f} bpm.")
    print("=" * 70)

if __name__ == "__main__":
    main()
