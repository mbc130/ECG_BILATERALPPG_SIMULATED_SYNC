# Cardisense Synchronized ECG + Bilateral PPG Acquisition Simulator

A complete simulation, synchronization, and diagnostic modeling suite for the **Cardisense** project.

This tool bridges the gap before physical hardware clock synchronization is deployed by taking standalone ECG recordings (`ADS1292R`) and standalone bilateral PPG recordings (`AFE4490 dual`), synchronizing them with physiological phase-locked precision, and exporting the exact combined acquisition CSV and beat-by-beat PAT/PTT timings.

---

## 🌟 Key Features

1. **Automatic Discovery & Pairing**:
   - Automatically discovers all ECG recordings in `Prototype\ADS1292r\ecg_ads1292\stm32_ecg\ecg_ads1292_stm32f411_arduino\new raw sps` (125, 250, 500, 1000, 2000 sps).
   - Automatically discovers all bilateral PPG recordings in `Prototype\AFE4490\Code\afe4490_dual` (500 Hz, 1000 Hz, dual-channel Red/IR).

2. **Phase-Locked Fiducial Synchronization**:
   - ECG R-peak detection using Pan-Tompkins derivative-energy method with parabolic refinement.
   - PPG systolic foot (onset) and systolic peak detection on both Left and Right channels.
   - Physiological cycle morphing: matches each real recorded bilateral PPG pulse to each real ECG cardiac cycle with microsecond timing.

3. **PAT & PTT Diagnostic Modeling**:
   - **PAT (Pulse Arrival Time)**: Time from ventricular electrical activation (ECG R-peak) to peripheral pulse foot:
     $$\text{PAT}_L = t_{\text{foot}, L} - t_R$$
     $$\text{PAT}_R = t_{\text{foot}, R} - t_R$$
   - **dPTT / Bilateral Transit Difference**:
     $$\Delta\text{PAT} = \text{PAT}_R - \text{PAT}_L$$
   - Pre-configured Clinical Profiles:
     - **Healthy Normal**: $\text{PAT}_L \approx 210$ ms, $\text{PAT}_R \approx 218$ ms ($\text{dPTT} \approx +8$ ms), normal dicrotic notch.
     - **Arterial Stiffness / Hypertension**: Shortened PAT ($\approx 148$ ms, fast Pulse Wave Velocity), elevated reflection.
     - **Unilateral PAD (Left Arm Stenosis)**: Severe delay on left limb ($\text{PAT}_L \approx 265$ ms, $\text{dPTT} \approx -60$ ms) and damped amplitude.
     - **Unilateral PAD (Right Arm Stenosis)**: Severe delay on right limb ($\text{PAT}_R \approx 265$ ms, $\text{dPTT} \approx +60$ ms) and damped amplitude.
     - **Respiratory Sinus Arrhythmia (RSA)**: Increased physiological beat-to-beat variability and jitter.

4. **Multi-Interface Options**:
   - **Interactive Web Dashboard**: Modern browser interface with live sliders, instant waveform preview, caliper markings, metric cards, and 1-click downloads.
   - **Native Desktop GUI**: Standalone Tkinter application with embedded interactive Matplotlib plot.
   - **Command-Line Interface (CLI)**: Automate batch simulation runs directly from PowerShell or scripts.

5. **Standardized Clinical Outputs**:
   - `cardisense_synchronized_acquisition.csv`: The continuous multi-modal acquisition stream formatted identically to a unified DAQ board.
   - `cardisense_timing_beats_pat_ptt.csv`: Beat-by-beat fiducial timestamps, instantaneous PAT_L, PAT_R, dPTT, HR, and diagnosis classification tags.
   - `cardisense_waveform_simulation.png`: Multi-panel diagnostic visualization showing full traces, zoom detail with PAT calipers, and PAT dynamics.

---

## 🚀 How to Run

### Option 1: Web Dashboard (Recommended)
Double-click `run_web_simulator.bat` or run:
```bash
python web_dashboard.py
```
Open your browser at `http://localhost:8050`.

### Option 2: Native Desktop GUI
Double-click `run_desktop_gui.bat` or run:
```bash
python desktop_gui.py
```

### Option 3: Command Line (CLI)
```bash
python cardisense_simulator.py --pat_l 210 --pat_r 220 --duration 30 --fs 500 --profile "Healthy Normal"
```

---

## 📂 Output File Formats

### 1. `cardisense_synchronized_acquisition.csv`
Continuous multi-channel stream:
- `timestamp_ms`: Continuous absolute timestamp (ms)
- `t_us`: Microsecond counter
- `time_s`: Elapsed time in seconds
- `ECG_raw`: Raw 24-bit ADC counts from ADS1292R
- `ECG_filtered`: Filtered ECG (0.5 - 40 Hz)
- `CH2_BioZ`: Bioimpedance channel from ADS1292R
- `PPG_L_RED`: Left Hand Red photodiode ADC
- `PPG_L_IR`: Left Hand Infrared photodiode ADC
- `PPG_L_filtered`: Filtered Left AC pulse
- `PPG_R_RED`: Right Hand Red photodiode ADC
- `PPG_R_IR`: Right Hand Infrared photodiode ADC
- `PPG_R_filtered`: Filtered Right AC pulse
- `R_peak_marker`: `1` at R-peak, `0` elsewhere
- `PPG_L_foot_marker`: `1` at Left pulse foot, `0` elsewhere
- `PPG_R_foot_marker`: `1` at Right pulse foot, `0` elsewhere

### 2. `cardisense_timing_beats_pat_ptt.csv`
Beat-by-beat diagnostic measurements:
- `beat_num`: Cardiac cycle number
- `t_R_s`: ECG R-peak time (s)
- `RR_s`: Beat-to-beat RR interval (s)
- `HR_bpm`: Instantaneous heart rate (bpm)
- `t_foot_L_s`: Left PPG foot arrival time (s)
- `t_foot_R_s`: Right PPG foot arrival time (s)
- `PAT_L_ms`: Left Pulse Arrival Time (ms)
- `PAT_R_ms`: Right Pulse Arrival Time (ms)
- `dPTT_ms`: Bilateral pulse transit difference (ms)
- `t_peak_L_s`: Left systolic peak time (s)
- `t_peak_R_s`: Right systolic peak time (s)
- `PAT_peak_L_ms`: Left R-to-peak interval (ms)
- `PAT_peak_R_ms`: Right R-to-peak interval (ms)
- `disease_flag`: Simulation condition
