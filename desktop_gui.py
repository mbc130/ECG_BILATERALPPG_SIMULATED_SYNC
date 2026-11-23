"""
desktop_gui.py
--------------
Native Tkinter Desktop GUI for Cardisense ECG + Bilateral PPG Acquisition Simulator.
Runs without browser or external web server.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

from cardisense_simulator import CardisenseSimulator, OUTPUT_DIR

class CardisenseDesktopGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cardisense - Synchronized ECG & Bilateral PPG Simulator")
        self.root.geometry("1400x900")
        self.root.configure(bg="#0f172a")

        self.simulator = CardisenseSimulator()
        self.files = CardisenseSimulator.get_available_files()

        self._init_styles()
        self._build_layout()
        self._load_defaults()

    def _init_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#1e293b", foreground="#f8fafc", font=("Inter", 9))
        style.configure("TFrame", background="#1e293b")
        style.configure("TLabelframe", background="#1e293b", foreground="#38bdf8", font=("Inter", 10, "bold"))
        style.configure("TLabelframe.Label", background="#1e293b", foreground="#38bdf8")
        style.configure("TButton", font=("Inter", 9, "bold"), background="#0284c7", foreground="#ffffff")
        style.map("TButton", background=[("active", "#0369a1")])

    def _build_layout(self):
        # Top Header
        header = tk.Frame(self.root, bg="#0b0f19", height=50, padx=20, pady=10)
        header.pack(fill="x", side="top")
        
        lbl_brand = tk.Label(header, text="CARDISENSE SYNCHRONIZER", font=("Inter", 14, "bold"), bg="#0b0f19", fg="#38bdf8")
        lbl_brand.pack(side="left")
        
        lbl_sub = tk.Label(header, text="Multi-Modal Acquisition Simulator & PAT/PTT Diagnostic Workstation", font=("Inter", 10), bg="#0b0f19", fg="#94a3b8")
        lbl_sub.pack(side="left", padx=15)

        # Main Split
        main_frame = tk.Frame(self.root, bg="#0f172a", padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)

        # Left Control Panel
        left_panel = tk.Frame(main_frame, bg="#1e293b", width=340, padx=12, pady=12, relief="ridge", bd=1)
        left_panel.pack(side="left", fill="y", padx=(0, 10))

        # Controls in Left Panel
        tk.Label(left_panel, text="SIMULATION PARAMETERS", font=("Inter", 11, "bold"), bg="#1e293b", fg="#38bdf8").pack(anchor="w", pady=(0, 10))

        # Profile selection
        tk.Label(left_panel, text="Clinical Preset Profile:", bg="#1e293b", fg="#cbd5e1").pack(anchor="w")
        self.var_profile = tk.StringVar(value="Healthy Normal")
        profiles = [
            "Healthy Normal",
            "Arterial Stiffness / Hypertension",
            "Unilateral PAD (Left Arm Stenosis)",
            "Unilateral PAD (Right Arm Stenosis)",
            "Respiratory Sinus Arrhythmia (RSA)"
        ]
        cb_profile = ttk.Combobox(left_panel, textvariable=self.var_profile, values=profiles, state="readonly", width=32)
        cb_profile.pack(fill="x", pady=(2, 10))
        cb_profile.bind("<<ComboboxSelected>>", self._on_preset_change)

        # ECG File
        tk.Label(left_panel, text="ECG File (ADS1292R):", bg="#1e293b", fg="#cbd5e1").pack(anchor="w")
        self.var_ecg = tk.StringVar()
        ecg_names = [os.path.basename(f) for f in self.files["ecg"]]
        self.cb_ecg = ttk.Combobox(left_panel, textvariable=self.var_ecg, values=ecg_names, state="readonly", width=32)
        self.cb_ecg.pack(fill="x", pady=(2, 10))

        # PPG File
        tk.Label(left_panel, text="Bilateral PPG File (AFE4490):", bg="#1e293b", fg="#cbd5e1").pack(anchor="w")
        self.var_ppg = tk.StringVar()
        ppg_names = [os.path.basename(f) for f in self.files["ppg"]]
        self.cb_ppg = ttk.Combobox(left_panel, textvariable=self.var_ppg, values=ppg_names, state="readonly", width=32)
        self.cb_ppg.pack(fill="x", pady=(2, 10))

        # Target Sampling Rate
        tk.Label(left_panel, text="Sampling Rate (Hz):", bg="#1e293b", fg="#cbd5e1").pack(anchor="w")
        self.var_fs = tk.StringVar(value="500")
        ttk.Combobox(left_panel, textvariable=self.var_fs, values=["125", "250", "500", "1000"], state="readonly").pack(fill="x", pady=(2, 10))

        # Duration
        tk.Label(left_panel, text="Duration (seconds):", bg="#1e293b", fg="#cbd5e1").pack(anchor="w")
        self.scale_dur = tk.Scale(left_panel, from_=10, to=60, orient="horizontal", bg="#1e293b", fg="#f8fafc", highlightthickness=0)
        self.scale_dur.set(30)
        self.scale_dur.pack(fill="x", pady=(2, 10))

        # PAT Left
        tk.Label(left_panel, text="PAT Left Arm (ms):", bg="#1e293b", fg="#22c55e", font=("Inter", 9, "bold")).pack(anchor="w")
        self.scale_pat_l = tk.Scale(left_panel, from_=120, to=320, orient="horizontal", bg="#1e293b", fg="#22c55e", highlightthickness=0, command=self._update_dptt_label)
        self.scale_pat_l.set(210)
        self.scale_pat_l.pack(fill="x", pady=(2, 6))

        # PAT Right
        tk.Label(left_panel, text="PAT Right Arm (ms):", bg="#1e293b", fg="#f43f5e", font=("Inter", 9, "bold")).pack(anchor="w")
        self.scale_pat_r = tk.Scale(left_panel, from_=120, to=320, orient="horizontal", bg="#1e293b", fg="#f43f5e", highlightthickness=0, command=self._update_dptt_label)
        self.scale_pat_r.set(218)
        self.scale_pat_r.pack(fill="x", pady=(2, 6))

        # dPTT display
        self.lbl_dptt_val = tk.Label(left_panel, text="Bilateral dPTT: +8 ms", font=("Inter", 10, "bold"), bg="#1e293b", fg="#fbbf24")
        self.lbl_dptt_val.pack(anchor="w", pady=(0, 10))

        # Run Button
        btn_run = tk.Button(left_panel, text="RUN SIMULATION", font=("Inter", 11, "bold"), bg="#0284c7", fg="white", activebackground="#0369a1", activeforeground="white", pady=10, relief="flat", command=self.run_simulation)
        btn_run.pack(fill="x", pady=(10, 15))

        # Export Buttons
        tk.Label(left_panel, text="EXPORT SYNCHRONIZED FILES:", font=("Inter", 9, "bold"), bg="#1e293b", fg="#94a3b8").pack(anchor="w", pady=(10, 4))
        
        btn_csv_comb = tk.Button(left_panel, text="Save Combined Acquisition CSV", bg="#0f172a", fg="#38bdf8", relief="groove", command=self.save_combined_csv)
        btn_csv_comb.pack(fill="x", pady=3)

        btn_csv_beats = tk.Button(left_panel, text="Save Beat Timings CSV", bg="#0f172a", fg="#38bdf8", relief="groove", command=self.save_beats_csv)
        btn_csv_beats.pack(fill="x", pady=3)

        # Right Panel: Visualization Canvas
        right_panel = tk.Frame(main_frame, bg="#0b0f19", relief="ridge", bd=1)
        right_panel.pack(side="right", fill="both", expand=True)

        # Canvas Frame
        self.canvas_frame = tk.Frame(right_panel, bg="#0b0f19")
        self.canvas_frame.pack(fill="both", expand=True)

        self.fig = plt.figure(figsize=(10, 7), dpi=100)
        self.fig.patch.set_facecolor('#0f172a')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        self.toolbar.config(background="#1e293b")
        self.toolbar.update()

    def _load_defaults(self):
        # Select 500sps by default if available
        for f in self.files["ecg"]:
            if "500sps" in f:
                self.var_ecg.set(os.path.basename(f))
                break
        if not self.var_ecg.get() and self.files["ecg"]:
            self.var_ecg.set(os.path.basename(self.files["ecg"][0]))

        for f in self.files["ppg"]:
            if "500Hz" in f:
                self.var_ppg.set(os.path.basename(f))
                break
        if not self.var_ppg.get() and self.files["ppg"]:
            self.var_ppg.set(os.path.basename(self.files["ppg"][0]))

        self.run_simulation()

    def _update_dptt_label(self, *args):
        pl = self.scale_pat_l.get()
        pr = self.scale_pat_r.get()
        diff = pr - pl
        self.lbl_dptt_val.config(text=f"Bilateral dPTT: {diff:+} ms")

    def _on_preset_change(self, event=None):
        preset = self.var_profile.get()
        if preset == "Healthy Normal":
            self.scale_pat_l.set(210)
            self.scale_pat_r.set(218)
        elif preset == "Arterial Stiffness / Hypertension":
            self.scale_pat_l.set(148)
            self.scale_pat_r.set(152)
        elif preset == "Unilateral PAD (Left Arm Stenosis)":
            self.scale_pat_l.set(265)
            self.scale_pat_r.set(205)
        elif preset == "Unilateral PAD (Right Arm Stenosis)":
            self.scale_pat_l.set(205)
            self.scale_pat_r.set(265)
        self._update_dptt_label()

    def run_simulation(self):
        ecg_name = self.var_ecg.get()
        ppg_name = self.var_ppg.get()

        ecg_full = [f for f in self.files["ecg"] if os.path.basename(f) == ecg_name]
        ppg_full = [f for f in self.files["ppg"] if os.path.basename(f) == ppg_name]

        ecg_path = ecg_full[0] if ecg_full else None
        ppg_path = ppg_full[0] if ppg_full else None

        self.simulator = CardisenseSimulator(ecg_path=ecg_path, ppg_path=ppg_path)
        self.simulator.set_disease_preset(self.var_profile.get())

        target_fs = float(self.var_fs.get())
        dur = float(self.scale_dur.get())
        pat_l = float(self.scale_pat_l.get())
        pat_r = float(self.scale_pat_r.get())

        df_comb, df_beats, summary = self.simulator.run_simulation(
            duration_s=dur,
            target_fs=target_fs,
            pat_l_ms=pat_l,
            pat_r_ms=pat_r,
            disease_profile=self.var_profile.get()
        )

        self.last_df_comb = df_comb
        self.last_df_beats = df_beats
        self.last_summary = summary

        # Redraw matplotlib figure
        self.fig.clear()
        out_png = os.path.join(OUTPUT_DIR, "cardisense_waveform_simulation.png")
        self.simulator.generate_waveform_plot(df_comb, df_beats, summary, out_png)
        self.simulator.export_csvs(df_comb, df_beats, summary, OUTPUT_DIR)

        # Plot directly in Figure
        self._plot_in_figure(df_comb, df_beats, summary)
        self.canvas.draw()

    def _plot_in_figure(self, df_comb, df_beats, summary):
        time_s = df_comb['time_s'].values
        gs = self.fig.add_gridspec(3, 1, hspace=0.35)

        ax0 = self.fig.add_subplot(gs[0, 0])
        ax0.plot(time_s, df_comb['ECG_filtered'], color='#38bdf8', lw=1.0)
        ax0.plot(df_beats['t_R_s'], np.interp(df_beats['t_R_s'], time_s, df_comb['ECG_filtered']), 'rv', ms=5)
        ax0.set_title(f"Synchronized ECG (ADS1292R) - Mean HR: {summary['mean_hr']:.1f} bpm", color='#f8fafc', fontsize=10, loc='left')
        ax0.set_facecolor('#1e293b')
        ax0.tick_params(colors='#94a3b8')
        ax0.grid(True, color='#334155', ls=':')

        ax1 = self.fig.add_subplot(gs[1, 0], sharex=ax0)
        ax1.plot(time_s, df_comb['PPG_L_filtered'], color='#22c55e', lw=1.1, label='Left PPG')
        ax1.plot(time_s, df_comb['PPG_R_filtered'], color='#f43f5e', lw=1.1, ls='--', label='Right PPG')
        ax1.set_title(f"Bilateral PPG (AFE4490) - PAT_L: {summary['mean_pat_l']:.1f} ms | PAT_R: {summary['mean_pat_r']:.1f} ms | dPTT: {summary['mean_dptt']:+.1f} ms", color='#f8fafc', fontsize=10, loc='left')
        ax1.set_facecolor('#1e293b')
        ax1.tick_params(colors='#94a3b8')
        ax1.grid(True, color='#334155', ls=':')
        ax1.legend(loc='upper right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')

        # Superimposed
        ax2 = self.fig.add_subplot(gs[2, 0], sharex=ax0)
        norm_ecg = df_comb['ECG_filtered'] / (np.max(np.abs(df_comb['ECG_filtered'])) + 1e-6)
        norm_l = df_comb['PPG_L_filtered'] / (np.max(np.abs(df_comb['PPG_L_filtered'])) + 1e-6)
        norm_r = df_comb['PPG_R_filtered'] / (np.max(np.abs(df_comb['PPG_R_filtered'])) + 1e-6)
        ax2.plot(time_s, norm_ecg, color='#38bdf8', lw=1.0, alpha=0.8, label='ECG')
        ax2.plot(time_s, norm_l, color='#22c55e', lw=1.0, alpha=0.9, label='PPG Left')
        ax2.plot(time_s, norm_r, color='#f43f5e', lw=1.0, alpha=0.9, label='PPG Right')
        ax2.set_xlabel("Time (seconds)", color='#f8fafc')
        ax2.set_facecolor('#1e293b')
        ax2.tick_params(colors='#94a3b8')
        ax2.grid(True, color='#334155', ls=':')
        ax2.legend(loc='upper right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')

        self.fig.tight_layout()

    def save_combined_csv(self):
        target = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialfile="cardisense_synchronized_acquisition.csv")
        if target:
            src = os.path.join(OUTPUT_DIR, "cardisense_synchronized_acquisition.csv")
            if os.path.exists(src):
                import shutil
                shutil.copyfile(src, target)
                messagebox.showinfo("Export Success", f"Saved Combined CSV to:\n{target}")

    def save_beats_csv(self):
        target = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialfile="cardisense_timing_beats_pat_ptt.csv")
        if target:
            src = os.path.join(OUTPUT_DIR, "cardisense_timing_beats_pat_ptt.csv")
            if os.path.exists(src):
                import shutil
                shutil.copyfile(src, target)
                messagebox.showinfo("Export Success", f"Saved Beat Timings CSV to:\n{target}")

def main():
    root = tk.Tk()
    app = CardisenseDesktopGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
