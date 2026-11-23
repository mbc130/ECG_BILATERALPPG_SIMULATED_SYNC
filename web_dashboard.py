"""
web_dashboard.py
----------------
Local web-based interface for Cardisense ECG + Bilateral PPG Acquisition Simulation.
Zero external dependencies: uses Python's built-in http.server and socketserver.
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
import threading
import base64

from cardisense_simulator import CardisenseSimulator, OUTPUT_DIR

PORT = 8050
HTML_FILE = os.path.join(os.path.dirname(__file__), "web", "index.html")

# Global state to keep track of latest simulation result
LATEST_SIMULATION = {
    "summary": None,
    "files": {}
}

class CardisenseWebHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(status=204)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            if os.path.exists(HTML_FILE):
                with open(HTML_FILE, "r", encoding="utf-8") as f:
                    content = f.read()
                self._set_headers("text/html; charset=utf-8")
                self.wfile.write(content.encode("utf-8"))
            else:
                self._set_headers("text/plain", 404)
                self.wfile.write(b"Index file not found.")

        elif path == "/api/files":
            files = CardisenseSimulator.get_available_files()
            ecg_list = [{"name": os.path.basename(f), "path": f} for f in files["ecg"]]
            ppg_list = [{"name": os.path.basename(f), "path": f} for f in files["ppg"]]
            self._set_headers("application/json")
            self.wfile.write(json.dumps({"ecg": ecg_list, "ppg": ppg_list}).encode("utf-8"))

        elif path == "/api/download/mode_timings":
            # Downloads the mode-specific beat timings CSV
            mode = query.get("mode", [None])[0]
            if mode:
                mode_slug = CardisenseSimulator.get_mode_slug(mode)
            elif LATEST_SIMULATION["summary"]:
                mode_slug = LATEST_SIMULATION["summary"].get("mode_slug", "current_mode")
            else:
                mode_slug = "normal_mode"

            csv_path = os.path.join(OUTPUT_DIR, f"cardisense_{mode_slug}_timings.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(OUTPUT_DIR, "cardisense_timing_beats_pat_ptt.csv")

            if os.path.exists(csv_path):
                fname = f"cardisense_{mode_slug}_timings.csv"
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
                self.end_headers()
                with open(csv_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._set_headers("text/plain", 404)
                self.wfile.write(b"Mode timing file not found. Please run simulation first.")

        elif path == "/api/download/mode_waveforms":
            # Downloads the compact 3-5 cardiac cycles waveform CSV for current mode
            mode = query.get("mode", [None])[0]
            if mode:
                mode_slug = CardisenseSimulator.get_mode_slug(mode)
            elif LATEST_SIMULATION["summary"]:
                mode_slug = LATEST_SIMULATION["summary"].get("mode_slug", "current_mode")
            else:
                mode_slug = "normal_mode"

            csv_path = os.path.join(OUTPUT_DIR, f"cardisense_{mode_slug}_compact_waveform.csv")
            if os.path.exists(csv_path):
                fname = f"cardisense_{mode_slug}_compact_waveform.csv"
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
                self.end_headers()
                with open(csv_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._set_headers("text/plain", 404)
                self.wfile.write(b"Waveform file not found. Please run simulation first.")

        elif path == "/api/download/full_acquisition":
            csv_path = os.path.join(OUTPUT_DIR, "cardisense_synchronized_acquisition.csv")
            if os.path.exists(csv_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="cardisense_full_continuous_acquisition.csv"')
                self.end_headers()
                with open(csv_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._set_headers("text/plain", 404)
                self.wfile.write(b"Full acquisition file not found. Please run simulation first.")

        elif path == "/api/image/waveform":
            img_path = os.path.join(OUTPUT_DIR, "cardisense_waveform_simulation.png")
            if os.path.exists(img_path):
                self._set_headers("image/png")
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._set_headers("text/plain", 404)
                self.wfile.write(b"Image not found.")

        else:
            self._set_headers("text/plain", 404)
            self.wfile.write(b"404 Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/simulate":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)
            try:
                params = json.loads(post_body.decode("utf-8"))
            except Exception as e:
                self._set_headers("application/json", 400)
                self.wfile.write(json.dumps({"error": f"Invalid JSON: {e}"}).encode("utf-8"))
                return

            ecg_file = params.get("ecg_file")
            ppg_file = params.get("ppg_file")
            target_fs = float(params.get("target_fs", 500.0))
            duration_s = float(params.get("duration_s", 25.0))
            pat_l_ms = float(params.get("pat_l_ms", 210.0))
            pat_r_ms = float(params.get("pat_r_ms", 218.0))
            jitter_ms = float(params.get("jitter_ms", 2.0))
            disease_profile = params.get("disease_profile", "Healthy Normal")

            sim = CardisenseSimulator(ecg_path=ecg_file, ppg_path=ppg_file)
            sim.set_disease_preset(disease_profile)

            try:
                df_comb, df_beats, summary = sim.run_simulation(
                    duration_s=duration_s,
                    target_fs=target_fs,
                    pat_l_ms=pat_l_ms,
                    pat_r_ms=pat_r_ms,
                    jitter_ms=jitter_ms,
                    disease_profile=disease_profile
                )

                img_path = os.path.join(OUTPUT_DIR, "cardisense_waveform_simulation.png")
                sim.generate_waveform_plot(df_comb, df_beats, summary, img_path)
                exported_files = sim.export_csvs(df_comb, df_beats, summary, OUTPUT_DIR)

                LATEST_SIMULATION["summary"] = summary
                LATEST_SIMULATION["files"] = exported_files

                # Base64 image for instant display
                with open(img_path, "rb") as img_f:
                    img_b64 = base64.b64encode(img_f.read()).decode("utf-8")

                mode_slug = summary["mode_slug"]

                response_data = {
                    "success": True,
                    "summary": summary,
                    "image_b64": f"data:image/png;base64,{img_b64}",
                    "beats": df_beats.head(25).to_dict(orient="records"),
                    "mode_timings_url": f"/api/download/mode_timings?mode={urllib.parse.quote(disease_profile)}",
                    "mode_waveforms_url": f"/api/download/mode_waveforms?mode={urllib.parse.quote(disease_profile)}",
                    "mode_timings_filename": f"cardisense_{mode_slug}_timings.csv",
                    "mode_waveforms_filename": f"cardisense_{mode_slug}_compact_waveform.csv",
                    "files": exported_files
                }

                self._set_headers("application/json")
                self.wfile.write(json.dumps(response_data).encode("utf-8"))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._set_headers("application/json", 500)
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
        else:
            self._set_headers("text/plain", 404)
            self.wfile.write(b"404 Not Found")

def start_server(open_browser=False):
    os.makedirs(os.path.join(os.path.dirname(__file__), "web"), exist_ok=True)
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, CardisenseWebHandler)
    url = f"http://localhost:{PORT}"
    print(f"\n=======================================================")
    print(f" CARDISENSE SYNCHRONIZATION SIMULATOR WEB INTERFACE")
    print(f" Web Server running at: {url}")
    print(f"=======================================================\n")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    start_server(open_browser=False)
