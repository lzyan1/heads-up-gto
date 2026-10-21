"""Dependency-free loopback web server and single-worker solve queue."""
import argparse
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
import traceback
from urllib.parse import urlparse
import uuid
import webbrowser

from .cards import parse_cards, combo_text
from .game import Settings, State
from .sizing import profile_for, sizing_metadata
from .preflop import chart_ranges, reconstruct
from .backends import create_solver
from .screenshot_charts import screenshot_chart
from .navigation import preflop_view
from .reference_line import supports_reference, comparison_report

STATIC = Path(__file__).parent / "static"
JOBS = {}
LOCK = threading.Lock()


def require_builtin_ranges(payload):
    if payload.get("range_source", "wizard") != "wizard" or any(key in payload for key in ("chart", "bb_range", "sb_range")):
        raise ValueError("Only the hardcoded GTO Wizard preflop charts are supported. Reload the app to use the current settings.")


def app_settings(data):
    settings = Settings.from_dict(data)
    # Legacy sizing inputs cannot alter the scenario size plus all-in menu. Keep the
    # generic engine controls only for internal diagnostic games.
    return replace(settings, sizes=((), (), ()), max_raises=2, include_all_in=True)


def iteration_count(value):
    if type(value) is not int or not 10 <= value <= 2**53 - 1:
        raise ValueError("Iterations must be a whole number of at least 10.")
    return value


def training_budget(payload):
    if "time_limit_seconds" in payload:
        if "iterations" in payload:
            raise ValueError("Choose either an iteration count or a time limit, not both.")
        seconds = payload["time_limit_seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 2**53 - 1:
            raise ValueError("Time limit must be a positive, finite number of seconds.")
        return {"time_limit_seconds": seconds}
    return {"iterations": iteration_count(payload.get("iterations", 2000))}


def running_stats(solver, budget):
    return {**solver.stats(), "status": "running", "target": solver.iterations + budget["iterations"] if "iterations" in budget else None,
            "run_seconds": 0, "time_limit_seconds": budget.get("time_limit_seconds")}


def prepare(payload):
    require_builtin_ranges(payload)
    board = parse_cards(payload.get("flop", ""), 3)
    preflop = reconstruct(payload.get("preflop", ""))
    settings = app_settings(payload.get("settings", {}))
    budget = training_budget(payload)
    chart = screenshot_chart()
    ranges = chart_ranges(chart, preflop, board)
    name, is_demo = chart["name"], False
    approximation = {"frequency_step": chart["frequency_step"], "source": chart["provenance"]["source"], "jam_restriction": chart["provenance"]["jam_restriction"]}
    profile = profile_for(preflop.pot, preflop.stacks)
    root = State(preflop.pot, preflop.stacks, board, sizing_profile=profile)
    solver = create_solver(root, ranges, settings)
    return solver, budget, {"name": name, "is_demo": is_demo, "range_source": "wizard", "sizing_status": dict.fromkeys(("flop", "turn", "river"), "scenario_heuristic"), "sizing": sizing_metadata(profile), "preflop": preflop.history, "combos": [len(r) for r in ranges], "weighted_combos": [round(sum(r.values()), 2) for r in ranges], "pot": root.pot, "stacks": root.stacks, "settings": asdict(settings), "training_budget": budget, "range_approximation": approximation, "reference_line_available": supports_reference(root)}


def run_job(job, budget):
    try:
        def report(stats):
            job["stats"] = stats
        job["stats"] = job["solver"].train(progress=report, **budget)
    except Exception as exc:
        traceback.print_exc()
        job["stats"] = {**job["solver"].stats(), "status": "error", "error": str(exc)}
    finally:
        job["busy"] = False


def job_public(job):
    return {"id": job["id"], "stats": job["stats"], "metadata": job["metadata"], "busy": job["busy"]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_json(self, value, status=200):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def safe_origin(self):
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed:
            raise ValueError("This app accepts localhost requests only.")
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{h}" for h in allowed}:
            raise ValueError("Cross-origin requests are disabled.")

    def do_GET(self):
        try:
            self.safe_origin()
            path = urlparse(self.path).path
            if path == "/api/health":
                return self.send_json({"ok": True})
            if path == "/api/builtin-chart":
                return self.send_json(screenshot_chart())
            if path.startswith("/api/jobs/"):
                job = JOBS.get(path.split("/")[3])
                if not job:
                    return self.send_json({"error": "Solve not found; start a new solve."}, 404)
                return self.send_json(job_public(job))
            names = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/style.css": ("style.css", "text/css"), "/model.mjs": ("model.mjs", "text/javascript")}
            if path not in names:
                return self.send_json({"error": "Not found"}, 404)
            name, mime = names[path]
            data = (STATIC / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", f"{mime}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 400)

    def do_POST(self):
        try:
            self.safe_origin()
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Send application/json.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4_000_000:
                raise ValueError("JSON request must be smaller than 4 MB.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request must be a JSON object.")
            path = urlparse(self.path).path
            if path == "/api/settings/validate":
                app_settings(payload.get("settings", {}))
                training_budget(payload)
                return self.send_json({"ok": True})
            if path == "/api/preflop":
                require_builtin_ranges(payload)
                return self.send_json(preflop_view(payload.get("history", []), screenshot_chart()))
            if path == "/api/preview":
                _, _, metadata = prepare(payload)
                return self.send_json(metadata)
            if path == "/api/solve":
                with LOCK:
                    if any(job["busy"] for job in JOBS.values()):
                        return self.send_json({"error": "Another solve is running. Stop it before starting a new one."}, 409)
                    solver, budget, metadata = prepare(payload)
                    if len(JOBS) >= 2:
                        del JOBS[next(iter(JOBS))]
                    job = {"id": uuid.uuid4().hex, "solver": solver, "metadata": metadata, "request": payload, "stats": running_stats(solver, budget), "busy": True}
                    JOBS[job["id"]] = job
                    threading.Thread(target=run_job, args=(job, budget), daemon=True).start()
                return self.send_json(job_public(job), 202)
            if path.startswith("/api/jobs/"):
                bits = path.split("/")
                job = JOBS.get(bits[3])
                if not job:
                    return self.send_json({"error": "Solve not found."}, 404)
                command = bits[4] if len(bits) > 4 else ""
                if command == "stop":
                    job["solver"].cancel.set()
                    return self.send_json({"ok": True})
                if command == "resume":
                    budget = training_budget(payload)
                    with LOCK:
                        if any(j["busy"] for j in JOBS.values()):
                            return self.send_json({"error": "A solve is already running."}, 409)
                        # A resume keeps its card abstraction and seed. Only the
                        # optional decision cap may change without a new solve.
                        if "max_nodes" in payload:
                            job["solver"].settings = app_settings({**asdict(job["solver"].settings), "max_nodes": payload["max_nodes"]})
                        job["metadata"] = {**job["metadata"], "settings": asdict(job["solver"].settings), "training_budget": budget}
                        job["solver"].cancel.clear()
                        job["busy"] = True
                        job["stats"] = running_stats(job["solver"], budget)
                        threading.Thread(target=run_job, args=(job, budget), daemon=True).start()
                    return self.send_json(job_public(job), 202)
                if command == "refine":
                    budget = training_budget(payload)
                    node_path = payload.get("path", [])
                    if not isinstance(node_path, list) or len(node_path) > 60 or any(not isinstance(x, str) for x in node_path):
                        raise ValueError("Invalid action path.")
                    with LOCK:
                        if any(j["busy"] for j in JOBS.values()):
                            return self.send_json({"error": "A solve is already running."}, 409)
                        state, ranges, fallback = job["solver"].locate(node_path)
                        if state.terminal or state.pending:
                            raise ValueError("Refine an active decision after dealing the next board card.")
                        settings = app_settings(payload.get("settings", asdict(job["solver"].settings)))
                        solver = create_solver(replace(state, path=()), ranges, settings)
                        metadata = {**job["metadata"], "pot": state.pot, "stacks": state.stacks, "combos": [len(r) for r in ranges], "local_refinement": True, "parent_path": node_path, "fallback_used": fallback, "settings": asdict(settings), "training_budget": budget, "reference_line_available": supports_reference(solver.root)}
                        request = {"original": job["request"], "refine_path": node_path}
                        new_job = {"id": uuid.uuid4().hex, "solver": solver, "metadata": metadata, "request": request, "stats": running_stats(solver, budget), "busy": True}
                        if len(JOBS) >= 2:
                            del JOBS[next(iter(JOBS))]
                        JOBS[new_job["id"]] = new_job
                        threading.Thread(target=run_job, args=(new_job, budget), daemon=True).start()
                    return self.send_json(job_public(new_job), 202)
                if command == "diagnostics":
                    if job["busy"]:
                        return self.send_json({"error": "Stop or finish training before requesting diagnostics."}, 409)
                    return self.send_json({"stats": job["solver"].stats(), "coverage": job["solver"].coverage_diagnostics()})
                if command == "comparison":
                    if job["busy"]:
                        return self.send_json({"error": "Stop or finish training before comparing the line."}, 409)
                    return self.send_json({**comparison_report(job["solver"]),
                                           "metadata": job["metadata"], "root_state": asdict(job["solver"].root),
                                           "root_ranges": [{combo_text(h): w for h, w in r.items()} for r in job["solver"].ranges]})
                if command in ("node", "export"):
                    if job["busy"]:
                        return self.send_json({"error": "Wait for the solve to finish, or stop it, before exploring."}, 409)
                    node_path = payload.get("path", [])
                    if not isinstance(node_path, list) or len(node_path) > 60 or any(not isinstance(x, str) for x in node_path):
                        raise ValueError("Invalid action path.")
                    selected = parse_cards(payload["hand"], 2) if payload.get("hand", "").strip() else None
                    node = job["solver"].inspect(node_path, selected)
                    if command == "export":
                        return self.send_json({"version": 1, "notice": "Experimental approximate strategy; no full-game GTO guarantee or exploitability certificate.", "request": job["request"], "metadata": job["metadata"], "stats": job["stats"], "root_state": asdict(job["solver"].root), "root_ranges": [{combo_text(h): w for h, w in r.items()} for r in job["solver"].ranges], "node": node})
                    return self.send_json(node)
            return self.send_json({"error": "Not found"}, 404)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception:
            traceback.print_exc()
            self.send_json({"error": "Unexpected server error. See the local terminal for details."}, 500)


def main():
    parser = argparse.ArgumentParser(description="Local heads-up postflop study tool")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open the interface in your browser")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Heads-up GTO study tool: {url}\nPress Ctrl+C to stop.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping…", flush=True)
    finally:
        for job in JOBS.values():
            job["solver"].cancel.set()
        server.server_close()


if __name__ == "__main__":
    main()
