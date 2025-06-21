import threading
import time
import uuid
from typing import Dict, Any, Optional

class AnalysisExecutor:
    """Manages background analysis execution, progress, and cancellation."""
    _jobs: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()

    def start_analysis_execution(self, analysis_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            if analysis_id in self._jobs:
                job = self._jobs[analysis_id]
                if job["status"] in ("running", "pending"):
                    raise RuntimeError("Analysis already running")
            job = {
                "status": "running",
                "progress_percent": 0.0,
                "current_step": "initializing",
                "cancelled": False,
                "estimated_duration": self.calculate_estimated_duration(1),
                "start_time": time.time(),
                "params": params or {},
                "thread": None
            }
            self._jobs[analysis_id] = job
            t = threading.Thread(target=self._run_analysis, args=(analysis_id,))
            job["thread"] = t
            t.start()
        return {
            "status": "started",
            "estimated_duration": job["estimated_duration"],
            "job_id": analysis_id
        }

    def _run_analysis(self, analysis_id: str):
        steps = ["preprocessing", "volume_calculation", "thickness_analysis", "postprocessing"]
        total_steps = len(steps)
        for i, step in enumerate(steps):
            with self._lock:
                job = self._jobs[analysis_id]
                if job["cancelled"]:
                    job["status"] = "cancelled"
                    job["progress_percent"] = float(i) / total_steps * 100
                    job["current_step"] = step
                    return
                job["current_step"] = step
                job["progress_percent"] = float(i) / total_steps * 100
            time.sleep(0.2)  # Simulate work
        with self._lock:
            job = self._jobs[analysis_id]
            job["status"] = "completed"
            job["progress_percent"] = 100.0
            job["current_step"] = "finished"
            job["completion_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(analysis_id)
            if not job:
                raise KeyError("Analysis not found")
            status = {
                "analysis_id": analysis_id,
                "status": job["status"],
                "progress_percent": job["progress_percent"],
                "current_step": job["current_step"]
            }
            if job["status"] == "completed":
                status["completion_time"] = job.get("completion_time")
            if job["status"] == "failed":
                status["error_message"] = job.get("error_message", "Unknown error")
            return status

    def cancel_analysis(self, analysis_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(analysis_id)
            if not job:
                raise KeyError("Analysis not found")
            if job["status"] == "completed":
                raise RuntimeError("Analysis already completed")
            if job["status"] == "cancelled":
                raise RuntimeError("Analysis already cancelled")
            job["cancelled"] = True
            job["status"] = "cancelled"
        return {"status": "cancelled", "analysis_id": analysis_id}

    @staticmethod
    def is_valid_status(status: str) -> bool:
        return status in ("pending", "running", "completed", "failed", "cancelled")

    @staticmethod
    def is_valid_priority(priority: str) -> bool:
        return priority in ("low", "normal", "high", "urgent")

    @staticmethod
    def validate_execution_parameters(params: Dict[str, Any]) -> bool:
        if not isinstance(params, dict):
            return False
        if "priority" in params and not AnalysisExecutor.is_valid_priority(params["priority"]):
            return False
        if "notify_on_completion" in params and not isinstance(params["notify_on_completion"], bool):
            return False
        if "save_intermediate_results" in params and not isinstance(params["save_intermediate_results"], bool):
            return False
        return True

    @staticmethod
    def generate_analysis_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def calculate_progress(current: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return min(100.0, max(0.0, float(current) / total * 100))

    @staticmethod
    def calculate_estimated_duration(surface_count: int) -> float:
        # Simulate: 1s per surface
        return max(1.0, float(surface_count) * 1.0)

    @staticmethod
    def can_cancel(status: str) -> bool:
        return status in ("pending", "running")

    @staticmethod
    def format_error_message(msg: str, step: str) -> str:
        return f"Error during {step}: {msg}"

    @staticmethod
    def classify_error(msg: str) -> str:
        if "not found" in msg.lower():
            return "input_error"
        if "memory" in msg.lower():
            return "resource_error"
        return "processing_error" 