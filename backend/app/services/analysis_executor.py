import threading
import time
import uuid
import numpy as np
from typing import Dict, Any, Optional
from app.services.surface_processor import SurfaceProcessor
from app.services.surface_cache import surface_cache

class AnalysisExecutor:
    """Manages background analysis execution, progress, and cancellation."""
    _jobs: Dict[str, Dict[str, Any]] = {}
    _results_cache: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()

    def __init__(self):
        self.surface_processor = SurfaceProcessor()

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
                "estimated_duration": self.calculate_estimated_duration(len(params.get('surface_ids', []))),
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
        with self._lock:
            job = self._jobs[analysis_id]
            params = job['params']
            surfaces_to_process = []
            surface_ids = params.get('surface_ids', [])

        # Step 1: Load surfaces from cache
        for surface_id in surface_ids:
            cached_surface = surface_cache.get(surface_id)
            if not cached_surface:
                with self._lock:
                    job['status'] = 'failed'
                    job['error_message'] = f"Surface data not found in cache for ID: {surface_id}"
                return
            
            # Convert lists back to numpy arrays for processing
            vertices = np.array(cached_surface['vertices'])
            faces = np.array(cached_surface['faces']) if cached_surface.get('faces') else None
            surfaces_to_process.append({
                'vertices': vertices,
                'faces': faces,
                'name': cached_surface.get('filename', surface_id)
            })

        # Step 2: Generate baseline surface if requested
        if params.get('generate_base_surface') and surfaces_to_process:
            with self._lock:
                job['current_step'] = 'generating_baseline'
                job['progress_percent'] = 25.0
            
            offset = params.get('base_surface_offset', 0)
            # Use the first surface as the reference for baseline generation
            reference_surface_vertices = surfaces_to_process[0]['vertices']
            
            try:
                baseline_vertices = self.surface_processor.generate_base_surface(reference_surface_vertices, offset)
                # The baseline is just points, so faces are None.
                surfaces_to_process.insert(0, {'vertices': baseline_vertices, 'faces': None, 'name': 'Baseline'})
            except Exception as e:
                with self._lock:
                    job['status'] = 'failed'
                    job['error_message'] = f"Failed to generate baseline surface: {e}"
                return

        # --- Placeholder for further analysis ---
        # Simulate work for other steps
        time.sleep(2) 

        with self._lock:
            job["status"] = "completed"
            job["progress_percent"] = 100.0
            job["current_step"] = "finished"
            job["completion_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            
            # Store results for visualization
            self._results_cache[analysis_id] = {
                "surfaces": surfaces_to_process,
                "analysis_metadata": {"status": "completed"}
            }

    def get_results(self, analysis_id: str, include: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get analysis results with optional filtering. Returns None if not available."""
        with self._lock:
            if analysis_id not in self._jobs:
                return None
            job = self._jobs[analysis_id]
            if job["status"] != "completed":
                return None
            results = self._results_cache.get(analysis_id)
            if not results:
                return None
            if include:
                filtered_results = {"analysis_metadata": results["analysis_metadata"]}
                if include == "volume":
                    filtered_results["volume_results"] = results.get("volume_results", [])
                elif include == "thickness":
                    filtered_results["thickness_results"] = results.get("thickness_results", [])
                elif include == "compaction":
                    filtered_results["compaction_rates"] = results.get("compaction_rates", [])
                return filtered_results
            return results

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
                status["results"] = self.get_results(analysis_id)
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
            job["cancellation_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
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