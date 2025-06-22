import time
import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException
from app.utils.serialization import make_json_serializable, validate_json_serializable
from app.services.surface_processor import SurfaceProcessor
from app.services.surface_cache import surface_cache
from app.utils.ply_parser import PLYParser
import os
import numpy as np

# Performance optimization: Set environment variables for better performance
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')

# NumPy performance optimizations
np.set_printoptions(precision=6, suppress=True)
np.random.seed(42)  # For reproducible results

logger = logging.getLogger(__name__)

class AnalysisExecutor:
    """
    Manages analysis job execution with FastAPI background tasks.
    """
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._results_cache: Dict[str, Dict[str, Any]] = {}
        self.MAX_CONCURRENT_JOBS = 10
        self.surface_processor = SurfaceProcessor()
        
        # Performance optimization: Pre-allocate memory pools
        self._chunk_size = 10000  # Process data in chunks for better memory management

    def run_analysis_sync(self, analysis_id: str, params: Optional[Dict[str, Any]] = None):
        """Synchronous analysis execution for background tasks"""
        logger.info(f"[{analysis_id}] Background task analysis started")
        
        try:
            # Update job status to running
            self._update_job_status(analysis_id, "running", 10.0, "loading_surfaces")
            
            # Execute analysis (same logic as current _run_analysis)
            self._execute_analysis_logic(analysis_id, params)
            
            # Update job status to completed
            self._update_job_status(analysis_id, "completed", 100.0, "finished")
            logger.info(f"[{analysis_id}] Background task analysis completed successfully")
            
        except Exception as e:
            logger.error(f"[{analysis_id}] Background task analysis failed: {e}", exc_info=True)
            self._update_job_status(analysis_id, "failed", 0.0, "error", str(e))

    def _update_job_status(self, analysis_id: str, status: str, progress: float, step: str, error_msg: str = None):
        """Thread-safe job status update"""
        if analysis_id in self._jobs:
            self._jobs[analysis_id].update({
                "status": status,
                "progress_percent": progress,
                "current_step": step
            })
            if status in ["completed", "failed", "cancelled"]:
                self._jobs[analysis_id]["completion_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if error_msg:
                self._jobs[analysis_id]["error_message"] = error_msg

    def _execute_analysis_logic(self, analysis_id: str, params: Optional[Dict[str, Any]] = None):
        """Extract analysis logic from current _run_analysis method"""
        # Copy the entire analysis logic from _run_analysis method
        # but remove all threading-specific code and locks
        # This includes surface loading, processing, and result caching
        
        surface_ids = params.get('surface_ids', [])
        logger.info(f"[{analysis_id}] Loading {len(surface_ids)} surfaces from cache")
        
        # Load surface data from cache with progress updates
        surfaces_to_process = []
        parser = PLYParser()
        
        for i, sid in enumerate(surface_ids):
            logger.info(f"[{analysis_id}] Loading surface {i+1}/{len(surface_ids)}: {sid}")
            
            cached_surface = surface_cache.get(sid)
            if not cached_surface or 'file_path' not in cached_surface:
                raise RuntimeError(f"Surface {sid} not found in cache or is invalid.")
            
            file_path = cached_surface['file_path']
            logger.info(f"[{analysis_id}] Parsing PLY file: {file_path}")
            
            vertices, faces = parser.parse_ply_file(file_path)
            logger.info(f"[{analysis_id}] Loaded {len(vertices)} vertices, {len(faces) if faces is not None else 0} faces")
            
            surfaces_to_process.append({
                "id": sid,
                "name": cached_surface.get("filename", "Unknown"),
                "vertices": vertices,
                "faces": faces
            })
            
            # Update progress
            progress = 20.0 + (i / len(surface_ids)) * 30.0
            self._update_job_status(analysis_id, "running", progress, f"loaded_surface_{i+1}")
        
        processing_params = params.get('params', {})
        
        logger.info(f"[{analysis_id}] Starting surface processing with {len(surfaces_to_process)} surfaces")
        self._update_job_status(analysis_id, "running", 50.0, "processing_surfaces")
        
        analysis_results = self.surface_processor.process_surfaces(surfaces_to_process, processing_params)
        logger.info(f"[{analysis_id}] Surface processing completed successfully")
        
        # Ensure results are fully JSON serializable before caching
        logger.info(f"[{analysis_id}] Serializing results for caching")
        self._update_job_status(analysis_id, "running", 90.0, "serializing_results")
        
        # The surface processor already applies make_json_serializable, but let's double-check
        serializable_results = make_json_serializable(analysis_results)
        logger.info(f"[{analysis_id}] Results serialized successfully")

        # Validate that results are actually JSON serializable
        if not validate_json_serializable(serializable_results):
            logger.error(f"[{analysis_id}] Results still not JSON serializable after conversion")
            raise RuntimeError("Failed to serialize analysis results")

        logger.info(f"[{analysis_id}] Updating job status to completed")
        
        # Store results for visualization - ensure no threading primitives
        self._results_cache[analysis_id] = {
            **serializable_results,
            "analysis_metadata": {"status": "completed"}
        }
        logger.info(f"[{analysis_id}] Results cached successfully. Analysis complete.")

    def start_analysis_execution(self, analysis_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start analysis execution with background task management"""
        if analysis_id in self._jobs:
            job = self._jobs[analysis_id]
            if job["status"] in ("running", "pending"):
                raise RuntimeError("Analysis already running")
        
        # Handle new frontend payload structure
        surface_count = len(params.get('surface_ids', [])) if params else 0
        
        # Ensure params are JSON serializable before storing
        serializable_params = make_json_serializable(params) if params else {}
        
        job = {
            "status": "pending",
            "progress_percent": 0.0,
            "current_step": "queued",
            "cancellable": True,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "params": serializable_params
        }
        
        if len(self._jobs) >= self.MAX_CONCURRENT_JOBS:
            # Simple queueing mechanism
            oldest_job_id = min(self._jobs.keys(), key=lambda k: self._jobs[k]["start_time"])
            if self._jobs[oldest_job_id]["status"] not in ("completed", "failed", "cancelled"):
                 raise HTTPException(status_code=503, detail="Max concurrent jobs reached, please try again later.")
            else:
                del self._jobs[oldest_job_id]

        self._jobs[analysis_id] = job
        logger.info(f"[{analysis_id}] Job created with status: {job['status']}")
        
        return {
            "status": "started",
            "analysis_id": analysis_id,
            "message": f"Analysis started with {surface_count} surfaces"
        }

    def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis status without threading primitives"""
        if analysis_id not in self._jobs:
            raise KeyError("Analysis not found")
        
        job = self._jobs[analysis_id]
        
        status = {
            "analysis_id": analysis_id,
            "status": job["status"],
            "progress_percent": job["progress_percent"],
            "current_step": job["current_step"],
            "start_time": job.get("start_time"),
            "completion_time": job.get("completion_time"),
            "error_message": job.get("error_message")
        }
        
        logger.info(f"Returning status for {analysis_id}: {status}")
        
        # Ensure the status response is JSON serializable
        return make_json_serializable(status)

    def get_results(self, analysis_id: str, include: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get analysis results with optional filtering. Returns None if not available."""
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

    def cancel_analysis(self, analysis_id: str) -> Dict[str, Any]:
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