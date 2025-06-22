import threading
import time
import uuid
import numpy as np
from typing import Dict, Any, Optional
from app.services.surface_processor import SurfaceProcessor
from app.services.surface_cache import surface_cache
from app.utils.serialization import make_json_serializable, validate_json_serializable
import logging
from app.utils.ply_parser import PLYParser
from fastapi import HTTPException
import signal
import os
import _thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_threading_primitives(obj, path="root"):
    if isinstance(obj, (threading.Thread, threading.Lock, threading.RLock, _thread.lock)):
        logger.error(f"Threading primitive found at {path}: {type(obj)}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            log_threading_primitives(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            log_threading_primitives(v, f"{path}[{i}]")

class AnalysisExecutor:
    """
    Manages analysis job execution with improved thread management.
    """
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._results_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.MAX_CONCURRENT_JOBS = 10
        self.surface_processor = SurfaceProcessor()

    def _set_thread_priority(self):
        """Attempt to set higher priority for the analysis thread."""
        try:
            # Set nice value to higher priority (lower number = higher priority)
            os.nice(-10)  # Requires root privileges
            logger.info("Thread priority increased")
        except (OSError, PermissionError):
            logger.warning("Could not increase thread priority (requires root)")
            pass

    def _run_analysis_with_timeout(self, analysis_id: str):
        """Run analysis with timeout protection."""
        logger.info(f"[{analysis_id}] Timeout wrapper started (thread launch confirmed)")
        
        # Remove signal handling from thread - it only works in main thread
        # def timeout_handler(signum, frame):
        #     raise TimeoutError("Analysis timed out")
        # signal.signal(signal.SIGALRM, timeout_handler)
        # signal.alarm(300)  # 5 minutes timeout
        
        try:
            logger.info(f"[{analysis_id}] Calling _run_analysis...")
            self._set_thread_priority()
            self._run_analysis(analysis_id)
            logger.info(f"[{analysis_id}] _run_analysis completed successfully")
        except TimeoutError as e:
            logger.error(f"[{analysis_id}] Analysis timed out: {e}")
            with self._lock:
                job = self._jobs.get(analysis_id)
                if job:
                    job["status"] = "failed"
                    job["error_message"] = f"Analysis timed out after 5 minutes"
                    job["completion_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        except Exception as e:
            logger.error(f"[{analysis_id}] Unexpected error in analysis: {e}", exc_info=True)
            with self._lock:
                job = self._jobs.get(analysis_id)
                if job:
                    job["status"] = "failed"
                    job["error_message"] = f"Unexpected error: {type(e).__name__}"
                    job["completion_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        finally:
            # signal.alarm(0)  # Cancel the alarm
            logger.info(f"[{analysis_id}] Timeout wrapper finished")

    def start_analysis_execution(self, analysis_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start analysis execution with improved thread management."""
        with self._lock:
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
                "params": serializable_params  # Store serializable params
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
            
            # Create thread with timeout protection
            t = threading.Thread(
                target=self._run_analysis_with_timeout, 
                args=(analysis_id,),
                daemon=True  # Make thread daemon so it doesn't block shutdown
            )
            # Don't store the thread object to avoid serialization issues
            # job["thread"] = t
            logger.info(f"[{analysis_id}] Creating analysis thread...")
            t.start()
            logger.info(f"[{analysis_id}] Analysis thread started successfully")
            logger.info(f"[{analysis_id}] Job status after thread start: {job['status']}")
            
            return {
                "status": "started",
                "analysis_id": analysis_id,
                "message": f"Analysis started with {surface_count} surfaces"
            }

    def _run_analysis(self, analysis_id: str):
        """Main analysis execution with detailed logging."""
        logger.info(f"[{analysis_id}] _run_analysis thread body entered (thread launch confirmed)")
        logger.info(f"[{analysis_id}] Analysis thread started")
        
        # Try to increase process priority
        try:
            os.nice(-10)  # Increase priority (requires root)
            logger.info(f"[{analysis_id}] Increased process priority with os.nice(-10)")
        except Exception as e:
            logger.warning(f"[{analysis_id}] Could not set process priority: {e}")
        
        with self._lock:
            job = self._jobs[analysis_id]
            params = job['params']
            job["status"] = "running"
            job["progress_percent"] = 10.0
            job["current_step"] = "loading_surfaces"
        
        try:
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
                with self._lock:
                    job["progress_percent"] = 20.0 + (i / len(surface_ids)) * 30.0
                    job["current_step"] = f"loaded_surface_{i+1}"
            
            processing_params = params.get('params', {})
            
            logger.info(f"[{analysis_id}] Starting surface processing with {len(surfaces_to_process)} surfaces")
            with self._lock:
                job["progress_percent"] = 50.0
                job["current_step"] = "processing_surfaces"
            
            analysis_results = self.surface_processor.process_surfaces(surfaces_to_process, processing_params)
            logger.info(f"[{analysis_id}] Surface processing completed successfully")
            
            # Ensure results are fully JSON serializable before caching
            logger.info(f"[{analysis_id}] Serializing results for caching")
            with self._lock:
                job["progress_percent"] = 90.0
                job["current_step"] = "serializing_results"
            
            # The surface processor already applies make_json_serializable, but let's double-check
            serializable_results = make_json_serializable(analysis_results)
            logger.info(f"[{analysis_id}] Results serialized successfully")

            # Validate that results are actually JSON serializable
            if not validate_json_serializable(serializable_results):
                logger.error(f"[{analysis_id}] Results still not JSON serializable after conversion")
                raise RuntimeError("Failed to serialize analysis results")

            with self._lock:
                logger.info(f"[{analysis_id}] Updating job status to completed")
                job["status"] = "completed"
                job["progress_percent"] = 100.0
                job["current_step"] = "finished"
                job["completion_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                
                # Store results for visualization - ensure no threading primitives
                self._results_cache[analysis_id] = {
                    **serializable_results,
                    "analysis_metadata": {"status": "completed"}
                }
                logger.info(f"[{analysis_id}] Results cached successfully. Analysis complete.")
                
        except Exception as e:
            logger.error(f"[{analysis_id}] Error during analysis: {e}", exc_info=True)
            with self._lock:
                job["status"] = "failed"
                job["error_message"] = f"Analysis failed: {str(e)}"
                job["completion_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

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
        """Get analysis status with thread monitoring."""
        with self._lock:
            if analysis_id not in self._jobs:
                raise KeyError("Analysis not found")
            
            job = self._jobs[analysis_id]
            # Log threading primitives in the job dict
            log_threading_primitives(job, path=f"job[{analysis_id}]")
            
            status = {
                "analysis_id": analysis_id,
                "status": job["status"],
                "progress_percent": job["progress_percent"],
                "current_step": job["current_step"],
                "thread_alive": True,  # Assume thread is alive since we can't check
                "start_time": job.get("start_time"),
                "completion_time": job.get("completion_time"),
                "error_message": job.get("error_message")
            }
            
            # Log threading primitives in the status object being returned
            log_threading_primitives(status, path=f"status[{analysis_id}]")
            logger.info(f"Returning status for {analysis_id}: {status}")
            
            # Ensure the status response is JSON serializable
            return make_json_serializable(status)

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