from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional
from app.services.analysis_executor import AnalysisExecutor

router = APIRouter(prefix="/api/analysis", tags=["analysis"])
executor = AnalysisExecutor()

@router.post("/{analysis_id}/execute")
async def execute_analysis(analysis_id: str, request: Request):
    try:
        params = await request.json() if request.headers.get("content-type", "").startswith("application/json") else None
        if params and not executor.validate_execution_parameters(params):
            raise HTTPException(status_code=400, detail="Invalid execution parameters")
        result = executor.start_analysis_execution(analysis_id, params)
        return JSONResponse(status_code=202, content=result)
    except RuntimeError as e:
        if "already running" in str(e):
            raise HTTPException(status_code=409, detail="Analysis already running")
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))
    except Exception as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail="Analysis not found")
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

@router.get("/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    try:
        status = executor.get_analysis_status(analysis_id)
        return status
    except KeyError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

@router.post("/{analysis_id}/cancel")
async def cancel_analysis(analysis_id: str):
    try:
        result = executor.cancel_analysis(analysis_id)
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    except RuntimeError as e:
        if "already completed" in str(e):
            raise HTTPException(status_code=400, detail="Analysis already completed")
        if "already cancelled" in str(e):
            raise HTTPException(status_code=400, detail="Analysis already cancelled")
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e)) 