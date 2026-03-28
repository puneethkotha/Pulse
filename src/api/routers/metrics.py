"""
/metrics endpoint — return evaluation results from saved artifacts.
"""

from fastapi import APIRouter, HTTPException
from src.api.dependencies import app_state

router = APIRouter()


@router.get("/metrics")
def get_metrics() -> dict:
    if app_state.eval_metrics is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation metrics not found. Run scripts/evaluate.py first.",
        )
    return app_state.eval_metrics
