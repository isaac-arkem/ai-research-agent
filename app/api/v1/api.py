"""
API v1 router — same layout as arkemgpt-api (app/api/v1).
"""

from fastapi import APIRouter

from app.api.v1.endpoints import health, research

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(research.router, tags=["Research"])
