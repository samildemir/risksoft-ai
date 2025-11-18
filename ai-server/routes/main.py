from fastapi import APIRouter
from routes.chat_router import router as chat_router
from routes.indexing_route import router as indexing_router
from routes.risk_route import router as risk_router

# Main API router with version prefix
router = APIRouter(prefix="/api/v1")

# Include all sub-routers
router.include_router(chat_router)
router.include_router(indexing_router)
router.include_router(risk_router)
