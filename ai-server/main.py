from core.database import initialize_database, db_session
from fastapi import FastAPI, Response
from routes.main import router
import psutil
import os
from datetime import datetime
import time
from typing import Any, Dict

# Track service start time
SERVICE_START_TIME = time.time()

initialize_database(db_session)

app = FastAPI(
    title="Risksoft AI Server",
    description="AI-powered features for Risksoft platform",
    version="1.0.0",
)

app.include_router(router)


@app.get("/health", tags=["Health"])
async def health_check(response: Response):
    """
    Modern health check endpoint with system metrics and database status
    """
    uptime_seconds = time.time() - SERVICE_START_TIME

    health_data: Dict[str, Any] = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime": f"{uptime_seconds:.2f}s",
        "environment": os.getenv("NODE_ENV", "unknown"),
    }

    # System metrics
    try:
        memory = psutil.virtual_memory()
        system_info: Dict[str, Any] = {
            "memory": {
                "total_mb": round(memory.total / 1024 / 1024),
                "available_mb": round(memory.available / 1024 / 1024),
                "usage_percent": memory.percent,
            },
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "platform": os.uname().sysname if hasattr(os, "uname") else "unknown",
        }
        health_data["system"] = system_info
    except Exception as e:
        health_data["system"] = {"error": str(e)}

    status_code = 200 if health_data["status"] == "healthy" else 503
    response.status_code = status_code

    return health_data
