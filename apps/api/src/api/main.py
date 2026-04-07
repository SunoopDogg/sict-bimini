from fastapi import FastAPI

from api.core.config import settings
from api.routers import health

app = FastAPI(title=settings.app_name, debug=settings.debug)
app.include_router(health.router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": settings.app_name}
