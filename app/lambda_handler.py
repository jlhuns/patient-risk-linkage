"""
Lambda entry point. Mangum adapts the ASGI app (FastAPI/Starlette) to the
Lambda/API Gateway request-response shape — same app code as the container
deployment, just a different front door.
"""

from mangum import Mangum

from app.main import app

handler = Mangum(app)
