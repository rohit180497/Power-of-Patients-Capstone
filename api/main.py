from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
from contextlib import asynccontextmanager

# Import routers
from routers import auth, patient, researcher, chat

# Import database initialization
from core.database import initialize_db
from core.lifespan import lifespan_context

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app with lifespan
app = FastAPI(
    title="Power of Patient API",
    description="Agentic medical platform for TBI management and research",
    version="1.0.0",
    lifespan=lifespan_context
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(patient.router, prefix="/api/patient", tags=["Patient"])
app.include_router(researcher.router, prefix="/api/researcher", tags=["Researcher"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])

# Health check endpoint
@app.get("/health", tags=["Health"])
def health_check():
    """API health check endpoint"""
    from datetime import datetime
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Frontend routes - for demo purposes
templates = Jinja2Templates(directory="frontend/templates")

@app.get("/", tags=["Frontend"])
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", tags=["Frontend"])
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/patient/chat", tags=["Frontend"])
async def patient_chat(request: Request):
    return templates.TemplateResponse("patient_chat.html", {"request": request})

@app.get("/researcher/dashboard", tags=["Frontend"])
async def researcher_dashboard(request: Request):
    return templates.TemplateResponse("researcher_dashboard.html", {"request": request})

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)