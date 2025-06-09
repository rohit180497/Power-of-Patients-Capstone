from typing import Any, Dict, List, Optional
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.database import initialize_db
import agents.pandas_agent
import agents.medpalm_agent 
import agents.knowledge_base_agent

# Configure logging
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan_context(app: FastAPI):
    """
    Lifespan context manager for FastAPI
    
    Handles startup and shutdown events:
    - Database initialization
    - LLM model loading
    - Agent initialization
    - Resource cleanup
    
    Args:
        app: FastAPI application
    """
    # Startup: initialize resources
    logger.info("Starting application: initializing resources")
    
    # Initialize database
    initialize_db()
    logger.info("Database initialized")
    
    # Initialize LLM models
    try:
        # Load agent models
        agents.medpalm_agent.initialize()
        logger.info("MedPalm agent initialized")
        
        agents.pandas_agent.initialize()
        logger.info("PandasAgent initialized")
        
        agents.knowledge_base_agent.initialize()
        logger.info("Knowledge base agent initialized")
    except Exception as e:
        logger.error(f"Error initializing agents: {str(e)}")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown: cleanup resources
    logger.info("Application shutting down: cleaning up resources")
    
    try:
        # Cleanup agent resources if needed
        agents.medpalm_agent.cleanup()
        agents.pandas_agent.cleanup()
        agents.knowledge_base_agent.cleanup()
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
    
    logger.info("Application shutdown complete")