from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import graph  # Import your router
import logging

# CHANGE: Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://sim-hub.poly.edu:8001"],  # React frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router for graph-related endpoints
app.include_router(graph.router)

# CHANGE: Add startup event handler
@app.on_event("startup")
async def startup_event():
    # Initialize the database and connection pool
    logger.info("Application starting up - initializing database and connection pool")
    graph.initialize_database()
    logger.info("Database and connection pool initialized successfully")

# CHANGE: Add shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    # Add cleanup logic if needed
    logger.info("Application shutting down - cleaning up resources")
    # If your pool has a close method, you might want to call it here
    # For example: graph.close_pool()
    logger.info("Resources cleaned up successfully")