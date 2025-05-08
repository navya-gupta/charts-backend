from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import graph  # Import your router

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
