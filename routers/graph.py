from fastapi import APIRouter, UploadFile, HTTPException, Form, Depends
from utils.file_handler import save_upload_file
from services.data_processor import extract_dataframe_from_csv
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
import os
from datetime import datetime
from utils.sql_queries import CREATE_DATABASE, CREATE_USER_UPLOADS_TABLE, INSERT_USER_UPLOAD
from dotenv import load_dotenv
import csv
import io
from services.graph_functions import get_shift_factors as calculate_shift_factors
from services.graph_calculator import calculate_graph_data, calculate_sheer_modulus_vs_frequency, calculate_relaxation_modulus_vs_time, calculate_master_curve_graph_data
import logging

# CHANGE: Setup proper logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

router = APIRouter()

# Replace global uploaded_file_path with a dictionary to store paths per upload_id
upload_paths = {}  # Maps upload_id to file path
cached_shift_factors = {}

# CHANGE: Database connection pool - lazy initialization
pool = None

# CHANGE: Database configuration from environment variables with timeout
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "connect_timeout": 5  # CHANGE: Add timeout to database operations
}

# CHANGE: Lazy initialization function for database connection pool
def initialize_database_pool():
    global pool
    if pool is None:
        try:
            pool = MySQLConnectionPool(
                pool_name="graph_app_pool",
                pool_size=5,
                **db_config
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {str(e)}")
            raise

# CHANGE: Function to get connection from pool
def get_connection():
    if pool is None:
        initialize_database_pool()
    try:
        return pool.get_connection()
    except Exception as e:
        logger.error(f"Failed to get connection from pool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

# CHANGE: Use dependency injection pattern for database connections
def get_db():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()

# CHANGE: Initialize database connection without database initially
def get_db_connection():
    try:
        # CHANGE: Added timeout
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=5
        )
        return conn
    except mysql.connector.Error as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

# CHANGE: Create database if it doesn't exist - with proper error handling
def create_database():
    try:
        # CHANGE: Use with statement for connection
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(CREATE_DATABASE.format(db_name=os.getenv("DB_NAME")))
                conn.commit()
        logger.info(f"Database {os.getenv('DB_NAME')} created or confirmed to exist")
    except mysql.connector.Error as e:
        logger.error(f"Error creating database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating database: {str(e)}")

# CHANGE: Create table if it doesn't exist - with proper error handling
def create_user_uploads_table():
    try:
        # CHANGE: Use with statement for connection and cursor
        with mysql.connector.connect(**db_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(CREATE_USER_UPLOADS_TABLE)
                conn.commit()
        logger.info("User uploads table created or confirmed to exist")
    except mysql.connector.Error as e:
        logger.error(f"Error creating table: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating table: {str(e)}")

# CHANGE: Helper function to generate prefixed IDs - with proper connection handling
def generate_id(prefix, table_name, conn):
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT MAX(SUBSTRING(id, {len(prefix)+2})) FROM {table_name} WHERE id LIKE '{prefix}-%'")
            max_id = cursor.fetchone()[0]
            new_id_num = 1 if max_id is None else int(max_id) + 1
            return f"{prefix}-{new_id_num}"
    except mysql.connector.Error as e:
        logger.error(f"Error generating ID: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database operation failed: {str(e)}")

# CHANGE: Initialize database on startup function
def initialize_database():
    create_database()
    create_user_uploads_table()
    initialize_database_pool()

# CHANGE: Explicitly initialize database at server startup
# This should be called from a startup event handler in the main application
# initialize_database()

# Single endpoint for form and CSV upload
@router.post("/upload-csv/")
async def upload_csv(
    file: UploadFile, 
    full_name: str = Form(...), 
    university: str = Form(...), 
    department: str = Form(...), 
    make_and_model_of_machine: str = Form(...), 
    email_address: str = Form(...)
):
    logger.info("Processing CSV upload request")
    global upload_paths
    try:
        # Reset file pointer to the beginning before saving
        file.file.seek(0)

        # CHANGE: Save file to filesystem with proper error handling
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{full_name.replace(' ', '_')}_{timestamp}.csv"
            file_path = save_upload_file(file, file_name)
            logger.info(f"Saved file path: {file_path}")
            # Normalize path to forward slashes for consistency
            file_path = file_path.replace("\\", "/")
            logger.info(f"Normalized file path: {file_path}")
        except IOError as e:
            logger.error(f"File operation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File save operation failed: {str(e)}")

        # CHANGE: Insert form data and CSV content into MySQL with proper connection handling
        try:
            # CHANGE: Use with statement for database operations
            with mysql.connector.connect(**db_config) as conn:
                upload_id = generate_id("U", "user_uploads", conn)
                timestamp = datetime.now()
                
                with conn.cursor() as cursor:
                    cursor.execute(
                        INSERT_USER_UPLOAD,
                        (upload_id, full_name, university, department, make_and_model_of_machine, email_address, timestamp)
                    )
                conn.commit()

            # Store the file path in the dictionary with the upload_id as key
            upload_paths[upload_id] = file_path
            logger.info(f"Upload paths: {upload_paths}")

            return {
                "status": "success",
                "message": "Form and CSV content uploaded successfully",
                "upload_id": upload_id
            }
        except mysql.connector.Error as e:
            logger.error(f"Database operation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database operation failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# CHANGE: Helper function to validate upload_id
def validate_upload_id(upload_id: str):
    if not upload_id or upload_id not in upload_paths:
        logger.warning(f"Invalid upload_id provided: {upload_id}")
        raise HTTPException(status_code=400, detail="No valid upload_id provided or file not found")
    return upload_paths[upload_id]

# Update endpoints to use upload_id from the request or a default
@router.get("/get-shift-factors/")
async def fetch_shift_factors(upload_id: str = None):
    global cached_shift_factors
    file_path = validate_upload_id(upload_id)
    
    if cached_shift_factors.get(upload_id) is not None:
        return {"status": "success", "shift_factors": cached_shift_factors[upload_id]}

    try:
        # CHANGE: Add proper error handling for file operations
        try:
            data = extract_dataframe_from_csv(file_path)
        except IOError as e:
            logger.error(f"File operation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File operation failed: {str(e)}")
            
        cached_shift_factors[upload_id] = calculate_shift_factors(data)
        return {"status": "success", "shift_factors": cached_shift_factors[upload_id]}
    except Exception as e:
        logger.error(f"Error calculating shift factors: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/get-graph-data")
async def get_graph_data(upload_id: str = None):
    file_path = validate_upload_id(upload_id)
    
    try:
        # CHANGE: Add proper error handling for file operations
        try:
            dataframe_from_csv = extract_dataframe_from_csv(file_path)
        except IOError as e:
            logger.error(f"File operation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File operation failed: {str(e)}")
            
        graph_data = calculate_graph_data(dataframe_from_csv)
        logger.info("Graph data calculated successfully")
        return {
            "status": "success",
            "graphData": graph_data
        }
    except Exception as e:
        logger.error(f"Error calculating graph data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/get-sheer-modulus-vs-frequency")
async def get_sheer_modulus_vs_frequency(upload_id: str = None):
    logger.info(f"Processing sheer modulus request for upload_id: {upload_id}")
    file_path = validate_upload_id(upload_id)
    
    logger.info(f"Processing file: {file_path}")
    try:
        # CHANGE: Add proper error handling for file operations
        try:
            dataframe_from_csv = extract_dataframe_from_csv(file_path)
        except IOError as e:
            logger.error(f"File operation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File operation failed: {str(e)}")
            
        graph_data = calculate_sheer_modulus_vs_frequency(dataframe_from_csv)
        return {
            "status": "success",
            "graphData": graph_data
        }
    except Exception as e:
        logger.error(f"Error calculating sheer modulus: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/get-relaxtion-modulus-vs-time")
async def get_relaxtation_modulus_with_time(upload_id: str = None, a_upper_bound: float = 500, d_upper_bound: float = 500):
    global cached_shift_factors
    file_path = validate_upload_id(upload_id)
    
    try:
        # CHANGE: Add proper error handling for file operations
        try:
            data = extract_dataframe_from_csv(file_path)
        except IOError as e:
            logger.error(f"File operation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File operation failed: {str(e)}")
        
        if cached_shift_factors.get(upload_id) is None:
            cached_shift_factors[upload_id] = calculate_shift_factors(data)
            logger.info(f"Shift Factors calculated: {cached_shift_factors[upload_id]}")

        if cached_shift_factors.get(upload_id) is None:
            raise ValueError("Shift factors could not be calculated.")
        
        graph_data = calculate_relaxation_modulus_vs_time(data, cached_shift_factors[upload_id], a_upper_bound=a_upper_bound, d_upper_bound=d_upper_bound)
        return {
            "status": "success",    
            "graphData": graph_data
        }
    except Exception as e:
        logger.error(f"Error calculating relaxation modulus: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/get-master-curve-for-all-temperatures")
async def get_master_curve_for_all_temperatures(upload_id: str = None, a_upper_bound: float = 500, d_upper_bound: float = 500):
    file_path = validate_upload_id(upload_id)
    
    try:
        # CHANGE: Add proper error handling for file operations
        try:
            data = extract_dataframe_from_csv(file_path)
        except IOError as e:
            logger.error(f"File operation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File operation failed: {str(e)}")
            
        graph_data = calculate_master_curve_graph_data(data, a_upper_bound=a_upper_bound, d_upper_bound=d_upper_bound)
        return {
            "status": "success",
            "graphData": graph_data
        }
    except Exception as e:
        logger.error(f"Error calculating master curve: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))