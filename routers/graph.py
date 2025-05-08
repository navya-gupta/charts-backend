from fastapi import APIRouter, UploadFile, HTTPException, Form
from utils.file_handler import save_upload_file
from services.data_processor import extract_dataframe_from_csv
import mysql.connector
import os
from datetime import datetime
from utils.sql_queries import CREATE_DATABASE, CREATE_USER_UPLOADS_TABLE, INSERT_USER_UPLOAD
from dotenv import load_dotenv
import csv
import io
from services.graph_functions import get_shift_factors as calculate_shift_factors
from services.graph_calculator import calculate_graph_data, calculate_sheer_modulus_vs_frequency, calculate_relaxation_modulus_vs_time, calculate_master_curve_graph_data

# Load environment variables from .env file
load_dotenv()

router = APIRouter()

# Replace global uploaded_file_path with a dictionary to store paths per upload_id
upload_paths = {}  # Maps upload_id to file path
cached_shift_factors = {}

# Database configuration from environment variables
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

# Initialize database connection without database initially
def get_db_connection():
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    return conn

# Create database if it doesn't exist
def create_database():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(CREATE_DATABASE.format(db_name=os.getenv("DB_NAME")))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating database: {str(e)}")

# Create table if it doesn't exist
def create_user_uploads_table():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(CREATE_USER_UPLOADS_TABLE)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating table: {str(e)}")

# Initialize database and table on startup
create_database()
create_user_uploads_table()

# Helper function to generate prefixed IDs
def generate_id(prefix, table_name, conn):
    cursor = conn.cursor()
    cursor.execute(f"SELECT MAX(SUBSTRING(id, {len(prefix)+2})) FROM {table_name} WHERE id LIKE '{prefix}-%'")
    max_id = cursor.fetchone()[0]
    new_id_num = 1 if max_id is None else int(max_id) + 1
    return f"{prefix}-{new_id_num}"

# Single endpoint for form and CSV upload
@router.post("/upload-csv/")
async def upload_csv(file: UploadFile, full_name: str = Form(...), university: str = Form(...), department: str = Form(...), make_and_model_of_machine: str = Form(...), email_address: str = Form(...)):
    print("KK1")
    global upload_paths
    try:
        # Reset file pointer to the beginning before saving
        file.file.seek(0)

        # Save file to filesystem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{full_name.replace(' ', '_')}_{timestamp}.csv"
        file_path = save_upload_file(file, file_name)
        print("KK2")
        print(f"Saved file path: {file_path}")
        # Normalize path to forward slashes for consistency
        file_path = file_path.replace("\\", "/")
        print(f"Normalized file path: {file_path}")

        # Insert form data and CSV content into MySQL with "U-" prefixed ID
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        upload_id = generate_id("U", "user_uploads", conn)
        timestamp = datetime.now()
        cursor.execute(
            INSERT_USER_UPLOAD,
            (upload_id, full_name, university, department, make_and_model_of_machine, email_address, timestamp)
        )
        conn.commit()

        # Store the file path in the dictionary with the upload_id as key
        upload_paths[upload_id] = file_path
        print(f"Upload paths: {upload_paths}")

        cursor.close()
        conn.close()

        return {
            "status": "success",
            "message": "Form and CSV content uploaded successfully",
            "upload_id": upload_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Update endpoints to use upload_id from the request or a default
@router.get("/get-shift-factors/")
async def fetch_shift_factors(upload_id: str = None):
    global upload_paths, cached_shift_factors
    if not upload_id or upload_id not in upload_paths:
        raise HTTPException(status_code=400, detail="No valid upload_id provided or file not found")
    
    file_path = upload_paths[upload_id]
    if cached_shift_factors.get(upload_id) is not None:
        return {"status": "success", "shift_factors": cached_shift_factors[upload_id]}

    try:
        data = extract_dataframe_from_csv(file_path)
        cached_shift_factors[upload_id] = calculate_shift_factors(data)
        return {"status": "success", "shift_factors": cached_shift_factors[upload_id]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/get-graph-data")
async def get_graph_data(upload_id: str = None):
    global upload_paths
    if not upload_id or upload_id not in upload_paths:
        raise HTTPException(status_code=400, detail="No valid upload_id provided or file not found")
    
    file_path = upload_paths[upload_id]
    try:
        dataframe_from_csv = extract_dataframe_from_csv(file_path)
        graph_data = calculate_graph_data(dataframe_from_csv)
        print("Graph data: ", graph_data)
        return {
            "status": "success",
            "graphData": graph_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/get-sheer-modulus-vs-frequency")
async def get_sheer_modulus_vs_frequency(upload_id: str = None):
    print("Upload paths:")
    global upload_paths
    print(upload_paths)
    if not upload_id or upload_id not in upload_paths:
        raise HTTPException(status_code=400, detail="No valid upload_id provided or file not found")
    
    file_path = upload_paths[upload_id]
    print(f"Processing file: {file_path}")
    try:
        dataframe_from_csv = extract_dataframe_from_csv(file_path)
        graph_data = calculate_sheer_modulus_vs_frequency(dataframe_from_csv)
        return {
            "status": "success",
            "graphData": graph_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/get-relaxtion-modulus-vs-time")
async def get_relaxtation_modulus_with_time(upload_id: str = None, a_upper_bound: float = 500, d_upper_bound: float = 500):
    global upload_paths, cached_shift_factors
    if not upload_id or upload_id not in upload_paths:
        raise HTTPException(status_code=400, detail="No valid upload_id provided or file not found")
    
    file_path = upload_paths[upload_id]
    try:
        data = extract_dataframe_from_csv(file_path)
        
        if cached_shift_factors.get(upload_id) is None:
            cached_shift_factors[upload_id] = calculate_shift_factors(data)
            print("Shift Factors:", cached_shift_factors[upload_id])

        if cached_shift_factors.get(upload_id) is None:
            raise ValueError("Shift factors could not be calculated.")
        
        graph_data = calculate_relaxation_modulus_vs_time(data, cached_shift_factors[upload_id], a_upper_bound=a_upper_bound, d_upper_bound=d_upper_bound)
        return {
            "status": "success",    
            "graphData": graph_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/get-master-curve-for-all-temperatures")
async def get_master_curve_for_all_temperatures(upload_id: str = None, a_upper_bound: float = 500, d_upper_bound: float = 500):
    global upload_paths
    if not upload_id or upload_id not in upload_paths:
        raise HTTPException(status_code=400, detail="No valid upload_id provided or file not found")
    
    file_path = upload_paths[upload_id]
    try:
        data = extract_dataframe_from_csv(file_path)
        graph_data = calculate_master_curve_graph_data(data, a_upper_bound=a_upper_bound, d_upper_bound=d_upper_bound)
        return {
            "status": "success",
            "graphData": graph_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))