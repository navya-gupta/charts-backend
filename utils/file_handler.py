import os
from fastapi import UploadFile

UPLOAD_DIR = "static/uploads"     

# utils/file_handler.py
def save_upload_file(upload_file: UploadFile, custom_filename: str) -> str:
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_location = os.path.join(UPLOAD_DIR, custom_filename)
    with open(file_location, "wb") as f:
        f.write(upload_file.file.read())  # Ensure this reads the content
    return file_location