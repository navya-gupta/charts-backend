# utils/sql_queries.py

CREATE_DATABASE = """
CREATE DATABASE IF NOT EXISTS {db_name}
"""

CREATE_USER_UPLOADS_TABLE = """
CREATE TABLE IF NOT EXISTS user_uploads (
    id VARCHAR(255) PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    university VARCHAR(255) NOT NULL,
    department VARCHAR(255) NOT NULL,
    machine_used VARCHAR(255) NOT NULL,
    model_number VARCHAR(255) NOT NULL,
    email_address VARCHAR(255) NOT NULL,
    timestamp DATETIME NOT NULL
)
"""

INSERT_USER_UPLOAD = """
INSERT INTO user_uploads (id, full_name, university, department, machine_used, model_number, email_address, timestamp)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""