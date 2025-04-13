# utils/sql_queries.py

CREATE_DATABASE = """
CREATE DATABASE IF NOT EXISTS {db_name}
"""

CREATE_USER_UPLOADS_TABLE = """
CREATE TABLE IF NOT EXISTS user_uploads (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    university VARCHAR(255) NOT NULL,
    designation VARCHAR(255) NOT NULL,
    machine_used VARCHAR(255) NOT NULL,
    csv_content LONGTEXT,
    csv_name VARCHAR(255) NOT NULL,
    timestamp DATETIME NOT NULL
)
"""

INSERT_USER_UPLOAD = """
INSERT INTO user_uploads (id, name, university, designation, machine_used, csv_content, csv_name, timestamp)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""