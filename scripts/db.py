from __future__ import annotations

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('POSTGRES_HOST'),
    database=os.getenv('POSTGRES_DB'),
    user=os.getenv('POSTGRES_USER'),
    password=os.getenv('POSTGRES_PASSWORD'),
    port=os.getenv('POSTGRES_PORT'),
)

cur = conn.cursor()

create_table_query = """
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    base64_image TEXT NOT NULL,
    probability FLOAT NOT NULL,
    predicted_class VARCHAR(255) NOT NULL,
    alternatives JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
cur.execute(create_table_query)
conn.commit()
