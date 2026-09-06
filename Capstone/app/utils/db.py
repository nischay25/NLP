import sqlite3
import os
from datetime import datetime
from backend.app.config import settings

def get_db_connection():
    # Ensure directory exists
    db_dir = os.path.dirname(settings.DATABASE_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summarization_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            filename TEXT,
            input_word_count INTEGER,
            summary TEXT,
            summary_word_count INTEGER,
            processing_time REAL,
            model_used TEXT
        )
    """)
    conn.commit()
    conn.close()

def add_history(filename: str, input_word_count: int, summary: str, summary_word_count: int, processing_time: float, model: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO summarization_history (filename, input_word_count, summary, summary_word_count, processing_time, model_used)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (filename, input_word_count, summary, summary_word_count, processing_time, model))
    conn.commit()
    conn.close()

def get_history(search_query: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if search_query:
        cursor.execute("""
            SELECT * FROM summarization_history 
            WHERE filename LIKE ? OR summary LIKE ? 
            ORDER BY date DESC
        """, (f"%{search_query}%", f"%{search_query}%"))
    else:
        cursor.execute("SELECT * FROM summarization_history ORDER BY date DESC")
    
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        result.append({
            "id": row["id"],
            "date": row["date"],
            "filename": row["filename"],
            "input_word_count": row["input_word_count"],
            "summary": row["summary"],
            "summary_word_count": row["summary_word_count"],
            "processing_time": row["processing_time"],
            "model_used": row["model_used"]
        })
    return result

def delete_history_item(item_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM summarization_history WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
