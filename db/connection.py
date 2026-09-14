"""
MindMap AI - Database Connection Manager
Provides unified connection handling for MySQL with graceful fallback to SQLite
per design.md §31 and RMD.md §15.
"""

import sys
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
import pandas as pd
import mysql.connector
from mysql.connector import Error as MySQLError

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

SQLITE_DB_PATH = config.DATA_DIR / "processed" / "mindmap_ai.db"


def get_mysql_connection(create_db_if_missing: bool = True) -> Optional[mysql.connector.MySQLConnection]:
    """
    Attempts to establish a connection to the configured MySQL database.
    """
    try:
        if create_db_if_missing:
            # First connect without specifying database to create it if needed
            init_conn = mysql.connector.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
            )
            cursor = init_conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.DB_NAME};")
            cursor.close()
            init_conn.close()

        conn = mysql.connector.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME,
        )
        return conn
    except MySQLError as e:
        return None
    except Exception:
        return None


def get_sqlite_connection() -> sqlite3.Connection:
    """
    Returns a SQLite connection for local fallback analytics.
    """
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(SQLITE_DB_PATH))


def get_connection() -> Tuple[Any, str]:
    """
    Returns (connection_object, engine_type) where engine_type is 'mysql' or 'sqlite'.
    """
    mysql_conn = get_mysql_connection(create_db_if_missing=True)
    if mysql_conn is not None and mysql_conn.is_connected():
        return mysql_conn, "mysql"
    
    return get_sqlite_connection(), "sqlite"


def execute_query(query: str, params: Optional[Tuple] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes a SQL query against MySQL (or SQLite fallback) and returns the result as a DataFrame.
    
    Returns:
    - (df_result, metadata_dict)
    """
    conn, engine = get_connection()
    meta = {"engine": engine, "success": True, "error": None}

    try:
        if engine == "mysql":
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
            conn.close()
            return df, meta
        else:
            # SQLite fallback
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
            conn.close()
            return df, meta
    except Exception as e:
        if conn:
            conn.close()
        meta["success"] = False
        meta["error"] = str(e)
        return pd.DataFrame(), meta


def check_db_status() -> Dict[str, Any]:
    """
    Checks database health and returns connection status.
    """
    conn, engine = get_connection()
    if engine == "mysql" and conn is not None:
        conn.close()
        return {
            "status": "connected",
            "engine": "MySQL",
            "details": f"Connected to MySQL at {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}",
        }
    else:
        if conn:
            conn.close()
        return {
            "status": "fallback",
            "engine": "SQLite / Local DB",
            "details": f"Operating on local embedded database at {SQLITE_DB_PATH.name} (MySQL fallback active)",
        }
