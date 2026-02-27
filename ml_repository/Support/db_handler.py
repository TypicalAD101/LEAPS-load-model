import numpy as np
import sqlite3


class DBHandler:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)

    def create_db_table(self, table_name):
        cursor = self.conn.cursor()
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actual REAL,
            predicted REAL,
            abs_diff REAL
        );
        """
        cursor.execute(create_table_query)
        self.conn.commit()

    def insert_data(self, table_name, actual, predicted, abs_diff):
        cursor = self.conn.cursor()
        insert_query = f"""
        INSERT INTO {table_name} (actual, predicted, abs_diff)
        VALUES (?, ?, ?);
        """
        
        # Robust conversion to Python float
        def to_float(val):
            if isinstance(val, (np.ndarray, list)):
                val = val[0] if len(val) > 0 else 0.0
            if hasattr(val, 'item'):
                val = val.item()
            return float(val)
        
        actual_float = to_float(actual)
        predicted_float = to_float(predicted)
        abs_diff_float = to_float(abs_diff)
        
        cursor.execute(insert_query, (actual_float, predicted_float, abs_diff_float))
        self.conn.commit()

    def close(self):
        self.conn.close()