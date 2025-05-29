import sqlite3
from typing import Optional


class ModelDatabase:
    def __init__(self, db_path: str = "database/models.db"):
        self.db_path = db_path

    def get_model_path(self, model_id: int) -> Optional[str]:
        """Получить путь к модели по ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT path FROM models WHERE id = ?", (model_id,))
        result = cursor.fetchone()

        conn.close()

        return result[0] if result else None