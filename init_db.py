import sqlite3
import os


def init_database():
    # Создаем директорию для БД если её нет
    os.makedirs('database', exist_ok=True)

    conn = sqlite3.connect('database/models.db')
    cursor = conn.cursor()

    # Создаем таблицу моделей
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            description TEXT
        )
    ''')

    # Добавляем первую модель
    cursor.execute('''
        INSERT OR REPLACE INTO models (id, name, path, description)
        VALUES (1, 'Forest Segmentation Model', 'weights/forest_segmentation_model_V1.pt', 
                'YOLO model for forest segmentation')
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully!")


if __name__ == "__main__":
    init_database()