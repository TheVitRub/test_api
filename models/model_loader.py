import torch
from ultralytics import YOLO
import os
import tempfile
import shutil


class SafeModelLoader:
    """Безопасный загрузчик для YOLO моделей"""

    @staticmethod
    def load_yolo_model(model_path: str):
        """
        Загружает YOLO модель с обработкой различных версий PyTorch
        """
        try:
            # Метод 1: Прямая загрузка
            return YOLO(model_path)
        except Exception as e:
            print(f"Direct loading failed: {e}")

            try:
                # Метод 2: Загрузка с отключенным weights_only
                # Создаем временную копию модели
                temp_dir = tempfile.mkdtemp()
                temp_model_path = os.path.join(temp_dir, "temp_model.pt")
                shutil.copy2(model_path, temp_model_path)

                # Загружаем checkpoint
                checkpoint = torch.load(temp_model_path, map_location='cpu', weights_only=False)

                # Сохраняем с новыми настройками
                torch.save(checkpoint, temp_model_path, _use_new_zipfile_serialization=False)

                # Загружаем модель
                model = YOLO(temp_model_path)

                # Удаляем временные файлы
                shutil.rmtree(temp_dir)

                return model

            except Exception as e2:
                print(f"Alternative loading failed: {e2}")

                # Метод 3: Использование конфигурации YOLO
                try:
                    # Пытаемся создать модель из конфигурации и загрузить веса
                    model = YOLO('yolov8n-seg.yaml')  # Базовая конфигурация для сегментации
                    model.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
                    return model
                except:
                    raise Exception("Failed to load model with all methods")