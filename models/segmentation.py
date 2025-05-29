import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional
import torch
import warnings
import os
import sys

# Добавляем ultralytics в безопасные глобальные переменные
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.SegmentationModel'])

from ultralytics import YOLO
from .utils import merge_overlapping_segments, calculate_forest_area, calculate_forest_percentage


class ForestSegmentator:
    def __init__(self):
        self.model = None
        self.current_model_path = None

    def load_model(self, model_path: str):
        """Загружает модель YOLO"""
        if self.current_model_path != model_path:
            try:
                # Устанавливаем переменную окружения для ultralytics
                os.environ['YOLO_VERBOSE'] = 'False'

                # Подавляем предупреждения
                warnings.filterwarnings('ignore')

                # Проверяем существование файла
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")

                # Загружаем модель
                # Для PyTorch >= 2.6 нужно явно указать weights_only=False
                if hasattr(torch, '__version__') and torch.__version__ >= '2.6':
                    # Временно меняем поведение torch.load
                    original_load = torch.load

                    def custom_load(*args, **kwargs):
                        kwargs['weights_only'] = False
                        return original_load(*args, **kwargs)

                    torch.load = custom_load
                    self.model = YOLO(model_path)
                    torch.load = original_load
                else:
                    self.model = YOLO(model_path)

                self.current_model_path = model_path
                print(f"Model loaded successfully from: {model_path}")

            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise Exception(f"Failed to load model from {model_path}: {str(e)}")

    def segment_image(self, image: Image.Image, scale: float) -> Tuple[Image.Image, float, float]:
        """
        Выполняет сегментацию изображения и вычисляет площадь леса

        Args:
            image: Входное изображение PIL
            scale: Масштаб (м²/пиксель)

        Returns:
            Tuple[сегментированное изображение, площадь леса, процент леса]
        """
        if self.model is None:
            raise Exception("Model not loaded")

        # Конвертируем PIL изображение в numpy array
        img_array = np.array(image)

        # Выполняем сегментацию
        results = self.model(img_array, verbose=False)

        # Получаем маски сегментации
        if results[0].masks is not None:
            # Получаем все маски
            masks = results[0].masks.data.cpu().numpy()

            # Создаем объединенную маску для всех сегментов леса
            combined_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)

            for mask in masks:
                # Изменяем размер маски под размер изображения
                resized_mask = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]))
                # Добавляем к общей маске
                combined_mask = np.maximum(combined_mask, (resized_mask > 0.5).astype(np.uint8) * 255)

            # Объединяем перекрывающиеся сегменты
            merged_mask = merge_overlapping_segments(combined_mask)

            # Вычисляем площадь и процент
            forest_area = calculate_forest_area(merged_mask, scale)
            forest_percentage = calculate_forest_percentage(merged_mask)

            # Создаем визуализацию
            segmented_image = self._create_visualization(img_array, merged_mask)

            return Image.fromarray(segmented_image), forest_area, forest_percentage
        else:
            # Если сегментов не найдено, возвращаем исходное изображение
            return image, 0.0, 0.0

    def _create_visualization(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Создает визуализацию с наложением маски на изображение"""
        # Создаем цветную маску (зеленый цвет для леса)
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 1] = mask  # Зеленый канал

        # Накладываем маску на изображение с прозрачностью
        alpha = 0.4
        result = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

        # Добавляем контур для лучшей видимости
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        return result