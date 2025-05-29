import numpy as np
import cv2
from typing import Tuple


def calculate_forest_area(mask: np.ndarray, scale: float) -> float:
    """
    Вычисляет площадь леса на основе маски сегментации и масштаба

    Args:
        mask: Бинарная маска сегментации
        scale: Масштаб (квадратных метров на пиксель)

    Returns:
        Площадь леса в квадратных метрах
    """
    # Подсчитываем количество пикселей леса
    forest_pixels = np.sum(mask > 0)

    # Вычисляем площадь
    area = forest_pixels * (scale ** 2)

    return area


def merge_overlapping_segments(mask: np.ndarray) -> np.ndarray:
    """
    Объединяет перекрывающиеся или близкие сегменты в один

    Args:
        mask: Исходная маска сегментации

    Returns:
        Объединенная маска
    """
    # Преобразуем маску в бинарную
    binary_mask = (mask > 0).astype(np.uint8) * 255

    # Применяем морфологические операции для объединения близких областей
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # Закрытие для соединения близких областей
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Открытие для удаления мелких шумов
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Находим все контуры
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем новую маску
    merged_mask = np.zeros_like(mask)

    # Заполняем все контуры
    cv2.drawContours(merged_mask, contours, -1, 255, -1)

    return merged_mask


def calculate_forest_percentage(mask: np.ndarray) -> float:
    """
    Вычисляет процент площади, занимаемой лесом

    Args:
        mask: Маска сегментации

    Returns:
        Процент площади леса
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    forest_pixels = np.sum(mask > 0)

    percentage = (forest_pixels / total_pixels) * 100

    return percentage