import requests
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt

# Загружаем оригинальное изображение
original_img = Image.open("10452_sat_08 (1).jpg")

url = "http://localhost:8000/segment"
files = {"image": open("10452_sat_08 (1).jpg", "rb")}
data = {
    "id_model": 1,
    "scale": 0.5  # 0.5 м² на пиксель
}

response = requests.post(url, files=files, data=data)
result = response.json()

# Декодируем сегментированное изображение
base64_str = result['segmented_image'].split(',')[1]
img_bytes = base64.b64decode(base64_str)
segmented_img = Image.open(io.BytesIO(img_bytes))

# Показываем оба изображения
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

ax1.imshow(original_img)
ax1.set_title("Оригинальное изображение")
ax1.axis('off')

ax2.imshow(segmented_img)
ax2.set_title(f"Сегментированное изображение\nПлощадь леса: {result['forest_area_m2']} м² ({result['forest_percentage']:.1f}%)")
ax2.axis('off')

plt.tight_layout()
plt.show()

# Выводим статистику
print(result)
print(f"Площадь леса: {result['forest_area_m2']} м²")
print(f"Процент леса: {result['forest_percentage']}%")
print(f"Сообщение: {result['message']}")