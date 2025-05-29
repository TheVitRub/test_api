from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import io
from PIL import Image
import base64
from typing import Optional

from models.database import ModelDatabase
from models.segmentation import ForestSegmentator

app = FastAPI(title="Forest Segmentation Service")

# Инициализируем компоненты
db = ModelDatabase()
segmentator = ForestSegmentator()


@app.post("/segment")
async def segment_forest(
        image: UploadFile = File(...),
        id_model: int = Form(...),
        scale: float = Form(...)
):
    """
    Выполняет сегментацию леса на изображении

    Args:
        image: Изображение для сегментации
        id_model: ID модели в базе данных
        scale: Масштаб (квадратных метров на пиксель)

    Returns:
        JSON с сегментированным изображением и площадью леса
    """
    try:
        # Получаем путь к модели из БД
        model_path = db.get_model_path(id_model)
        if not model_path:
            raise HTTPException(status_code=404, detail=f"Model with id {id_model} not found")

        # Загружаем модель
        segmentator.load_model(model_path)

        # Читаем изображение
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))

        # Конвертируем в RGB если необходимо
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Выполняем сегментацию
        segmented_image, forest_area, forest_percentage = segmentator.segment_image(pil_image, scale)

        # Конвертируем результат в base64
        buffered = io.BytesIO()
        segmented_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return JSONResponse({
            "status": "success",
            "segmented_image": f"data:image/png;base64,{img_base64}",
            "forest_area_m2": round(forest_area, 2),
            "forest_percentage": round(forest_percentage, 2),
            "message": f"Forest covers {forest_percentage:.1f}% of the image, total area: {forest_area:.0f} m²"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Forest Segmentation Service is running"}


@app.get("/models")
async def get_available_models():
    """Получить список доступных моделей"""
    # Здесь можно добавить логику для получения всех моделей из БД
    return {"models": [{"id": 1, "name": "Forest Segmentation Model"}]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)