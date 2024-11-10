from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import logging
import pickle
from models import ModelManager

app = FastAPI()
model_manager = ModelManager()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelManager")

class TrainRequest(BaseModel):
    """
    Модель запроса для обучения модели.
    
    Attributes:
        model_type (str): Тип модели для обучения.
        hyperparameters (Dict[str, Any]): Гиперпараметры для модели.
        training_data (List[List[float]]): Обучающие данные.
        target_data (List[float]): Целевые значения для обучения.
        name (str): Имя модели (необязательно).
    """
    model_type: str
    hyperparameters: Dict[str, Any]
    training_data: List[List[float]]
    target_data: List[float]
    name: str = None

class PredictRequest(BaseModel):
    """
    Модель запроса для предсказания.
    
    Attributes:
        model_id (str): ID модели для предсказания.
        data (List[List[float]]): Данные для предсказания.
        filename (str): Имя файла для сохранения предсказаний.
    """
    model_id: str
    data: List[List[float]]
    filename: str

class RetrainRequest(BaseModel):
    """
    Модель запроса для переобучения модели.
    
    Attributes:
        model_id (str): ID модели для переобучения.
        data (List[List[float]]): Данные для переобучения.
        target (List[float]): Целевые значения для переобучения.
        filename (str): Имя файла для сохранения предсказаний.
    """
    model_id: str
    data: List[List[float]]
    target: List[float]
    filename: str

@app.get("/status")
def get_status():
    """
    Проверяет статус сервиса.
    
    Returns:
        dict: Словарь с информацией о статусе сервиса.
    """
    logger.info("Проверка статуса сервиса")
    return {"status": "Сервис работает"}

@app.get("/models")
def get_available_models():
    """
    Получает список доступных типов моделей.
    
    Returns:
        list: Список доступных типов моделей.
    """
    logger.info("Получение доступных типов моделей")
    return model_manager.get_available_model_types()

@app.get("/models/info")
def get_models_info():
    """
    Получает информацию обо всех обученных моделях.
    
    Returns:
        dict: Словарь с информацией об обученных моделях.
    """
    logger.info("Получение информации обо всех обученных моделях")
    return model_manager.get_models_info()

@app.post("/train")
def train_model(request: TrainRequest):
    """
    Обучает модель на основе предоставленных данных и гиперпараметров.
    
    Args:
        request (TrainRequest): Запрос на обучение модели.

    Returns:
        dict: Словарь с ID обученной модели.

    Raises:
        HTTPException: Если произошла ошибка при обучении модели.
    """
    logger.info(f"Обучение модели типа {request.model_type} с гиперпараметрами {request.hyperparameters} и именем {request.name}")
    try:
        model_id = model_manager.train_model(request.model_type, request.hyperparameters, request.training_data, request.target_data, request.name)
        return {"model_id": model_id}
    except ValueError as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(request: PredictRequest):
    """
    Делает предсказание с использованием указанной модели и сохраняет результаты в файл.
    
    Args:
        request (PredictRequest): Запрос на предсказание.

    Returns:
        dict: Словарь с информацией о сохранении предсказаний.

    Raises:
        HTTPException: Если произошла ошибка при предсказании.
    """
    logger.info(f"Предсказание с моделью {request.model_id}")
    try:
        predictions = model_manager.predict(request.model_id, request.data)
        with open(request.filename, 'wb') as f:
            pickle.dump(predictions, f)
        return {"detail": "Предсказания сохранены", "filename": request.filename}
    except ValueError as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    """
    Удаляет модель по её ID.
    
    Args:
        model_id (str): ID модели для удаления.

    Returns:
        dict: Словарь с информацией об удалении модели.

    Raises:
        HTTPException: Если произошла ошибка при удалении модели.
    """
    logger.info(f"Удаление модели {model_id}")
    try:
        model_manager.delete_model(model_id)
        return {"detail": "Модель удалена"}
    except ValueError as e:
        logger.error(f"Ошибка при удалении модели: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain")
def retrain_model(request: RetrainRequest):
    """
    Переобучает существующую модель на новых данных и сохраняет результаты в файл.
    
    Args:
        request (RetrainRequest): Запрос на переобучение модели.

    Returns:
        dict: Словарь с информацией о переобучении модели и сохранении предсказаний.

    Raises:
        HTTPException: Если произошла ошибка при переобучении модели.
    """
    logger.info(f"Переобучение модели {request.model_id}")
    try:
        predictions = model_manager.retrain_model(request.model_id, request.data, request.target)
        with open(request.filename, 'wb') as f:
            pickle.dump(predictions, f)
        return {"detail": "Модель переобучена и предсказания сохранены", "filename": request.filename}
    except ValueError as e:
        logger.error(f"Ошибка при переобучении модели: {e}")
        raise HTTPException(status_code=400, detail=str(e))
