"""
FastAPI inference service for Titanic survival prediction.

This module provides a REST API service for making predictions using the trained
Titanic survival model. It includes authentication, health checks, single and batch
prediction endpoints, with comprehensive logging and error handling.
"""

from datetime import datetime
import os
import pickle
import time

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import mlflow
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.base import BaseEstimator

from src.data_models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.globals import logger


class InferenceService:
    """Inference service for the Titanic survival prediction model"""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the inference service.

        Args:
            cfg (DictConfig): Configuration object containing model paths and settings
        """
        self.cfg = cfg
        self.model: BaseEstimator = None
        self.model_loaded = False
        self.model_name = cfg["names"]["model_name"]

        # Authentication
        load_dotenv()
        self.api_key = os.getenv("API_KEY")

        logger.info("Inference service initialized")

    def load_model(self) -> None:
        """Load the trained model and preprocessor"""
        try:
            start_time = time.time()
            if self.cfg["mlflow"]["is_tracking_enabled"]:
                mlflow.set_tracking_uri(self.cfg["mlflow"]["tracking_uri"])
                client = mlflow.tracking.MlflowClient()
                version = client.get_latest_versions(name=self.model_name)[0].version
                logger.error(f"version: {version}")
                model_uri = f"models:/{self.model_name}/{version}"
                self.model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Loaded model from MLflow: {model_uri}")
            else:
                # Fallback to pickle file
                model_path = os.path.join(
                    self.cfg["paths"]["models_parent_dir"], f"{self.model_name}.pkl"
                )

                if os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        self.model = pickle.load(f)
                    logger.info(f"Loaded model from pickle: {model_path}")
                else:
                    raise FileNotFoundError(f"Model not found at {model_path}")

            self.model_loaded = True
            load_time = time.time() - start_time

            logger.success(f"Model and preprocessor loaded successfully in {load_time:.3f}s")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def verify_api_key(
        self, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
    ) -> bool:
        """Verify API key authentication"""
        if credentials.credentials != self.api_key:
            logger.warning(f"Invalid API key attempt: {credentials.credentials[:10]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return True

    def _prepare_input_data(self, request: PredictionRequest) -> pd.DataFrame:
        """
        Convert request to DataFrame for prediction.

        Args:
            request (PredictionRequest): Input prediction request

        Returns:
            pd.DataFrame: Formatted DataFrame ready for model prediction
        """
        data = {
            "PassengerId": request.PassengerId or 0,
            "Pclass": request.Pclass,
            "Name": request.Name,
            "Sex": request.Sex,
            "Age": request.Age,
            "SibSp": request.SibSp,
            "Parch": request.Parch,
            "Ticket": request.Ticket,
            "Fare": request.Fare,
            "Cabin": request.Cabin,
            "Embarked": request.Embarked,
        }

        return pd.DataFrame([data])

    def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """Make prediction for a single passenger"""
        if not self.model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
            )

        start_time = time.time()

        try:
            # Prepare input data
            df = self._prepare_input_data(request)

            # Make prediction
            prediction = self.model.predict(df)

            prediction_time = time.time() - start_time

            logger.info(
                f"Prediction made for passenger {request.PassengerId}: "
                f"time={prediction_time:.3f}s"
            )

            return PredictionResponse(
                prediction=int(prediction),
                passenger_id=request.PassengerId,
                prediction_time=prediction_time,
            )

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}",
            )

    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Make predictions for multiple passengers"""
        if not self.model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
            )

        start_time = time.time()
        predictions = []

        logger.info(f"Starting batch prediction for {len(request.passengers)} passengers")

        try:
            for passenger in request.passengers:
                prediction = self.predict_single(passenger)
                predictions.append(prediction)

            batch_time = time.time() - start_time

            logger.info(
                f"Batch prediction completed: {len(predictions)} predictions in {batch_time:.3f}s"
            )

            return BatchPredictionResponse(
                predictions=predictions,
                total_predictions=len(predictions),
                batch_processing_time=batch_time,
            )

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch prediction failed: {str(e)}",
            )


cfg = OmegaConf.load("config.yaml")
inference_service = InferenceService(cfg=cfg)

# Create FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="ML inference API for predicting Titanic passenger survival with authentication and detailed logging",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        inference_service.load_model()
        logger.info("FastAPI application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")

    return HealthResponse(
        status="healthy" if inference_service.model_loaded else "unhealthy",
        model_loaded=inference_service.model_loaded,
        model_name=inference_service.model_name,
        timestamp=datetime.now(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest, authenticated: bool = Depends(inference_service.verify_api_key)
):
    """Make a prediction for a single passenger"""
    logger.info(f"Single prediction request received for passenger {request.PassengerId}")
    return inference_service.predict_single(request)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    authenticated: bool = Depends(inference_service.verify_api_key),
):
    """Make predictions for multiple passengers"""
    logger.info(f"Batch prediction request received for {len(request.passengers)} passengers")
    return inference_service.predict_batch(request)


@app.get("/openapi.json", include_in_schema=False)
async def get_openapi(
    authenticated: bool = Depends(inference_service.verify_api_key)
):
    """Custom OpenAPI schema endpoint"""
    return app.openapi()


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI inference server...")
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
