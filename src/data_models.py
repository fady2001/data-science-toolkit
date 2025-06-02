from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request model for single prediction"""

    PassengerId: Optional[int] = Field(None, description="Passenger ID")
    Pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    Name: str = Field(..., min_length=1, description="Passenger name")
    Sex: str = Field(..., description="Gender (male or female)")
    Age: Optional[float] = Field(None, ge=0, le=150, description="Age in years")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Ticket: str = Field(..., min_length=1, description="Ticket number")
    Fare: Optional[float] = Field(None, ge=0, description="Passenger fare")
    Cabin: Optional[str] = Field(None, description="Cabin number")
    Embarked: Optional[str] = Field(None, description="Port of embarkation (C, Q, S)")

    @field_validator("Sex")
    def validate_sex(cls, v):
        if v.lower() not in ["male", "female"]:
            raise ValueError('Sex must be either "male" or "female"')
        return v.lower()

    @field_validator("Embarked")
    def validate_embarked(cls, v):
        if v is not None and v.upper() not in ["C", "Q", "S"]:
            raise ValueError("Embarked must be one of: C, Q, S")
        return v.upper() if v else None
    
class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""

    passengers: List[PredictionRequest] = Field(..., min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    prediction: int = Field(..., description="Survival prediction (0: No, 1: Yes)")
    passenger_id: Optional[int] = Field(None, description="Passenger ID if provided")
    prediction_time: float = Field(..., description="Prediction time in seconds")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""

    predictions: List[PredictionResponse]
    total_predictions: int
    batch_processing_time: float


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    model_name: str
    timestamp: datetime