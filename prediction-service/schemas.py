from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class PredictRequest(BaseModel):
    data: Dict[str, Any]

class PredictResponse(BaseModel):
    prediction: str
    probabilities: Optional[List[float]]
    class_labels: List[str] = Field(default_factory=list)
    algorithm: str
    feature_names: List[str]
