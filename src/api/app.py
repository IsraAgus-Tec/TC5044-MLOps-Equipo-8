from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from joblib import load

MODEL_PATH = Path("models/energy_efficiency_rf.joblib")
METADATA_PATH = MODEL_PATH.with_suffix(".json")


class EnergySample(BaseModel):
    relative_compactness: float = Field(..., ge=0, le=1.5)
    surface_area: float = Field(..., gt=0)
    wall_area: float = Field(..., gt=0)
    roof_area: float = Field(..., gt=0)
    overall_height: float = Field(..., gt=0)
    orientation: float = Field(..., ge=1, le=8)
    glazing_area: float = Field(..., ge=0)
    glazing_area_distribution: float = Field(..., ge=0)


class PredictionResult(BaseModel):
    heating_load: float
    cooling_load: float


class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]


class PredictionPayload(BaseModel):
    samples: List[EnergySample]


app = FastAPI(
    title="Energy Efficiency API",
    description="FastAPI service that serves the energy efficiency regression model.",
    version="1.0.0",
)


def _load_artifacts():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No se encontró el modelo en {MODEL_PATH}. Ejecuta `python -m src.main` para entrenar y exportarlo."
        )
    if not METADATA_PATH.exists():
        raise RuntimeError(
            f"No se encontró el archivo de metadatos en {METADATA_PATH}. Ejecuta el pipeline para regenerarlo."
        )
    metadata = json.loads(METADATA_PATH.read_text())
    model = load(MODEL_PATH)
    feature_columns = metadata.get("feature_columns")
    if not feature_columns:
        raise RuntimeError("El archivo de metadatos no contiene la lista de columnas de entrada.")
    return model, feature_columns, metadata


MODEL, FEATURE_COLUMNS, MODEL_METADATA = _load_artifacts()


@app.get("/healthz", tags=["health"])
def health_check():
    return {
        "status": "ok",
        "model": MODEL_METADATA.get("model_name"),
        "artifact_path": MODEL_METADATA.get("artifact_path"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: PredictionPayload):
    if not payload.samples:
        raise HTTPException(status_code=400, detail="Payload vacío: proporciona al menos un registro en 'samples'.")

    df = pd.DataFrame([sample.dict() for sample in payload.samples])
    df = df[FEATURE_COLUMNS]
    try:
        preds = MODEL.predict(df)
    except Exception as exc:  # pragma: no cover - guard against model issues
        raise HTTPException(status_code=500, detail=f"Error al inferir: {exc}") from exc

    results = [
        PredictionResult(heating_load=float(pred[0]), cooling_load=float(pred[1]))
        for pred in preds
    ]
    return PredictionResponse(predictions=results)
