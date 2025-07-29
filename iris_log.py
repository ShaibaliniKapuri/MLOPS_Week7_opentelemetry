from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import time
import json
import joblib
import pandas as pd
import numpy as np

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# 1. SETUP TRACING
# -----------------
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# 2. SETUP STRUCTURED LOGGING
# ---------------------------
logger = logging.getLogger("iris-classifier-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '{"severity": "%(levelname)s", "message": "%(message)s", "timestamp": "%(asctime)s"}'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


# 3. FASTAPI APP & STATE
# ----------------------
app = FastAPI(title="🌸 Iris Classifier API with Logging & Tracing")

# Application state now includes the model object
app_state = {"is_ready": False, "is_alive": True, "model": None}


# 4. INPUT SCHEMA
# ---------------
# Updated to match the features of the Iris dataset
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# 5. APPLICATION LIFECYCLE EVENTS
# -------------------------------
@app.on_event("startup")
async def startup_event():
    """
    On startup, load the Iris classifier model and set the app to ready.
    """
    logger.info(json.dumps({"event": "startup", "message": "Loading model..."}))
    time.sleep(2)  # Simulate work (e.g., downloading model)
    try:
        # Load the pre-trained model from the file
        app_state["model"] = joblib.load("model.joblib")
        app_state["is_ready"] = True
        logger.info(json.dumps({"event": "startup", "message": "Model loaded successfully. Service is ready."}))
    except FileNotFoundError:
        logger.error(json.dumps({
            "event": "startup_error",
            "error": "model.joblib not found",
            "message": "Model file is missing. The service will not become ready."
        }))
        app_state["is_alive"] = False # Or handle as a non-fatal error if appropriate
    except Exception as e:
        logger.error(json.dumps({
            "event": "startup_error",
            "error": str(e),
            "message": "An unexpected error occurred during model loading."
        }))
        app_state["is_alive"] = False


# 6. PROBES & MIDDLEWARE (Unchanged but still important)
# -----------------------------------------------------
@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

# 7. EXCEPTION HANDLER (Unchanged)
# --------------------------------
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x") if span.is_recording() else "N/A"
    logger.error(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

# 8. PREDICTION ENDPOINT (Updated)
# --------------------------------
@app.post("/predict", tags=["Prediction"])
async def predict(input_data: IrisInput):
    """
    Receives Iris features, predicts the species using the loaded model,
    and logs the transaction.
    """
    if not app_state["is_ready"] or app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Service is not ready; the model has not been loaded.")

    with tracer.start_as_current_span("iris_model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            # Convert Pydantic model to a dictionary for logging
            input_dict = input_data.dict()
            
            # Create a DataFrame for the model
            input_df = pd.DataFrame([input_dict])

            # Use the loaded model to make a prediction
            prediction = app_state["model"].predict(input_df)
            result = {"predicted_class": prediction[0]}

            latency = round((time.time() - start_time) * 1000, 2)

            # Log the successful prediction details
            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": input_dict,
                "result": result,
                "latency_ms": latency,
                "status": "success"
            }))

            return result

        except Exception as e:
            # Log any errors that occur during prediction
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail=f"Prediction failed. Trace ID: {trace_id}")