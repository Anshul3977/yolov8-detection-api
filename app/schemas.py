from pydantic import BaseModel, Field


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionOut(BaseModel):
    class_id: int
    label: str
    confidence: float = Field(ge=0, le=1)
    box: Box


class ImageSize(BaseModel):
    width: int
    height: int


class TimingsOut(BaseModel):
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


class DetectResponse(BaseModel):
    model: str
    filename: str | None
    image: ImageSize
    count: int
    detections: list[DetectionOut]
    timings: TimingsOut


class BatchItem(BaseModel):
    filename: str | None
    image: ImageSize
    count: int
    detections: list[DetectionOut]


class BatchResponse(BaseModel):
    model: str
    batch_size: int
    results: list[BatchItem]
    timings: TimingsOut


class HealthResponse(BaseModel):
    status: str
    model: str
    providers: list[str]
    classes: int
    imgsz: int
