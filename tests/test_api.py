import io

from PIL import Image


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["classes"] == 80
    assert "CPUExecutionProvider" in body["providers"]


def test_detect_finds_bus_and_people(client, bus_bytes):
    r = client.post("/detect", files={"file": ("bus.jpg", bus_bytes, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    labels = [d["label"] for d in body["detections"]]
    assert "bus" in labels
    assert labels.count("person") >= 3
    assert body["image"] == {"width": 810, "height": 1080}
    for d in body["detections"]:
        b = d["box"]
        assert 0 <= b["x1"] < b["x2"] <= 810 and 0 <= b["y1"] < b["y2"] <= 1080
        assert d["confidence"] >= 0.25


def test_conf_threshold_filters(client, bus_bytes):
    files = {"file": ("bus.jpg", bus_bytes, "image/jpeg")}
    low = client.post("/detect?conf=0.1", files=files).json()["count"]
    high = client.post("/detect?conf=0.8", files=files).json()["count"]
    assert high < low


def test_batch_matches_single(client, bus_bytes, zidane_bytes):
    files = [("files", ("bus.jpg", bus_bytes, "image/jpeg")), ("files", ("zidane.jpg", zidane_bytes, "image/jpeg"))]
    r = client.post("/detect/batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["batch_size"] == 2
    single = client.post("/detect", files={"file": ("bus.jpg", bus_bytes, "image/jpeg")}).json()
    assert body["results"][0]["count"] == single["count"]
    assert "person" in [d["label"] for d in body["results"][1]["detections"]]


def test_png_input(client):
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), "white").save(buf, format="PNG")
    r = client.post("/detect", files={"file": ("blank.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_rejects_non_image(client):
    r = client.post("/detect", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_oversized_batch(client, bus_bytes):
    files = [("files", (f"{i}.jpg", bus_bytes, "image/jpeg")) for i in range(17)]
    assert client.post("/detect/batch", files=files).status_code == 413


def test_metrics_exposed(client, bus_bytes):
    client.post("/detect", files={"file": ("bus.jpg", bus_bytes, "image/jpeg")})
    text = client.get("/metrics").text
    assert "model_inference_duration_seconds_bucket" in text
    assert 'http_requests_total{method="POST",path="/detect",status="200"}' in text
    assert 'detections_total{label="bus"}' in text


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "YOLOv8" in r.text
