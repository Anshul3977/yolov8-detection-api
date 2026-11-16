from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as c:  # runs lifespan -> loads the ONNX model
        yield c


@pytest.fixture
def bus_bytes() -> bytes:
    return (ASSETS / "bus.jpg").read_bytes()


@pytest.fixture
def zidane_bytes() -> bytes:
    return (ASSETS / "zidane.jpg").read_bytes()
