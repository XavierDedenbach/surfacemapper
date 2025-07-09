import numpy as np
import pytest
import logging
from app.services import volume_calculator

@pytest.fixture(autouse=True)
def capture_warnings(caplog):
    caplog.set_level(logging.WARNING)
    yield caplog

def test_volume_calculation_with_utm_coordinates(capture_warnings):
    # UTM coordinates (meters)
    bottom = np.array([[500000, 4100000, 0], [500100, 4100000, 0], [500100, 4100100, 0], [500000, 4100100, 0]], dtype=np.float32)
    top = bottom.copy()
    top[:, 2] += 10  # 10 meters above
    vol = volume_calculator.calculate_volume_between_surfaces(bottom, top)
    assert vol > 0
    assert not any('WGS84' in r.message for r in capture_warnings.records)

def test_volume_calculation_with_wgs84_coordinates(capture_warnings):
    # WGS84 coordinates (degrees)
    bottom = np.array([[-120.0, 35.0, 0], [-120.0, 35.001, 0], [-119.999, 35.001, 0], [-119.999, 35.0, 0]], dtype=np.float32)
    top = bottom.copy()
    top[:, 2] += 10
    vol = volume_calculator.calculate_volume_between_surfaces(bottom, top)
    assert vol > 0
    assert any('WGS84' in r.message for r in capture_warnings.records)

def test_surface_area_with_utm_coordinates(capture_warnings):
    surface = np.array([[500000, 4100000, 0], [500100, 4100000, 0], [500100, 4100100, 0], [500000, 4100100, 0]], dtype=np.float32)
    area = volume_calculator.calculate_surface_area(surface)
    assert area > 0
    assert not any('WGS84' in r.message for r in capture_warnings.records)

def test_surface_area_with_wgs84_coordinates(capture_warnings):
    surface = np.array([[-120.0, 35.0, 0], [-120.0, 35.001, 0], [-119.999, 35.001, 0], [-119.999, 35.0, 0]], dtype=np.float32)
    area = volume_calculator.calculate_surface_area(surface)
    assert area > 0
    assert any('WGS84' in r.message for r in capture_warnings.records) 