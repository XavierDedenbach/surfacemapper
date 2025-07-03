import numpy as np
from app.services.surface_processor import SurfaceProcessor

# Create test vertices
vertices = np.array([
    [5.0, 5.0, 1.0],   # Inside polygon
    [15.0, 5.0, 2.0],  # Inside polygon
    [10.0, 15.0, 3.0], # Inside polygon
    [20.0, 20.0, 4.0], # Outside polygon
    [0.0, 0.0, 5.0],   # Outside polygon
])

# Define polygon boundary (diamond shape)
boundary = [
    (10.0, 0.0),   # Bottom
    (20.0, 10.0),  # Right
    (10.0, 20.0),  # Top
    (0.0, 10.0)    # Left
]

surface_processor = SurfaceProcessor()

# Test each point individually
for i, vertex in enumerate(vertices):
    point = vertex[:2]  # x, y coordinates
    is_inside = surface_processor._is_point_in_polygon(point, np.array(boundary))
    print(f"Point {i}: {point} -> Inside: {is_inside}")

# Test the full clipping
clipped_vertices = surface_processor.clip_to_boundary(vertices, boundary)
print(f"\nClipped vertices count: {len(clipped_vertices)}")
print(f"Clipped vertices: {clipped_vertices}") 