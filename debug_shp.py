import fiona
print('Fiona version:', fiona.__version__)
print('Available drivers:')
for k, v in fiona.supported_drivers.items():
    print(f'  {k}: {v}')

import tempfile
import os
from shapely.geometry import Point

# Create test points
test_points = [Point(-122.4194, 37.7749)]

# Create temporary directory
temp_dir = tempfile.mkdtemp()
print(f"Created temp dir: {temp_dir}")

temp_path = os.path.join(temp_dir, 'test')
print(f"Temp path: {temp_path}")

# Create schema
schema = {'geometry': 'Point', 'properties': [('id', 'int')]}

try:
    # Write SHP file
    with fiona.open(temp_path, 'w', driver='ESRI Shapefile', 
                   schema=schema, crs='EPSG:4326') as dst:
        dst.write({
            'geometry': test_points[0].__geo_interface__,
            'properties': {'id': 0}
        })
    print("File written successfully")
    
    # Check what files were created
    shp_path = temp_path + '.shp'
    print(f"SHP path: {shp_path}")
    print(f"File exists: {os.path.exists(shp_path)}")
    print(f"Dir contents: {os.listdir(temp_dir)}")
    
    # Try to read the file
    with fiona.open(shp_path, 'r') as src:
        features = list(src)
        print(f"Read {len(features)} features")
        print(f"First feature: {features[0]}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Clean up
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True) 