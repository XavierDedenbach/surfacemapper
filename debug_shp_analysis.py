import fiona
import shapely.geometry as sgeom
from shapely.geometry import Point, Polygon, MultiPoint, LineString
import os

def analyze_shp_file(file_path):
    """Analyze the geometry types in a SHP file"""
    print(f"Analyzing SHP file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        print("File not found!")
        return
    
    try:
        with fiona.open(file_path, 'r') as src:
            print(f"CRS: {src.crs}")
            print(f"Schema: {src.schema}")
            print(f"Total features: {len(src)}")
            
            geom_types = set()
            feature_count = 0
            
            print("\nFirst 5 features:")
            for feature in src:
                geom = sgeom.shape(feature['geometry'])
                geom_type = type(geom).__name__
                geom_types.add(geom_type)
                
                if feature_count < 5:
                    print(f"  Feature {feature_count + 1}: {geom_type}")
                    print(f"    Geometry: {geom}")
                    print(f"    Properties: {feature['properties']}")
                    print()
                
                feature_count += 1
                
                if feature_count >= 5:
                    break
            
            print(f"Unique geometry types found: {geom_types}")
            print(f"Total features analyzed: {feature_count}")
            
    except Exception as e:
        print(f"Error analyzing SHP file: {e}")

if __name__ == "__main__":
    shp_file = "drone_surfaces/27June20250541PM1619tonspartialcover/27June20250541PM1619tonspartialcover.shp"
    analyze_shp_file(shp_file) 