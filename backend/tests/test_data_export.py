import pytest
import tempfile
import os
import json
import pandas as pd
from pathlib import Path
from app.services.data_export import DataExporter
from app.models.data_models import (
    AnalysisResults, VolumeResult, ThicknessResult, CompactionResult,
    StatisticalAnalysis, QualityMetrics, DetailedAnalysisReport
)


class TestDataExport:
    @pytest.fixture
    def data_exporter(self):
        return DataExporter()

    @pytest.fixture
    def sample_analysis_results(self):
        """Sample analysis results for testing"""
        return AnalysisResults(
            volume_results=[
                VolumeResult(
                    layer_designation="Surface 0 to Surface 1",
                    volume_cubic_yards=1000.5,
                    confidence_interval=(950.0, 1050.0),
                    uncertainty=50.0
                )
            ],
            thickness_results=[
                ThicknessResult(
                    layer_designation="Surface 0 to Surface 1",
                    average_thickness_feet=2.5,
                    min_thickness_feet=1.0,
                    max_thickness_feet=4.0,
                    confidence_interval=(2.0, 3.0)
                )
            ],
            compaction_results=[
                CompactionResult(
                    layer_designation="Surface 0 to Surface 1",
                    compaction_rate_lbs_per_cubic_yard=120.0,
                    tonnage_used=50.0
                )
            ],
            processing_metadata={
                "processing_time": 120.5,
                "points_processed": 10000,
                "algorithm_version": "1.0"
            }
        )

    @pytest.fixture
    def sample_statistical_data(self):
        """Sample statistical data for testing"""
        return StatisticalAnalysis(
            mean_value=10.5,
            median_value=10.0,
            standard_deviation=2.0,
            variance=4.0,
            skewness=0.1,
            kurtosis=-0.2,
            sample_count=1000,
            confidence_interval_95=(9.5, 11.5),
            percentiles={"25": 8.0, "50": 10.0, "75": 12.0, "90": 14.0, "95": 15.0, "99": 18.0}
        )

    @pytest.fixture
    def sample_surface_data(self):
        """Sample surface data for testing"""
        return {
            "vertices": [[0, 0, 0], [1, 0, 1], [0, 1, 2], [1, 1, 3]],
            "faces": [[0, 1, 2], [1, 3, 2]],
            "metadata": {
                "surface_name": "Test Surface",
                "point_count": 4,
                "face_count": 2
            }
        }

    def test_export_analysis_results_csv(self, data_exporter, sample_analysis_results):
        """Test exporting analysis results to CSV"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            file_path = data_exporter.export_analysis_results(
                sample_analysis_results, "csv", tmp_file.name
            )
            
            assert os.path.exists(file_path)
            assert file_path.endswith('.csv')
            
            # Verify CSV content
            df = pd.read_csv(file_path)
            assert len(df) > 0
            assert 'layer_designation' in df.columns
            
            os.unlink(file_path)

    def test_export_analysis_results_json(self, data_exporter, sample_analysis_results):
        """Test exporting analysis results to JSON"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            file_path = data_exporter.export_analysis_results(
                sample_analysis_results, "json", tmp_file.name
            )
            
            assert os.path.exists(file_path)
            assert file_path.endswith('.json')
            
            # Verify JSON content
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            assert 'volume_results' in data
            assert 'thickness_results' in data
            assert 'compaction_results' in data
            
            os.unlink(file_path)

    def test_export_analysis_results_excel(self, data_exporter, sample_analysis_results):
        """Test exporting analysis results to Excel"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            file_path = data_exporter.export_analysis_results(
                sample_analysis_results, "excel", tmp_file.name
            )
            
            assert os.path.exists(file_path)
            assert file_path.endswith('.xlsx')
            
            # Verify Excel content
            excel_file = pd.ExcelFile(file_path)
            assert len(excel_file.sheet_names) > 0
            
            os.unlink(file_path)

    def test_export_statistical_data_csv(self, data_exporter, sample_statistical_data):
        """Test exporting statistical data to CSV"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            file_path = data_exporter.export_statistical_data(
                sample_statistical_data, "csv", tmp_file.name
            )
            
            assert os.path.exists(file_path)
            assert file_path.endswith('.csv')
            
            # Verify CSV content
            df = pd.read_csv(file_path)
            assert len(df) > 0
            assert 'mean_value' in df.columns or 'statistic' in df.columns
            
            os.unlink(file_path)

    def test_export_statistical_data_json(self, data_exporter, sample_statistical_data):
        """Test exporting statistical data to JSON"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            file_path = data_exporter.export_statistical_data(
                sample_statistical_data, "json", tmp_file.name
            )
            
            assert os.path.exists(file_path)
            assert file_path.endswith('.json')
            
            # Verify JSON content
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            assert 'mean_value' in data
            assert 'standard_deviation' in data
            assert 'percentiles' in data
            
            os.unlink(file_path)

    def test_export_surface_data_ply(self, data_exporter, sample_surface_data):
        """Test exporting surface data to PLY format"""
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
            file_path = data_exporter.export_surface_data(
                sample_surface_data, "ply", tmp_file.name
            )
            
            assert os.path.exists(file_path)
            assert file_path.endswith('.ply')
            
            # Verify PLY content
            with open(file_path, 'r') as f:
                content = f.read()
            
            assert 'ply' in content.lower()
            assert 'format ascii' in content.lower()
            assert 'element vertex' in content.lower()
            assert 'element face' in content.lower()
            
            os.unlink(file_path)

    def test_export_surface_data_obj(self, data_exporter, sample_surface_data):
        """Test exporting surface data to OBJ format"""
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp_file:
            file_path = data_exporter.export_surface_data(
                sample_surface_data, "obj", tmp_file.name
            )
            
            assert os.path.exists(file_path)
            assert file_path.endswith('.obj')
            
            # Verify OBJ content
            with open(file_path, 'r') as f:
                content = f.read()
            
            assert 'v ' in content  # vertices
            assert 'f ' in content  # faces
            
            os.unlink(file_path)

    def test_export_surface_data_stl(self, data_exporter, sample_surface_data):
        """Test exporting surface data to STL format"""
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
            file_path = data_exporter.export_surface_data(
                sample_surface_data, "stl", tmp_file.name
            )
            
            assert os.path.exists(file_path)
            assert file_path.endswith('.stl')
            
            # Verify STL content
            with open(file_path, 'r') as f:
                content = f.read()
            
            assert 'solid' in content.lower()
            assert 'facet normal' in content.lower()
            assert 'endfacet' in content.lower()
            
            os.unlink(file_path)

    def test_export_surface_data_xyz(self, data_exporter, sample_surface_data):
        """Test exporting surface data to XYZ format"""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp_file:
            file_path = data_exporter.export_surface_data(
                sample_surface_data, "xyz", tmp_file.name
            )
            
            assert os.path.exists(file_path)
            assert file_path.endswith('.xyz')
            
            # Verify XYZ content
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) > 0
            # Check that lines contain 3 space-separated numbers
            for line in lines[:5]:  # Check first 5 lines
                if line.strip():
                    coords = line.strip().split()
                    assert len(coords) == 3
                    # Check that all values can be converted to float
                    for coord in coords:
                        float(coord)  # This will raise ValueError if not convertible
            
            os.unlink(file_path)

    def test_export_invalid_format(self, data_exporter, sample_analysis_results):
        """Test exporting with invalid format"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            with pytest.raises(ValueError):
                data_exporter.export_analysis_results(
                    sample_analysis_results, "invalid_format", tmp_file.name
                )
            
            os.unlink(tmp_file.name)

    def test_export_invalid_file_path(self, data_exporter, sample_analysis_results):
        """Test exporting to invalid file path"""
        with pytest.raises(ValueError):
            data_exporter.export_analysis_results(
                sample_analysis_results, "csv", "/invalid/path/file.csv"
            )

    def test_export_empty_data(self, data_exporter):
        """Test exporting empty data"""
        empty_results = AnalysisResults(
            volume_results=[],
            thickness_results=[],
            compaction_results=[],
            processing_metadata={}
        )
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            file_path = data_exporter.export_analysis_results(
                empty_results, "csv", tmp_file.name
            )
            
            assert os.path.exists(file_path)
            # Should create file even with empty data
            
            os.unlink(file_path)

    def test_export_large_dataset(self, data_exporter):
        """Test exporting large dataset"""
        # Create large dataset
        large_results = AnalysisResults(
            volume_results=[
                VolumeResult(
                    layer_designation=f"Layer {i}",
                    volume_cubic_yards=float(i * 100),
                    confidence_interval=(float(i * 95), float(i * 105)),
                    uncertainty=float(i * 5)
                ) for i in range(1000)
            ],
            thickness_results=[
                ThicknessResult(
                    layer_designation=f"Layer {i}",
                    average_thickness_feet=float(i * 0.1),
                    min_thickness_feet=float(i * 0.05),
                    max_thickness_feet=float(i * 0.15),
                    confidence_interval=(float(i * 0.08), float(i * 0.12))
                ) for i in range(1000)
            ],
            compaction_results=[
                CompactionResult(
                    layer_designation=f"Layer {i}",
                    compaction_rate_lbs_per_cubic_yard=float(i * 10),
                    tonnage_used=float(i * 5)
                ) for i in range(1000)
            ],
            processing_metadata={"large_dataset": True}
        )
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            import time
            start_time = time.time()
            
            file_path = data_exporter.export_analysis_results(
                large_results, "csv", tmp_file.name
            )
            
            end_time = time.time()
            
            assert os.path.exists(file_path)
            assert (end_time - start_time) < 30.0  # Should complete in reasonable time
            
            os.unlink(file_path)

    def test_export_with_metadata(self, data_exporter, sample_analysis_results):
        """Test exporting with additional metadata"""
        metadata = {
            "export_timestamp": "2024-01-01T12:00:00Z",
            "export_format": "csv",
            "data_version": "1.0"
        }
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            file_path = data_exporter.export_analysis_results(
                sample_analysis_results, "csv", tmp_file.name, metadata=metadata
            )
            
            assert os.path.exists(file_path)
            
            # Verify metadata is included in export
            df = pd.read_csv(file_path)
            assert len(df) > 0
            
            os.unlink(file_path)

    def test_export_multiple_formats_same_data(self, data_exporter, sample_analysis_results):
        """Test exporting same data to multiple formats"""
        formats = ["csv", "json", "xlsx"]
        files = []
        
        try:
            for fmt in formats:
                ext = fmt if fmt != "xlsx" else "xlsx"
                with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp_file:
                    file_path = data_exporter.export_analysis_results(
                        sample_analysis_results, fmt if fmt != "xlsx" else "excel", tmp_file.name
                    )
                    files.append(file_path)
                    assert os.path.exists(file_path)
                    assert file_path.endswith(f'.{ext}')
        finally:
            # Clean up
            for file_path in files:
                if os.path.exists(file_path):
                    os.unlink(file_path)

    def test_export_performance_benchmark(self, data_exporter, sample_analysis_results):
        """Test export performance"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            import time
            
            # Benchmark CSV export
            start_time = time.time()
            file_path = data_exporter.export_analysis_results(
                sample_analysis_results, "csv", tmp_file.name
            )
            csv_time = time.time() - start_time
            
            # Benchmark JSON export
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_json:
                start_time = time.time()
                json_path = data_exporter.export_analysis_results(
                    sample_analysis_results, "json", tmp_json.name
                )
                json_time = time.time() - start_time
                
                os.unlink(tmp_json.name)
            
            # Both should complete quickly
            assert csv_time < 5.0
            assert json_time < 5.0
            
            os.unlink(file_path) 