import pytest
import numpy as np
from app.services.volume_calculator import VolumeCalculator

class TestCompactionAnalysis:
    @pytest.fixture
    def volume_calculator(self):
        return VolumeCalculator()

    def test_single_layer_compaction(self, volume_calculator):
        volume = 100.0  # cubic yards
        tonnage = 50.0  # tons
        expected = (tonnage * 2000) / volume
        result = volume_calculator.calculate_compaction_rate(volume, tonnage)
        assert result == expected
        assert result == 1000.0

    def test_multiple_layers_compaction(self, volume_calculator):
        # Simulate multiple layers with different volumes and tonnages
        layers = [
            {'volume': 100.0, 'tonnage': 50.0},
            {'volume': 200.0, 'tonnage': 80.0},
            {'volume': 50.0, 'tonnage': 30.0},
        ]
        expected_rates = [
            (50.0 * 2000) / 100.0,
            (80.0 * 2000) / 200.0,
            (30.0 * 2000) / 50.0,
        ]
        for layer, expected in zip(layers, expected_rates):
            result = volume_calculator.calculate_compaction_rate(layer['volume'], layer['tonnage'])
            assert result == expected

    def test_zero_volume(self, volume_calculator):
        result = volume_calculator.calculate_compaction_rate(0.0, 50.0)
        assert result == 0.0

    def test_negative_volume(self, volume_calculator):
        result = volume_calculator.calculate_compaction_rate(-100.0, 50.0)
        assert result == 0.0

    def test_zero_tonnage(self, volume_calculator):
        result = volume_calculator.calculate_compaction_rate(100.0, 0.0)
        assert result == 0.0

    def test_negative_tonnage(self, volume_calculator):
        # Negative tonnage is not physically meaningful, but test for robustness
        result = volume_calculator.calculate_compaction_rate(100.0, -50.0)
        assert result == -1000.0

    def test_large_values(self, volume_calculator):
        result = volume_calculator.calculate_compaction_rate(1e6, 1e4)
        assert result == (1e4 * 2000) / 1e6

    def test_missing_input(self, volume_calculator):
        # Should raise TypeError if arguments are missing
        with pytest.raises(TypeError):
            volume_calculator.calculate_compaction_rate(100.0)
        with pytest.raises(TypeError):
            volume_calculator.calculate_compaction_rate()

    def test_invalid_input_types(self, volume_calculator):
        # Should raise TypeError for non-numeric input
        with pytest.raises(TypeError):
            volume_calculator.calculate_compaction_rate('a', 50.0)
        with pytest.raises(TypeError):
            volume_calculator.calculate_compaction_rate(100.0, 'b')

    def test_summary_statistics(self, volume_calculator):
        # If required: mean, min, max compaction rate for a set of layers
        layers = [
            {'volume': 100.0, 'tonnage': 50.0},
            {'volume': 200.0, 'tonnage': 80.0},
            {'volume': 50.0, 'tonnage': 30.0},
        ]
        rates = [volume_calculator.calculate_compaction_rate(l['volume'], l['tonnage']) for l in layers]
        mean_rate = np.mean(rates)
        min_rate = np.min(rates)
        max_rate = np.max(rates)
        assert mean_rate > 0
        assert min_rate > 0
        assert max_rate > 0
        assert min_rate <= mean_rate <= max_rate 