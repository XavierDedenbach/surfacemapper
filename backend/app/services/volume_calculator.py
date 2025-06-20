"""
Volume and thickness calculation service using PyVista algorithms
"""
import numpy as np
import pyvista as pv
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class VolumeResult:
    """Volume calculation result"""
    volume_cubic_yards: float
    confidence_interval: Tuple[float, float]
    uncertainty: float

@dataclass
class ThicknessResult:
    """Thickness calculation result"""
    average_thickness_feet: float
    min_thickness_feet: float
    max_thickness_feet: float
    confidence_interval: Tuple[float, float]

class VolumeCalculator:
    """
    Handles volume and thickness calculations using PyVista
    """
    
    def __init__(self):
        self.tolerance = 0.01  # 1% tolerance for cross-validation
    
    def calculate_volume_difference(self, surface1: np.ndarray, surface2: np.ndarray) -> VolumeResult:
        """
        Calculate volume difference between two surfaces using PyVista Delaunay triangulation
        """
        try:
            # Convert numpy arrays to PyVista point clouds
            pc1 = pv.PolyData(surface1)
            pc2 = pv.PolyData(surface2)
            
            # Create meshes using PyVista's triangulation
            mesh1 = pc1.delaunay_2d()
            mesh2 = pc2.delaunay_2d()
            
            # Calculate volumes using PyVista's volume calculation
            volume1 = mesh1.volume
            volume2 = mesh2.volume
            
            # Convert from cubic units to cubic yards (assuming input is in feet)
            volume_diff_cubic_yards = abs(volume1 - volume2) / 27.0  # 27 cubic feet = 1 cubic yard
            
            # Calculate confidence interval (simplified for now)
            uncertainty = volume_diff_cubic_yards * 0.05  # 5% uncertainty
            confidence_interval = (
                volume_diff_cubic_yards - uncertainty,
                volume_diff_cubic_yards + uncertainty
            )
            
            return VolumeResult(
                volume_cubic_yards=volume_diff_cubic_yards,
                confidence_interval=confidence_interval,
                uncertainty=uncertainty
            )
        except Exception as e:
            # Return zero volume if calculation fails
            return VolumeResult(
                volume_cubic_yards=0.0,
                confidence_interval=(0.0, 0.0),
                uncertainty=0.0
            )
    
    def calculate_layer_thickness(self, upper_surface: np.ndarray, lower_surface: np.ndarray) -> ThicknessResult:
        """
        Calculate layer thickness between two surfaces using PyVista
        """
        try:
            # Convert numpy arrays to PyVista point clouds
            pc_upper = pv.PolyData(upper_surface)
            pc_lower = pv.PolyData(lower_surface)
            
            # Create meshes using PyVista's triangulation
            mesh_upper = pc_upper.delaunay_2d()
            mesh_lower = pc_lower.delaunay_2d()
            
            # Calculate thickness as Z-difference between corresponding points
            # For simplicity, we'll use the average Z-difference
            z_upper = upper_surface[:, 2]
            z_lower = lower_surface[:, 2]
            
            thicknesses = z_upper - z_lower
            thicknesses = thicknesses[thicknesses > 0]  # Only positive thicknesses
            
            if len(thicknesses) == 0:
                return ThicknessResult(
                    average_thickness_feet=0.0,
                    min_thickness_feet=0.0,
                    max_thickness_feet=0.0,
                    confidence_interval=(0.0, 0.0)
                )
            
            average_thickness = np.mean(thicknesses)
            min_thickness = np.min(thicknesses)
            max_thickness = np.max(thicknesses)
            
            # Calculate confidence interval
            std_thickness = np.std(thicknesses)
            confidence_interval = (
                average_thickness - std_thickness,
                average_thickness + std_thickness
            )
            
            return ThicknessResult(
                average_thickness_feet=average_thickness,
                min_thickness_feet=min_thickness,
                max_thickness_feet=max_thickness,
                confidence_interval=confidence_interval
            )
        except Exception as e:
            # Return zero thickness if calculation fails
            return ThicknessResult(
                average_thickness_feet=0.0,
                min_thickness_feet=0.0,
                max_thickness_feet=0.0,
                confidence_interval=(0.0, 0.0)
            )
    
    def calculate_compaction_rate(self, volume_cubic_yards: float, tonnage: float) -> float:
        """
        Calculate compaction rate in lbs/cubic yard
        """
        if volume_cubic_yards <= 0:
            return 0.0
        return (tonnage * 2000) / volume_cubic_yards 