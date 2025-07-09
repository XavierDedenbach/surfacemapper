import os
import re
import pytest
from pathlib import Path

class TestDocumentationWorkflow:
    """Test that developer documentation specifies the new UTM workflow correctly"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.docs_dir = Path("../docs")
        self.config_dir = Path("../config")
        
        # Required documentation files
        self.required_files = [
            "../docs/algorithm_specifications.md",
            "../docs/api_documentation.md",
            "../config/processing_defaults.yaml"
        ]
        
        # Required workflow keywords
        self.workflow_keywords = [
            "UTM coordinates",
            "mesh operations",
            "project together",
            "WGS84 to UTM",
            "coordinate system",
            "meters",
            "projection"
        ]
        
        # Required workflow steps
        self.workflow_steps = [
            "Load SHP file in WGS84",
            "Project both mesh and boundary to UTM",
            "Perform all mesh operations in UTM",
            "Calculate area and volume in UTM",
            "No mesh operations in WGS84"
        ]
    
    def test_required_documentation_files_exist(self):
        """Test that all required documentation files exist"""
        for file_path in self.required_files:
            full_path = Path(file_path)
            assert full_path.exists(), f"Required documentation file {file_path} does not exist"
    
    def test_algorithm_specifications_workflow(self):
        """Test that algorithm specifications document the UTM workflow"""
        spec_file = "../docs/algorithm_specifications.md"
        assert os.path.exists(spec_file), f"Algorithm specifications file {spec_file} does not exist"
        
        with open(spec_file, 'r') as f:
            content = f.read()
        
        # Check for coordinate system requirements section
        assert "Coordinate System Requirements" in content, "Algorithm specifications must include Coordinate System Requirements section"
        
        # Check for workflow steps
        for step in self.workflow_steps:
            assert step in content, f"Algorithm specifications must include workflow step: {step}"
        
        # Check for required keywords
        for keyword in self.workflow_keywords:
            assert keyword in content, f"Algorithm specifications must include keyword: {keyword}"
    
    def test_api_documentation_workflow(self):
        """Test that API documentation specifies coordinate system requirements"""
        api_file = "../docs/api_documentation.md"
        assert os.path.exists(api_file), f"API documentation file {api_file} does not exist"
        
        with open(api_file, 'r') as f:
            content = f.read()
        
        # Check for coordinate system requirements section
        assert "Coordinate System Requirements" in content, "API documentation must include Coordinate System Requirements section"
        
        # Check for UTM requirements
        assert "UTM coordinates" in content, "API documentation must specify UTM coordinate requirements"
        assert "meters" in content, "API documentation must specify units in meters"
        
        # Check for workflow information
        assert "SHP files" in content, "API documentation must mention SHP file handling"
        assert "PLY files" in content, "API documentation must mention PLY file handling"
    
    def test_configuration_workflow(self):
        """Test that configuration specifies the new workflow"""
        config_file = "../config/processing_defaults.yaml"
        assert os.path.exists(config_file), f"Configuration file {config_file} does not exist"
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check for coordinate system configuration
        assert "coordinate_system:" in content, "Configuration must include coordinate_system section"
        assert "required_for_mesh_operations: \"UTM\"" in content, "Configuration must specify UTM requirement"
        
        # Check for projection workflow configuration
        assert "projection:" in content, "Configuration must include projection section"
        assert "project_immediately: true" in content, "Configuration must specify immediate projection"
        assert "project_mesh_and_boundary_together: true" in content, "Configuration must specify projection together"
    
    def test_workflow_clarity(self):
        """Test that workflow steps are clearly documented"""
        spec_file = "../docs/algorithm_specifications.md"
        
        with open(spec_file, 'r') as f:
            content = f.read()
        
        # Check for clear workflow sections
        assert "SHP Workflow" in content, "Documentation must include SHP Workflow section"
        assert "PLY Workflow" in content, "Documentation must include PLY Workflow section"
        
        # Check for numbered steps
        step_pattern = r'\d+\.\s+'
        steps = re.findall(step_pattern, content)
        assert len(steps) >= 4, "Documentation must include numbered workflow steps"
    
    def test_coordinate_system_validation(self):
        """Test that coordinate system validation is documented"""
        spec_file = "../docs/algorithm_specifications.md"
        
        with open(spec_file, 'r') as f:
            content = f.read()
        
        # Check for validation requirements
        assert "validation" in content.lower(), "Documentation must include validation requirements"
        assert "warning" in content.lower(), "Documentation must mention warning system"
        
        # Check for coordinate system checks
        assert "coordinate system" in content.lower(), "Documentation must mention coordinate system checks"
    
    def test_developer_guidelines_completeness(self):
        """Test that developer guidelines are complete"""
        spec_file = "../docs/algorithm_specifications.md"
        
        with open(spec_file, 'r') as f:
            content = f.read()
        
        # Check for developer guidance
        required_sections = [
            "Coordinate System Requirements",
            "SHP Workflow",
            "PLY Workflow",
            "Validation"
        ]
        
        for section in required_sections:
            assert section in content, f"Documentation must include section: {section}"
    
    def test_no_outdated_information(self):
        """Test that documentation doesn't contain outdated information"""
        spec_file = "../docs/algorithm_specifications.md"
        
        with open(spec_file, 'r') as f:
            content = f.read()
        
        # Check for absence of outdated terms
        outdated_terms = [
            "WGS84 mesh operations",
            "WGS84 triangulation",
            "WGS84 volume calculation",
            "degrees for calculations"
        ]
        
        for term in outdated_terms:
            assert term not in content, f"Documentation should not contain outdated term: {term}"
    
    def test_examples_provided(self):
        """Test that documentation provides examples"""
        spec_file = "../docs/algorithm_specifications.md"
        
        with open(spec_file, 'r') as f:
            content = f.read()
        
        # Check for example sections or code blocks
        assert "```" in content, "Documentation should include code examples"
        assert "example" in content.lower(), "Documentation should include examples"
    
    def test_workflow_consistency(self):
        """Test that workflow is consistent across documentation"""
        spec_file = "../docs/algorithm_specifications.md"
        api_file = "../docs/api_documentation.md"
        
        with open(spec_file, 'r') as f:
            spec_content = f.read()
        
        with open(api_file, 'r') as f:
            api_content = f.read()
        
        # Check that both documents mention the same workflow
        workflow_terms = ["UTM coordinates", "mesh operations", "project together"]
        
        for term in workflow_terms:
            assert term in spec_content, f"Algorithm specifications must include: {term}"
            assert term in api_content, f"API documentation must include: {term}"
    
    def test_configuration_consistency(self):
        """Test that configuration is consistent with documentation"""
        config_file = "../config/processing_defaults.yaml"
        spec_file = "../docs/algorithm_specifications.md"
        
        with open(config_file, 'r') as f:
            config_content = f.read()
        
        with open(spec_file, 'r') as f:
            spec_content = f.read()
        
        # Check that configuration matches documentation
        assert "UTM" in config_content, "Configuration must specify UTM"
        assert "UTM" in spec_content, "Documentation must specify UTM"
        
        assert "project_immediately" in config_content, "Configuration must specify immediate projection"
        assert "project together" in spec_content, "Documentation must specify projection together" 