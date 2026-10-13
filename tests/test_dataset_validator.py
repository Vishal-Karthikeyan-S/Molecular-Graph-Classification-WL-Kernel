"""
Unit tests for dataset integrity validator.
"""

import os
import tempfile
import pytest
from scripts.validate_dataset import validate_nci1_directory


def test_validate_nonexistent_directory():
    """Validates that a non-existent directory returns False gracefully."""
    result = validate_nci1_directory("non_existent_dir_xyz123")
    assert result is False


def test_validate_missing_files_in_directory():
    """Validates that an incomplete directory missing NCI1 files returns False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nci1_dir = os.path.join(tmpdir, "NCI1")
        os.makedirs(nci1_dir)
        # Create only one file instead of all 4 required files
        with open(os.path.join(nci1_dir, "NCI1_A.txt"), "w") as f:
            f.write("1, 2\n")
        
        result = validate_nci1_directory(tmpdir)
        assert result is False


def test_validate_complete_synthetic_directory():
    """Validates that a synthetic directory with all required files passes validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nci1_dir = os.path.join(tmpdir, "NCI1")
        os.makedirs(nci1_dir)
        required_files = [
            "NCI1_A.txt",
            "NCI1_graph_indicator.txt",
            "NCI1_graph_labels.txt",
            "NCI1_node_labels.txt"
        ]
        for f in required_files:
            with open(os.path.join(nci1_dir, f), "w") as fp:
                fp.write("dummy\n")

        result = validate_nci1_directory(tmpdir)
        assert result is True
