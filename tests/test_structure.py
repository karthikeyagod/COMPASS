"""
Phase 0 Architecture & Structure Verification Tests.

Verifies that all modules in `src/` can be imported cleanly, docstrings are present,
and `config.yaml` can be successfully parsed.
"""

import importlib
import os
from pathlib import Path
import unittest


class TestProjectStructure(unittest.TestCase):
    """
    Sanity test suite verifying the integrity of the project scaffolding.
    """

    def setUp(self) -> None:
        self.root_dir = Path(__file__).resolve().parent.parent

    def test_required_directories_exist(self) -> None:
        """Verify that all top-level project directories exist."""
        required_dirs = [
            "data",
            "models",
            "src",
            "training",
            "api",
            "dashboard",
            "tests",
            "notebooks",
        ]
        for dir_name in required_dirs:
            dir_path = self.root_dir / dir_name
            self.assertTrue(dir_path.exists() and dir_path.is_dir(), f"Missing directory: {dir_name}")

    def test_required_root_files_exist(self) -> None:
        """Verify that all top-level config and documentation files exist."""
        required_files = [
            "requirements.txt",
            "README.md",
            ".gitignore",
            "config.yaml",
        ]
        for file_name in required_files:
            file_path = self.root_dir / file_name
            self.assertTrue(file_path.exists() and file_path.is_file(), f"Missing file: {file_name}")

    def test_all_src_modules_importable(self) -> None:
        """Verify that all core src modules can be imported without error."""
        modules_to_test = [
            "src",
            "src.config",
            "src.data_loader",
            "src.validation",
            "src.preprocessing",
            "src.feature_engineering",
            "src.anomaly_detection",
            "src.drift_prediction",
            "src.safety_slope",
            "src.risk_engine",
            "src.explainability",
            "src.inference",
        ]
        for module_name in modules_to_test:
            mod = importlib.import_module(module_name)
            self.assertIsNotNone(mod, f"Failed to import module: {module_name}")
            self.assertTrue(bool(mod.__doc__), f"Module {module_name} is missing a docstring.")

    def test_config_loader(self) -> None:
        """Verify that src.config successfully loads config.yaml."""
        from src.config import Config, load_config

        config_data = load_config()
        self.assertIsInstance(config_data, dict)
        self.assertIn("project", config_data)
        self.assertEqual(config_data["project"]["name"], "sih-260170-latent-defect-detection")

        cfg_wrapper = Config()
        self.assertEqual(cfg_wrapper.get("project")["name"], "sih-260170-latent-defect-detection")


if __name__ == "__main__":
    unittest.main()
