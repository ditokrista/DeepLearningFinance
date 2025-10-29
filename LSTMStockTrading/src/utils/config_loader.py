"""
Configuration Loader
Centralized configuration management for the trading system
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Centralized configuration loader with environment variable support.

    Features:
    - Load YAML configuration files
    - Support for environment-specific configs
    - Environment variable interpolation
    - Configuration validation
    - Singleton pattern for consistent config access
    """

    _instance = None
    _config_cache: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.config_dir = Path(__file__).parent.parent.parent / "config"

        # Load environment variables
        load_dotenv()

        logger.info(f"Configuration directory: {self.config_dir}")

    def load_config(self, config_name: str, reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_name: Name of config file (without .yaml extension)
            reload: Force reload from disk, ignore cache

        Returns:
            Configuration dictionary
        """
        if config_name in self._config_cache and not reload:
            logger.debug(f"Loading {config_name} from cache")
            return self._config_cache[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            # Try subdirectories
            for subdir in self.config_dir.iterdir():
                if subdir.is_dir():
                    potential_path = subdir / f"{config_name}.yaml"
                    if potential_path.exists():
                        config_path = potential_path
                        break

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_name}.yaml")

        logger.info(f"Loading configuration from: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Interpolate environment variables
        config = self._interpolate_env_vars(config)

        # Cache the config
        self._config_cache[config_name] = config

        return config

    def _interpolate_env_vars(self, config: Any) -> Any:
        """
        Recursively interpolate environment variables in config.
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        """
        if isinstance(config, dict):
            return {k: self._interpolate_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._interpolate_env_vars(item) for item in config]
        elif isinstance(config, str):
            if config.startswith("${") and config.endswith("}"):
                var_spec = config[2:-1]
                if ":" in var_spec:
                    var_name, default = var_spec.split(":", 1)
                    return os.getenv(var_name, default)
                else:
                    return os.getenv(var_spec, config)
        return config

    def get(self, config_name: str, *keys, default: Any = None) -> Any:
        """
        Get a specific configuration value using dot notation.

        Args:
            config_name: Configuration file name
            *keys: Nested keys to traverse
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            config.get('base_config', 'system', 'logging', 'level')
        """
        config = self.load_config(config_name)

        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable."""
        return os.getenv(key, default)

    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files.

        Returns:
            Dictionary mapping config names to their contents
        """
        all_configs = {}

        # Load main configs
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            all_configs[config_name] = self.load_config(config_name)

        # Load subdirectory configs
        for subdir in self.config_dir.iterdir():
            if subdir.is_dir():
                for config_file in subdir.glob("*.yaml"):
                    config_name = config_file.stem
                    all_configs[config_name] = self.load_config(config_name)

        return all_configs

    def validate_config(self, config_name: str, required_keys: list) -> bool:
        """
        Validate that required keys exist in configuration.

        Args:
            config_name: Configuration file name
            required_keys: List of required keys (supports dot notation)

        Returns:
            True if all required keys exist

        Raises:
            ValueError: If required keys are missing
        """
        config = self.load_config(config_name)
        missing_keys = []

        for key_path in required_keys:
            keys = key_path.split('.')
            value = config

            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    missing_keys.append(key_path)
                    break

        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys in {config_name}: {missing_keys}"
            )

        return True

    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """
        Merge multiple configurations together.
        Later configs override earlier ones.

        Args:
            *config_names: Configuration file names to merge

        Returns:
            Merged configuration dictionary
        """
        merged = {}

        for config_name in config_names:
            config = self.load_config(config_name)
            merged = self._deep_merge(merged, config)

        return merged

    def _deep_merge(self, base: dict, update: dict) -> dict:
        """Recursively merge two dictionaries."""
        result = base.copy()

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def clear_cache(self):
        """Clear the configuration cache."""
        self._config_cache.clear()
        logger.info("Configuration cache cleared")


# Singleton instance
config_loader = ConfigLoader()


def load_config(config_name: str) -> Dict[str, Any]:
    """Convenience function to load configuration."""
    return config_loader.load_config(config_name)


def get_config_value(config_name: str, *keys, default: Any = None) -> Any:
    """Convenience function to get specific config value."""
    return config_loader.get(config_name, *keys, default=default)


if __name__ == "__main__":
    # Test configuration loading
    logging.basicConfig(level=logging.INFO)

    # Load base config
    base_config = load_config("base_config")
    print("Base config loaded successfully")
    print(f"System name: {base_config['system']['name']}")
    print(f"Initial capital: {base_config['trading']['risk']['max_position_size']}")

    # Load model config
    model_config = load_config("lstm_default")
    print("\nModel config loaded successfully")
    print(f"Model type: {model_config['model']['type']}")
    print(f"Hidden dim: {model_config['model']['architecture']['hidden_dim']}")

    # Test get method
    log_level = get_config_value("base_config", "system", "logging", "level")
    print(f"\nLog level: {log_level}")

    print("\nConfiguration system working correctly!")
