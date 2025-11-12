"""
Configuration Management Module

Handles configuration for models, training, and data processing.
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch
import yaml


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    input_dim: int = 1  # Will be updated based on features
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.3
    output_dim: int = 1
    model_type: str = 'enhanced'  # 'enhanced' or 'simple'


@dataclass
class TrainingConfig:
    """Configuration for training"""
    batch_size: int = 32
    num_epochs: int = 300
    learning_rate: float = 0.001
    patience: int = 30
    gradient_clip: float = 1.0
    seed: int = 42


@dataclass
class DataConfig:
    """Configuration for data processing"""
    stock_symbol: str = "AAPL"
    look_back: int = 60
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    use_technical_indicators: bool = True
    feature_set: str = 'default'  # 'minimal', 'default', 'extended', 'alpha'
    scaler_type: str = 'minmax'  # 'minmax' or 'standard'


@dataclass
class PathConfig:
    """Configuration for file paths"""
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "models" / "training result"
        self.artifacts_dir = self.project_root / "artifacts"

        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Master configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'model': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
                'output_dim': self.model.output_dim,
                'model_type': self.model.model_type,
            },
            'training': {
                'batch_size': self.training.batch_size,
                'num_epochs': self.training.num_epochs,
                'learning_rate': self.training.learning_rate,
                'patience': self.training.patience,
                'gradient_clip': self.training.gradient_clip,
                'seed': self.training.seed,
            },
            'data': {
                'stock_symbol': self.data.stock_symbol,
                'look_back': self.data.look_back,
                'train_ratio': self.data.train_ratio,
                'validation_ratio': self.data.validation_ratio,
                'test_ratio': self.data.test_ratio,
                'use_technical_indicators': self.data.use_technical_indicators,
                'feature_set': self.data.feature_set,
                'scaler_type': self.data.scaler_type,
            }
        }

    def save(self, path):
        """Save configuration to YAML file"""
        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls()

        # Update model config
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

        # Update training config
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)

        # Update data config
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)

        return config

    @classmethod
    def load(cls, path):
        """Load configuration from YAML file"""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def print_config(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("Configuration Summary")
        print("="*60)

        print("\nModel Configuration:")
        print(f"  Type: {self.model.model_type}")
        print(f"  Input dim: {self.model.input_dim}")
        print(f"  Hidden dim: {self.model.hidden_dim}")
        print(f"  Num layers: {self.model.num_layers}")
        print(f"  Dropout: {self.model.dropout}")

        print("\nTraining Configuration:")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Num epochs: {self.training.num_epochs}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Patience: {self.training.patience}")
        print(f"  Gradient clip: {self.training.gradient_clip}")

        print("\nData Configuration:")
        print(f"  Symbol: {self.data.stock_symbol}")
        print(f"  Look back: {self.data.look_back}")
        print(f"  Train ratio: {self.data.train_ratio}")
        print(f"  Validation ratio: {self.data.validation_ratio}")
        print(f"  Feature set: {self.data.feature_set}")

        print(f"\nDevice: {self.device}")
        print("="*60 + "\n")


def get_default_config(symbol="AAPL", model_type='enhanced'):
    """
    Get default configuration

    Args:
        symbol (str): Stock symbol
        model_type (str): Model type ('enhanced' or 'simple')

    Returns:
        Config: Default configuration
    """
    config = Config()
    config.data.stock_symbol = symbol
    config.model.model_type = model_type
    return config


def load_config(path):
    """
    Load configuration from file

    Args:
        path (str or Path): Path to config file

    Returns:
        Config: Loaded configuration
    """
    return Config.load(path)
