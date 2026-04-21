"""
Configuration Management Module

Handles configuration for models, training, and data processing.
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch
import yaml

from src.utils.torch_runtime import TorchRuntime, resolve_torch_runtime


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
class ComputeConfig:
    device: str = 'auto'
    mixed_precision: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    deterministic: bool = True
    benchmark: bool = False
    require_gpu: bool = False


@dataclass
class Config:
    """Master configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    device: torch.device = field(init=False)
    runtime: TorchRuntime = field(init=False, repr=False)

    def __post_init__(self):
        self._apply_base_config_defaults()
        self.refresh_runtime()

    def _apply_base_config_defaults(self):
        base_config_path = self.paths.project_root / "config" / "base_config.yaml"
        if not base_config_path.exists():
            return

        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f) or {}

        self._apply_system_settings(base_config)

    def _apply_system_settings(self, config_dict):
        system_config = config_dict.get('system', {})
        compute_config = system_config.get('compute', {})
        reproducibility_config = system_config.get('reproducibility', {})

        for key in ('device', 'mixed_precision', 'num_workers', 'pin_memory'):
            if key in compute_config and hasattr(self.compute, key):
                setattr(self.compute, key, compute_config[key])

        if 'deterministic' in reproducibility_config:
            self.compute.deterministic = reproducibility_config['deterministic']
        if 'benchmark' in reproducibility_config:
            self.compute.benchmark = reproducibility_config['benchmark']
        if 'seed' in reproducibility_config:
            self.training.seed = reproducibility_config['seed']

        data_config = config_dict.get('data') or {}
        features_config = data_config.get('features', {})
        if 'scaling_method' in features_config:
            self.data.scaler_type = features_config['scaling_method']

    def refresh_runtime(self, device_override=None):
        requested_device = device_override or self.compute.device
        self.runtime = resolve_torch_runtime(
            requested_device,
            require_accelerator=self.compute.require_gpu
        )
        self.device = self.runtime.device

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
            },
            'compute': {
                'device': self.compute.device,
                'mixed_precision': self.compute.mixed_precision,
                'num_workers': self.compute.num_workers,
                'pin_memory': self.compute.pin_memory,
                'deterministic': self.compute.deterministic,
                'benchmark': self.compute.benchmark,
                'require_gpu': self.compute.require_gpu,
                'resolved_device': str(self.device),
                'backend': self.runtime.backend,
            }
        }

    def save(self, path):
        """Save configuration to YAML file"""
        path = Path(path)
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls()
        config_dict = config_dict or {}
        config._apply_system_settings(config_dict)

        model_config = config_dict.get('model') or {}
        architecture_config = model_config.get('architecture', {})
        sequence_config = model_config.get('sequence', {})
        model_name = str(model_config.get('name', '')).lower()

        if 'simple' in model_name:
            config.model.model_type = 'simple'
        elif 'improved' in model_name or 'enhanced' in model_name:
            config.model.model_type = 'enhanced'

        for key, value in model_config.items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)

        for key, value in architecture_config.items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)

        if 'lookback_window' in sequence_config:
            config.data.look_back = sequence_config['lookback_window']

        training_config = config_dict.get('training') or {}
        optimizer_config = training_config.get('optimizer', {})
        early_stopping_config = training_config.get('early_stopping', {})

        for key, value in training_config.items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)

        if 'epochs' in training_config:
            config.training.num_epochs = training_config['epochs']
        if 'gradient_clip_val' in training_config:
            config.training.gradient_clip = training_config['gradient_clip_val']
        if 'train_split' in training_config:
            config.data.train_ratio = training_config['train_split']
        if 'val_split' in training_config:
            config.data.validation_ratio = training_config['val_split']
        if 'test_split' in training_config:
            config.data.test_ratio = training_config['test_split']
        if 'lr' in optimizer_config:
            config.training.learning_rate = optimizer_config['lr']
        if 'patience' in early_stopping_config:
            config.training.patience = early_stopping_config['patience']

        data_config = config_dict['data'] or {}
        features_config = data_config.get('features', {})

        for key, value in data_config.items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)

        if 'scaling_method' in features_config:
            config.data.scaler_type = features_config['scaling_method']

        if 'compute' in config_dict:
            for key, value in config_dict['compute'].items():
                if hasattr(config.compute, key):
                    setattr(config.compute, key, value)

        config.refresh_runtime()

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

        print("\nCompute Configuration:")
        print(f"  Requested device: {self.compute.device}")
        print(f"  Resolved device: {self.device}")
        print(f"  Backend: {self.runtime.backend}")
        print(f"  Device name: {self.runtime.device_name}")
        print(f"  Mixed precision: {self.compute.mixed_precision and self.device.type != 'cpu'}")
        print(f"  Num workers: {self.compute.num_workers}")
        print(f"  Pin memory: {self.compute.pin_memory and self.device.type != 'cpu'}")
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
