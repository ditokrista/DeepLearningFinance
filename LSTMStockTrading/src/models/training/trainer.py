"""
Model Training Module

Contains training logic, early stopping, and training utilities.
"""

import torch
import torch.nn as nn
from pathlib import Path


class EarlyStopping:
    """
    Early stopping to prevent overfitting

    Monitors validation loss and stops training when it stops improving.
    Optionally restores the best model weights.

    Args:
        patience (int): Number of epochs to wait before stopping
        min_delta (float): Minimum change in loss to qualify as improvement
        restore_best_weights (bool): Whether to restore best model weights
    """

    def __init__(self, patience=20, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        """
        Check if training should stop

        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Current model

        Returns:
            bool: Whether to stop training
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0

        return self.early_stop


class ModelTrainer:
    """
    Model Trainer with best practices

    Handles training loop, validation, early stopping, learning rate scheduling,
    gradient clipping, and model checkpointing.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to train on
        learning_rate (float): Initial learning rate
        num_epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        gradient_clip (float): Max norm for gradient clipping
        save_path (Path or str, optional): Path to save best model
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=0.001,
        num_epochs=300,
        patience=30,
        gradient_clip=1.0,
        save_path=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_clip = gradient_clip
        self.save_path = save_path

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self):
        """
        Train for one epoch

        Returns:
            float: Average training loss
        """
        self.model.train()
        train_loss = 0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            predictions = self.model(X_batch)
            loss = self.criterion(predictions.squeeze(), y_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )

            self.optimizer.step()
            train_loss += loss.item()

        return train_loss / len(self.train_loader)

    def validate(self):
        """
        Validate the model

        Returns:
            float: Average validation loss
        """
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = self.criterion(predictions.squeeze(), y_batch)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def train(self, verbose=True):
        """
        Full training loop

        Args:
            verbose (bool): Whether to print progress

        Returns:
            dict: Training history
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training on device: {self.device}")
            print(f"{'='*60}\n")

        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch [{epoch+1}/{self.num_epochs}] "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"LR: {current_lr:.6f}"
                )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.save_path:
                    torch.save(self.model.state_dict(), self.save_path)

            # Early stopping
            if self.early_stopping(val_loss, self.model):
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        if verbose:
            print(f"\nTraining completed. Best validation loss: {self.best_val_loss:.6f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    learning_rate=0.001,
    num_epochs=300,
    patience=30,
    gradient_clip=1.0,
    save_path=None,
    verbose=True
):
    """
    Convenience function to train a model

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to train on
        learning_rate (float): Initial learning rate
        num_epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        gradient_clip (float): Max norm for gradient clipping
        save_path (Path or str, optional): Path to save best model
        verbose (bool): Whether to print progress

    Returns:
        tuple: (train_losses, val_losses)
    """
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        patience=patience,
        gradient_clip=gradient_clip,
        save_path=save_path
    )

    history = trainer.train(verbose=verbose)

    return history['train_losses'], history['val_losses']


def set_seed(seed=42):
    """
    Set random seeds for reproducibility

    Args:
        seed (int): Random seed
    """
    import numpy as np
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
