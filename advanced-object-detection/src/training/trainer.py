"""
Training utilities for object detection models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, List, Optional, Callable
import logging
from tqdm import tqdm
import wandb
import os

logger = logging.getLogger(__name__)

class ObjectDetectionTrainer:
    """Trainer for object detection models"""
    
    def __init__(self, model: nn.Module, config: Dict):
        """Initialize trainer"""
        self.model = model
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        
        # Setup wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(project=config.get('project_name', 'object-detection'))
            wandb.config.update(config)
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adam')
        lr = opt_config.get('lr', 1e-3)
        weight_decay = opt_config.get('weight_decay', 0.0)
        
        if opt_type.lower() == 'adam':
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type.lower() == 'sgd':
            momentum = opt_config.get('momentum', 0.9)
            return SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        sched_config = self.config.get('scheduler', {})
        sched_type = sched_config.get('type', 'reduce_on_plateau')
        
        if sched_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 5)
            )
        elif sched_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get('T_max', 10)
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses.item()
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / batch_count,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to wandb
            if self.config.get('use_wandb', False) and batch_idx % 10 == 0:
                wandb.log({
                    'train_loss': losses.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        return {'loss': total_loss / batch_count}
    
    def validate(self, val_loader: DataLoader, 
                evaluator: Optional[Callable] = None) -> Dict:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        batch_count = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                # Move to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Get predictions
                outputs = self.model(images)
                
                # Store for evaluation
                all_predictions.extend(outputs)
                all_targets.extend(targets)
                
                batch_count += 1
        
        # Calculate metrics
        metrics = {}
        if evaluator:
            metrics = evaluator(all_predictions, all_targets)
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             num_epochs: int, evaluator: Optional[Callable] = None):
        """Full training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader, evaluator)
                logger.info(f"Epoch {epoch} - Validation Metrics: {val_metrics}")
                
                # Update scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                    else:
                        self.scheduler.step()
                
                # Save best model
                if val_metrics.get('mAP', 0) > self.best_metric:
                    self.best_metric = val_metrics['mAP']
                    self.save_checkpoint('best_model.pth')
            
            # Save checkpoint
            if epoch % self.config.get('save_interval', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filename}")
