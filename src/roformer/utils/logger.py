"""
Logging utilities for RoFormer LLM training
"""

import logging
import os
from datetime import datetime
from pathlib import Path
import sys


class Logger:
    """Custom logger for training"""
    
    def __init__(self, name: str, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_dir: Directory to save logs
            log_level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.log_file = log_file
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_metrics(self, metrics: dict, step: int):
        """Log training metrics"""
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        self.info(f"Step {step} - {metrics_str}")


def get_logger(name: str, log_dir: str = "logs", log_level: str = "INFO") -> Logger:
    """Get a logger instance"""
    return Logger(name, log_dir, log_level)
