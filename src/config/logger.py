"""
Logging Infrastructure for Wakeword Training Platform
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorama
from colorama import Fore, Style

# Initialize colorama for Windows support
colorama.init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output"""

    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"

        return super().format(record)


class WakewordLogger:
    """Centralized logger for the platform"""

    def __init__(self, name: str, log_dir: Optional[Path] = None):
        """
        Initialize logger

        Args:
            name: Logger name
            log_dir: Directory for log files (default: logs/)
        """
        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers

        self._setup_handlers()

    def _setup_handlers(self):
        """Setup console and file handlers"""

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = ColoredFormatter(
            '%(levelname)s | %(name)s | %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler (detailed, no colors)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Log file: {log_file}")

    def get_logger(self) -> logging.Logger:
        """Get the underlying logger"""
        return self.logger


class TrainingLogger(WakewordLogger):
    """Specialized logger for training"""

    def __init__(self):
        super().__init__("training", Path("logs/training"))

    def log_epoch(self, epoch: int, total_epochs: int, metrics: dict):
        """Log epoch results"""
        msg = f"Epoch [{epoch}/{total_epochs}]"
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" | {key}: {value:.4f}"
            else:
                msg += f" | {key}: {value}"
        self.logger.info(msg)

    def log_batch(self, epoch: int, batch: int, total_batches: int,
                   loss: float, speed: float):
        """Log batch progress"""
        self.logger.debug(
            f"Epoch [{epoch}] Batch [{batch}/{total_batches}] "
            f"| Loss: {loss:.4f} | Speed: {speed:.1f} samples/sec"
        )


class EvaluationLogger(WakewordLogger):
    """Specialized logger for evaluation"""

    def __init__(self):
        super().__init__("evaluation", Path("logs/evaluation"))

    def log_metrics(self, metrics: dict):
        """Log evaluation metrics"""
        self.logger.info("Evaluation Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")


class DataLogger(WakewordLogger):
    """Specialized logger for data processing"""

    def __init__(self):
        super().__init__("data", Path("logs"))

    def log_dataset_stats(self, stats: dict):
        """Log dataset statistics"""
        self.logger.info("Dataset Statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")


def get_logger(name: str = "wakeword") -> logging.Logger:
    """
    Get a logger instance

    Args:
        name: Logger name

    Returns:
        logging.Logger instance
    """
    return WakewordLogger(name).get_logger()


def get_training_logger() -> TrainingLogger:
    """Get training logger"""
    return TrainingLogger()


def get_evaluation_logger() -> EvaluationLogger:
    """Get evaluation logger"""
    return EvaluationLogger()


def get_data_logger() -> DataLogger:
    """Get data logger"""
    return DataLogger()


if __name__ == "__main__":
    # Test logging
    logger = get_logger("test")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    # Test training logger
    train_logger = get_training_logger()
    train_logger.log_epoch(1, 50, {
        "train_loss": 0.1234,
        "val_loss": 0.1456,
        "train_acc": 0.9567,
        "val_acc": 0.9432
    })