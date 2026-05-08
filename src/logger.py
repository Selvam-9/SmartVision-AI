import logging
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name="SmartVision", log_file="smartvision.log", level=logging.INFO):
    """Function to setup a logger with both file and console handlers."""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # If logger already has handlers, don't add more (prevents duplicate logs)
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')

    # File Handler
    file_path = LOG_DIR / log_file
    fh = logging.FileHandler(file_path, encoding='utf-8')
    fh.setFormatter(formatter)
    fh.setLevel(level)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)

    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# Create a default logger instance
logger = setup_logger()
