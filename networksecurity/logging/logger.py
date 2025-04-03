import logging
import os
from datetime import datetime

LOG_FILENAME = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILEPATH = os.path.join(LOGS_DIR, LOG_FILENAME)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] | %(levelname)s | %(filename)s | Line: %(lineno)d | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILEPATH),
        logging.StreamHandler(),
    ],
)

# Create a logger instance
logger = logging.getLogger(__name__)
