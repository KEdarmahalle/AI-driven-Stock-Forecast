import sys
import logging
from datetime import datetime
import time
from pathlib import Path
from stock_data_job import StockDataJob

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=log_dir / "stock_runner.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockRunner')

def main():
    logger.info("Starting Stock Data Runner")
    try:
        # Initialize the job
        job = StockDataJob()
        logger.info("Job initialized successfully")
        
        # Run the job
        job.run()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping gracefully...")
    except Exception as e:
        logger.error(f"Error in runner: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Stock Data Runner stopped")

if __name__ == "__main__":
    main() 