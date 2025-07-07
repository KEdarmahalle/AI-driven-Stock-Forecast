import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import sys
import os
import time
import logging
from pathlib import Path
from stock_data_job import StockDataJob

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

service_log = os.path.join(log_dir, "service.log")
logging.basicConfig(
    filename=service_log,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockDataService')

class StockDataService(win32serviceutil.ServiceFramework):
    _svc_name_ = "StockDataService"
    _svc_display_name_ = "Stock Data Collection Service"
    _svc_description_ = "Continuously collects stock data from yfinance and stores it in SQL Server"
    _svc_deps_ = ["MSSQLSERVER"]  # Depend on SQL Server service

    def __init__(self, args):
        try:
            win32serviceutil.ServiceFramework.__init__(self, args)
            self.stop_event = win32event.CreateEvent(None, 0, 0, None)
            self.job = None  # Initialize job as None
            logger.info("Service initialized")
        except Exception as e:
            logger.error(f"Service initialization failed: {str(e)}")
            raise

    def SvcStop(self):
        """Stop the service"""
        try:
            logger.info("Received stop signal")
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.stop_event)
            if self.job:
                self.job.stop()
            logger.info("Service stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping service: {str(e)}")
            raise

    def SvcDoRun(self):
        """Run the service"""
        try:
            logger.info("Starting service...")
            self.ReportServiceStatus(win32service.SERVICE_START_PENDING)
            
            # Initialize the job
            try:
                self.job = StockDataJob()
                logger.info("Job initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize job: {str(e)}")
                raise

            # Start the job
            try:
                self.ReportServiceStatus(win32service.SERVICE_RUNNING)
                logger.info("Service status set to RUNNING")
                self.job.run()
            except Exception as e:
                logger.error(f"Error in job execution: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Service failed: {str(e)}")
            self.ReportServiceStatus(win32service.SERVICE_STOPPED)
            raise

if __name__ == '__main__':
    try:
        if len(sys.argv) == 1:
            logger.info("Starting service control dispatcher")
            servicemanager.Initialize()
            servicemanager.PrepareToHostSingle(StockDataService)
            servicemanager.StartServiceCtrlDispatcher()
        else:
            logger.info(f"Handling command line: {sys.argv[1]}")
            win32serviceutil.HandleCommandLine(StockDataService)
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise 