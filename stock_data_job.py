import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
import time
import pyodbc
import logging
from logging.handlers import RotatingFileHandler
import os
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class StockDataJob:
    def __init__(self):
        # Configure logging
        self.setup_logging()
        
        # Database configuration from environment variables
        self.DB_CONFIG = {
            'server': os.getenv('DB_SERVER'),
            'database': os.getenv('DB_NAME'),
            'trusted_connection': os.getenv('DB_TRUSTED_CONNECTION', 'yes')  # Windows authentication
        }
        
        # Stock configuration
        self.TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        self.COMPANY_IDS = {
            'AAPL': 8,
            'MSFT': 12,
            'GOOGL': 10,
            'AMZN': 9,
            'META': 11
        }
        
        # Control flag for the service
        self._running = False
        self._stop_event = threading.Event()

    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, "stock_data_job.log")
        self.logger = logging.getLogger("StockDataJob")
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_db_connection(self):
        """Create and return a database connection"""
        conn_str = (
            f"DRIVER={{SQL Server}};"
            f"SERVER={self.DB_CONFIG['server']};"
            f"DATABASE={self.DB_CONFIG['database']};"
            f"Trusted_Connection={self.DB_CONFIG['trusted_connection']};"
        )
        return pyodbc.connect(conn_str)

    def get_latest_datetime(self):
        """Get the latest datetime from the database for all tickers"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                query = """
                SELECT MAX(date) 
                FROM [STOCK].[dbo].[CompanyStockdata]
                """
                cursor.execute(query)
                result = cursor.fetchone()
                return result[0] if result[0] else None
        except Exception as e:
            self.logger.error(f"Error getting latest datetime: {str(e)}")
            return None

    def insert_stock_data(self, data_df, company_id):
        """Insert stock data into the database"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare the insert query
                insert_query = """
                INSERT INTO [STOCK].[dbo].[CompanyStockdata]
                (date, [open], high, low, [close], volume, CompanyId)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                
                # Convert DataFrame rows to list of tuples for batch insert
                rows = []
                for idx, row in data_df.iterrows():
                    rows.append((
                        idx.to_pydatetime(),  # datetime
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume']),
                        company_id
                    ))
                
                # Execute batch insert
                cursor.executemany(insert_query, rows)
                conn.commit()
                
                self.logger.info(f"Inserted {len(rows)} rows for company ID {company_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error inserting data for company ID {company_id}: {str(e)}")
            return False

    def fetch_stock_data(self, ticker, start_date, end_date):
        """Fetch stock data from yfinance"""
        try:
            # Download data
            data = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval="1h",
                progress=False
            )
            
            if not data.empty:
                # Convert index to timezone-naive UTC
                if data.index.tz is not None:
                    data.index = data.index.tz_convert('UTC').tz_localize(None)
                
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def stop(self):
        """Stop the job"""
        self._running = False
        self._stop_event.set()

    def run(self):
        """Main job function"""
        self._running = True
        self.logger.info("Starting stock data update job")
        
        while self._running and not self._stop_event.is_set():
            try:
                # Get the latest datetime from the database
                latest_dt = self.get_latest_datetime()
                
                if latest_dt is None:
                    # If no data exists, start from 2 years ago
                    start_date = datetime.now() - timedelta(days=730)
                else:
                    # Start from the last record's datetime
                    start_date = latest_dt
                
                # End date is current time
                end_date = datetime.now()
                
                self.logger.info(f"Fetching data from {start_date} to {end_date}")
                
                # Process each ticker
                for ticker in self.TICKERS:
                    if self._stop_event.is_set():
                        break
                        
                    company_id = self.COMPANY_IDS[ticker]
                    
                    # Fetch data
                    data = self.fetch_stock_data(ticker, start_date, end_date)
                    
                    if data is not None and not data.empty:
                        # Insert into database
                        success = self.insert_stock_data(data, company_id)
                        if success:
                            self.logger.info(f"Successfully updated data for {ticker}")
                        else:
                            self.logger.error(f"Failed to update data for {ticker}")
                    else:
                        self.logger.warning(f"No new data available for {ticker}")
                
                if self._stop_event.is_set():
                    break
                
                # Wait for next update
                # Since US market closes, we can check less frequently during off-hours
                current_hour = datetime.now().hour
                
                # Determine sleep time based on market hours (EST/EDT)
                if 4 <= current_hour < 20:  # 4 AM to 8 PM EST/EDT
                    sleep_time = 3600  # 1 hour during market hours
                else:
                    sleep_time = 7200  # 2 hours during off-hours
                
                self.logger.info(f"Sleeping for {sleep_time/3600} hours")
                
                # Sleep in small intervals to allow for clean shutdown
                for _ in range(int(sleep_time / 10)):
                    if self._stop_event.is_set():
                        break
                    time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                if not self._stop_event.is_set():
                    time.sleep(300)  # Sleep for 5 minutes on error

        self.logger.info("Stock data update job stopped")

if __name__ == "__main__":
    job = StockDataJob()
    job.run() 