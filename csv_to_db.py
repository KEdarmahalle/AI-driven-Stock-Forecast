import pandas as pd
import os
import pyodbc
from dotenv import load_dotenv
import glob

# Load environment variables
load_dotenv()

# Constants
CSV_DIR = "savedcsv"  # Directory with CSV files
# Company ID mapping
COMPANY_IDS = {
    'AAPL': 8,
    'MSFT': 12,
    'GOOGL': 10,
    'AMZN': 9,
    'META': 11,
    'NVDA': 13,  # Added some extra mappings in case these files exist
    'TSLA': 14,
    'JPM': 15
}

def get_db_connection():
    """Create and return a database connection"""
    conn_str = (
        f"DRIVER={{SQL Server}};"
        f"SERVER={os.getenv('DB_SERVER')};"
        f"DATABASE={os.getenv('DB_NAME')};"
        f"Trusted_Connection={os.getenv('DB_TRUSTED_CONNECTION', 'yes')};"
    )
    return pyodbc.connect(conn_str)

def insert_stock_data(data_df, company_id, connection):
    """Insert stock data into the database"""
    try:
        cursor = connection.cursor()
        
        # Prepare the insert query
        insert_query = """
        INSERT INTO [STOCK].[dbo].[CompanyStockdata]
        (date, [open], high, low, [close], volume, CompanyId)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        # Convert DataFrame rows to list of tuples for batch insert
        rows = []
        for idx, row in data_df.iterrows():
            # Handle both formats (our format and yfinance format)
            if 'Open' in row:  # yfinance format
                rows.append((
                    idx.to_pydatetime(),  # datetime
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume']),
                    company_id
                ))
            else:  # Our format (possibly different)
                rows.append((
                    idx.to_pydatetime(),  # datetime
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume']),
                    company_id
                ))
        
        # Execute batch insert
        cursor.executemany(insert_query, rows)
        connection.commit()
        
        print(f"Inserted {len(rows)} rows for company ID {company_id}")
        return True
            
    except Exception as e:
        print(f"Error inserting data for company ID {company_id}: {str(e)}")
        return False

def main():
    # Check if CSV directory exists
    if not os.path.exists(CSV_DIR):
        print(f"Error: CSV directory '{CSV_DIR}' not found")
        return
    
    # Get list of CSV files
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in '{CSV_DIR}'")
        return
    
    print(f"Found {len(csv_files)} CSV files: {csv_files}")
    
    # Connect to database
    try:
        connection = get_db_connection()
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        return
    
    # Process each CSV file
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"Processing {filename}...")
        
        # Extract ticker from filename (assuming format: TICKER_something.csv)
        ticker = filename.split('_')[0].upper()
        
        if ticker not in COMPANY_IDS:
            print(f"Warning: Unknown ticker '{ticker}', skipping file")
            continue
        
        company_id = COMPANY_IDS[ticker]
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file, skiprows=2)  # Skip the first two rows with headers
            # The first row after skipping headers contains the actual column names
            df.columns = ['TimestampUTC', 'Close', 'High', 'Low', 'Open', 'Volume']
            df['TimestampUTC'] = pd.to_datetime(df['TimestampUTC'])  # Convert to datetime
            df.set_index('TimestampUTC', inplace=True)  # Set as index
            
            # Insert data into database
            insert_stock_data(df, company_id, connection)
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
    
    print("CSV import process completed")

if __name__ == "__main__":
    main() 