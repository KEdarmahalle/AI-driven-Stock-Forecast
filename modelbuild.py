# Optional: Install libraries if you haven't already
# pip install pandas numpy scikit-learn tensorflow matplotlib

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model # Added load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import time # For model saving filenames

# --- Configuration ---
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
input_directory = "savedcsv" # Directory where CSVs are saved
model_save_directory = "modelsaved" # Directory to save trained models
lookback_years = 2 # Matches the data fetched

# Model Hyperparameters (can be tuned)
SEQUENCE_LENGTH = 60  # Use previous 60 hours to predict the next hour
LSTM_UNITS = 100      # Number of neurons in LSTM layer
GRU_UNITS = 100       # Number of neurons in GRU layer
DROPOUT_RATE = 0.2    # Dropout for regularization
EPOCHS = 50           # Max number of training epochs
BATCH_SIZE = 32       # Number of samples per gradient update
VALIDATION_SPLIT = 0.15 # Hold back 15% for validation during training
TEST_SPLIT = 0.15       # Hold back 15% for final testing (after validation split)

# --- Create output directory for models ---
if not os.path.exists(model_save_directory):
    os.makedirs(model_save_directory)
    print(f"Created directory: {model_save_directory}")

# --- 1. Load and Combine Data ---
print("\n--- Loading and Combining Data ---")
all_dfs = {}
min_len = float('inf') # To find the minimum length for alignment

for ticker in tickers:
    file_name = f"{ticker}_hourly_data_{lookback_years}y.csv"
    file_path = os.path.join(input_directory, file_name)
    if os.path.exists(file_path):
        # Read CSV with correct parsing
        df = pd.read_csv(file_path, skiprows=2)  # Skip the first two rows with headers
        # The first row after skipping headers contains the actual column names
        df.columns = ['TimestampUTC', 'Close', 'High', 'Low', 'Open', 'Volume']
        df['TimestampUTC'] = pd.to_datetime(df['TimestampUTC'])  # Convert to datetime
        df.set_index('TimestampUTC', inplace=True)  # Set as index
        # Select only 'Close' price for simplicity first
        all_dfs[ticker] = df[['Close']].rename(columns={'Close': f'{ticker}_Close'})
        min_len = min(min_len, len(df))
        print(f"Loaded {ticker} data: {len(df)} rows")
    else:
        print(f"!!! Error: File not found {file_path}")
        # Handle error appropriately - exit or skip ticker

# Combine into a single dataframe, aligning by index (timestamp)
# Use an inner join to ensure all timestamps exist for all stocks
if len(all_dfs) == len(tickers): # Proceed only if all files were loaded
    combined_df = pd.concat(all_dfs.values(), axis=1, join='inner')
    # Ensure data is sorted chronologically (should be already, but double-check)
    combined_df.sort_index(inplace=True)
    print(f"\nCombined DataFrame shape: {combined_df.shape}")
    print(combined_df.head())
    print(combined_df.tail()) # Check the date range matches expectations
    # If lengths were different, inner join might reduce rows
    if len(combined_df) < min_len:
        print(f"Warning: Combined data has {len(combined_df)} rows due to inner join (original min was {min_len}).")

    # Features (closing prices of all stocks)
    features = combined_df.columns.tolist()
    n_features = len(features) # Should be 5

    # --- 2. Scale Data ---
    print("\n--- Scaling Data ---")
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale the entire dataset
    scaled_data = scaler.fit_transform(combined_df)
    # We need to save the scaler to inverse transform predictions later
    # (Alternatively, fit only on training data below after splitting)

    # --- 3. Create Sequences ---
    print("\n--- Creating Sequences ---")
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i-SEQUENCE_LENGTH:i, :]) # Input: previous SEQUENCE_LENGTH hours for all 5 stocks
        y.append(scaled_data[i, :])                   # Output: the current hour's prices for all 5 stocks (which we are predicting)

    X, y = np.array(X), np.array(y)
    print(f"Shape of X: {X.shape}") # Should be (n_samples, SEQUENCE_LENGTH, n_features=5)
    print(f"Shape of y: {y.shape}") # Should be (n_samples, n_outputs=5)

    # --- 4. Split Data (Train, Validation, Test) ---
    # Important: Split chronologically for time series data!
    print("\n--- Splitting Data (Train/Validation/Test) ---")

    # Calculate split indices
    n_samples = X.shape[0]
    n_test = int(n_samples * TEST_SPLIT)
    n_val = int((n_samples - n_test) * VALIDATION_SPLIT) # Validation from the remaining data
    n_train = n_samples - n_test - n_val

    X_train, X_temp, y_train, y_temp = X[:n_train], X[n_train:], y[:n_train], y[n_train:]
    X_val, X_test, y_val, y_test = X_temp[:n_val], X_temp[n_val:], y_temp[:n_val], y_temp[n_val:]

    print(f"Train set size: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation set size: X={X_val.shape}, y={y_val.shape}")
    print(f"Test set size: X={X_test.shape}, y={y_test.shape}")

    # Optional: Re-fit scaler ONLY on training data (more robust approach)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # Reshape train data for scaler: from (samples, timesteps, features) to (samples*timesteps, features)
    # train_data_reshaped = X_train.reshape(-1, n_features)
    # scaler.fit(train_data_reshaped)
    # Now apply transform to X_train, X_val, X_test and y_train, y_val, y_test
    # This is more complex to implement correctly here, so sticking with fit on all data for simplicity.
    # Be aware this slightly 'leaks' future info (scale) into the training phase.

    # --- 5. Build Models (LSTM and GRU) ---
    print("\n--- Building Models ---")

    def build_lstm_model(input_shape, units, dropout_rate, n_outputs):
        model = Sequential(name="LSTM_Model")
        # Use return_sequences=True if stacking LSTM layers
        model.add(LSTM(units=units, return_sequences=False, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        # Add more LSTM/Dropout layers if needed
        # model.add(LSTM(units=units // 2, return_sequences=False))
        # model.add(Dropout(dropout_rate))
        model.add(Dense(units=n_outputs)) # Output layer: 5 neurons for 5 stocks
        model.compile(optimizer='adam', loss='mean_squared_error') # MSE for regression
        print("LSTM Model Summary:")
        model.summary()
        return model

    def build_gru_model(input_shape, units, dropout_rate, n_outputs):
        model = Sequential(name="GRU_Model")
        # Use return_sequences=True if stacking GRU layers
        model.add(GRU(units=units, return_sequences=False, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        # Add more GRU/Dropout layers if needed
        model.add(Dense(units=n_outputs)) # Output layer: 5 neurons for 5 stocks
        model.compile(optimizer='adam', loss='mean_squared_error') # MSE for regression
        print("GRU Model Summary:")
        model.summary()
        return model

    # Input shape for the models: (SEQUENCE_LENGTH, n_features)
    input_shape = (X_train.shape[1], X_train.shape[2])

    lstm_model = build_lstm_model(input_shape, LSTM_UNITS, DROPOUT_RATE, n_features)
    gru_model = build_gru_model(input_shape, GRU_UNITS, DROPOUT_RATE, n_features)

    # --- 6. Train Models ---
    print("\n--- Training Models ---")
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Callbacks for LSTM
    lstm_save_path = os.path.join(model_save_directory, f'best_lstm_model_{timestamp}.keras')
    lstm_early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lstm_model_checkpoint = ModelCheckpoint(lstm_save_path, monitor='val_loss', save_best_only=True)

    print("\nTraining LSTM Model...")
    history_lstm = lstm_model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[lstm_early_stopping, lstm_model_checkpoint],
        verbose=1 # Set to 2 for less output per epoch, or 0 for silent
    )
    print(f"Best LSTM model saved to {lstm_save_path}")

    # Callbacks for GRU
    gru_save_path = os.path.join(model_save_directory, f'best_gru_model_{timestamp}.keras')
    gru_early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    gru_model_checkpoint = ModelCheckpoint(gru_save_path, monitor='val_loss', save_best_only=True)

    print("\nTraining GRU Model...")
    history_gru = gru_model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[gru_early_stopping, gru_model_checkpoint],
        verbose=1
    )
    print(f"Best GRU model saved to {gru_save_path}")

    # Optional: Load the best models explicitly if needed (if restore_best_weights wasn't used/failed)
    # lstm_model = load_model(lstm_save_path)
    # gru_model = load_model(gru_save_path)

    # --- 7. Evaluate Models on Test Set ---
    print("\n--- Evaluating Models on Test Set ---")

    # Evaluate LSTM
    lstm_loss = lstm_model.evaluate(X_test, y_test, verbose=0)
    print(f"LSTM Model Test Loss (MSE): {lstm_loss:.6f}")

    # Evaluate GRU
    gru_loss = gru_model.evaluate(X_test, y_test, verbose=0)
    print(f"GRU Model Test Loss (MSE): {gru_loss:.6f}")

    # Make predictions on the test set
    y_pred_lstm_scaled = lstm_model.predict(X_test)
    y_pred_gru_scaled = gru_model.predict(X_test)

    # Inverse transform predictions and actual values to original scale
    y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled)
    y_pred_gru = scaler.inverse_transform(y_pred_gru_scaled)
    y_test_actual = scaler.inverse_transform(y_test) # Inverse transform the actual test data

    # Calculate RMSE and MAE for each stock (more interpretable than overall MSE)
    print("\n--- Per-Stock Evaluation (Test Set) ---")
    results = {}
    for i, ticker in enumerate(tickers):
        lstm_mse = np.mean((y_pred_lstm[:, i] - y_test_actual[:, i])**2)
        lstm_rmse = np.sqrt(lstm_mse)
        lstm_mae = np.mean(np.abs(y_pred_lstm[:, i] - y_test_actual[:, i]))

        gru_mse = np.mean((y_pred_gru[:, i] - y_test_actual[:, i])**2)
        gru_rmse = np.sqrt(gru_mse)
        gru_mae = np.mean(np.abs(y_pred_gru[:, i] - y_test_actual[:, i]))

        results[ticker] = {
            'LSTM_RMSE': lstm_rmse, 'LSTM_MAE': lstm_mae,
            'GRU_RMSE': gru_rmse, 'GRU_MAE': gru_mae
        }
        print(f"\n--- {ticker} ---")
        print(f"  LSTM: RMSE={lstm_rmse:.4f}, MAE={lstm_mae:.4f}")
        print(f"  GRU : RMSE={gru_rmse:.4f}, MAE={gru_mae:.4f}")

    results_df = pd.DataFrame(results).T # Transpose for better readability
    print("\n--- Summary Evaluation ---")
    print(results_df)

    # --- 8. Display Predictions (Example for one stock) ---
    print("\n--- Plotting Example Predictions (Test Set) ---")

    def plot_predictions(ticker_index, ticker_name):
        plt.figure(figsize=(14, 7))
        plt.plot(y_test_actual[:, ticker_index], label='Actual Price', color='blue', alpha=0.7)
        plt.plot(y_pred_lstm[:, ticker_index], label='LSTM Prediction', color='red', linestyle='--', alpha=0.7)
        plt.plot(y_pred_gru[:, ticker_index], label='GRU Prediction', color='green', linestyle=':', alpha=0.7)
        plt.title(f'{ticker_name} - Actual vs Predicted Hourly Close Prices (Test Set)')
        plt.xlabel('Time Steps (Hours in Test Set)')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Plot for the first stock (e.g., AAPL)
    plot_predictions(0, tickers[0])
    # You can loop or manually call plot_predictions for other indices (1 to 4) if desired

    # --- 9. Predict Next Hour (Example) ---
    print("\n--- Predicting Next Hour ---")
    # Get the last sequence from the original *scaled* data
    last_sequence_scaled = scaled_data[-SEQUENCE_LENGTH:]
    # Reshape it for model input: (1, SEQUENCE_LENGTH, n_features)
    last_sequence_reshaped = last_sequence_scaled.reshape((1, SEQUENCE_LENGTH, n_features))

    # Predict using both models
    next_hour_pred_lstm_scaled = lstm_model.predict(last_sequence_reshaped)
    next_hour_pred_gru_scaled = gru_model.predict(last_sequence_reshaped)

    # Inverse transform the predictions
    next_hour_pred_lstm = scaler.inverse_transform(next_hour_pred_lstm_scaled)
    next_hour_pred_gru = scaler.inverse_transform(next_hour_pred_gru_scaled)

    print("Predicted Closing Prices for the Next Hour:")
    predictions_summary = {}
    for i, ticker in enumerate(tickers):
        predictions_summary[ticker] = {
            'LSTM Prediction': next_hour_pred_lstm[0, i],
            'GRU Prediction': next_hour_pred_gru[0, i]
        }
    print(pd.DataFrame(predictions_summary).T)


else:
    print("\n--- Could not proceed: Not all ticker CSV files were loaded. ---")

print("\n--- Script Finished ---")
print(f"Best models saved with timestamp {timestamp} in '{model_save_directory}'")