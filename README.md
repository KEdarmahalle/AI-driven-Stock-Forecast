# AI-Powered Stock Market Analysis System

A robust, production-ready platform for stock market analysis and prediction, combining machine learning, real-time data, and AI-powered insights. This project is designed for traders, investors, and data scientists seeking advanced analytics and automation.

---

## 🚀 Features

- **Real-time Stock Data**: Automated collection from Polygon.io
- **Dual ML Models**: LSTM & GRU for multi-horizon predictions
- **Technical Analysis**: SMA, RSI, Bollinger Bands, volume, and more
- **AI Market Insights**: Sentiment, news impact, risk, and trends via Gemini AI
- **Interactive Dashboard**: Streamlit + Plotly for real-time, customizable charts
- **Automated Scheduling**: Windows Task Scheduler integration
- **SQL Server Support**: Optional persistent storage
- **API Rate Limiting**: Built-in safeguards

---

## 🗂️ Project Structure

```
├── home.py                 # Streamlit dashboard
├── modelbuild.py           # ML model training & evaluation
├── stock_data_job.py       # Data collection service
├── stock_service.py        # Data processing
├── market_insights.py      # AI-powered analysis
├── csv_to_db.py            # Database integration
├── stock_runner.py         # Scheduled task runner
├── setup_scheduled_task.bat# Windows task setup
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── logs/                   # Application logs
├── modelsaved/             # Trained models
└── savedcsv/               # Historical data
```

---

## 🛠️ Prerequisites

- Python 3.8+
- Windows OS (for scheduling)
- SQL Server (optional)
- API keys for:
  - Polygon.io (stock data)
  - Google Gemini AI (market insights)

---

## ⚡ Quickstart

1. **Clone the repository:**
   ```bash
   git clone <github-repo-url>
   cd <your-repo-directory>
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment:**
   - Copy `.env.example` to `.env` and fill in your API keys and DB info.
4. **(Optional) Set up scheduled tasks:**
   ```bash
   setup_scheduled_task.bat
   ```
5. **Run the dashboard:**
   ```bash
   streamlit run home.py
   ```
   - Visit [http://localhost:8501](http://localhost:8501)

---

## 🧑‍💻 Usage

- **Dashboard:**
  - Select sector, stock, timeframe, and prediction horizon
  - Click "Load Data" for analysis
  - Use "Get Market Analysis" for AI insights
- **Model Training:**
  ```bash
  python modelbuild.py
  ```
  - Trains LSTM & GRU models, saves to `modelsaved/`
- **Data Collection:**
  - Automated, hourly, with rate limiting and caching

---

## 📝 GitHub & Security

- **.env, logs, models, and sensitive files are gitignored**
- **Never commit your real `.env` file**
- Share `.env.example` for config structure
- MIT License (see LICENSE)

---

## 🤝 Contributing

- PRs welcome! Add a `CONTRIBUTING.md` for guidelines if you want to accept contributions.

---

## 🙏 Acknowledgments

- Polygon.io, Google Gemini AI, Streamlit, TensorFlow, Plotly, TA-Lib

---

For questions or support, open an issue on GitHub.