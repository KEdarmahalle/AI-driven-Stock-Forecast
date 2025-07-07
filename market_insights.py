import streamlit as st
import os
from google import genai
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

class MarketInsights:
    def __init__(self):
        # Initialize Gemini API client (API key will be added to .env later)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.initialized = True
        else:
            self.initialized = False
            
    def get_market_insights(self, symbol, context=None):
        """
        Get market insights for a given stock symbol using Gemini API with Google Search grounding
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            context (str, optional): Additional context about the stock or specific questions
            
        Returns:
            dict: Market insights with various analysis dimensions
        """
        if not self.initialized:
            return {
                "error": "Gemini API key not found. Please add GEMINI_API_KEY to your .env file."
            }
        
        try:
            # Create a detailed prompt for comprehensive analysis
            base_prompt = f"""
            Provide a comprehensive market analysis for {symbol} stock. Include:
            1. Recent performance and key price movements
            2. Current market sentiment (bullish/bearish/neutral)
            3. Latest news that might impact the stock
            4. Key analyst opinions and price targets
            5. Upcoming events that could affect the stock (earnings, product launches, etc.)
            
            Format the response as JSON with these keys:
            - sentiment: overall market sentiment (bullish/bearish/neutral)
            - summary: brief 2-3 sentence summary of current situation
            - recent_news: array of 3-5 recent news items with date and source
            - price_targets: average analyst price target and range
            - risk_factors: key risk factors to watch
            - opportunities: potential positive catalysts
            """
            
            if context:
                base_prompt += f"\n\nAdditional context: {context}"
            
            # Configure Google Search as a tool for Gemini
            google_search_tool = genai.types.Tool(
                google_search=genai.types.GoogleSearch()
            )
            
            # Generate content with Google Search grounding
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=base_prompt,
                config=genai.types.GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                )
            )
            
            # Extract the content and parse as JSON
            try:
                import json
                result = json.loads(response.text)
                result['sources'] = self._extract_sources(response)
                return result
            except json.JSONDecodeError:
                # If response is not valid JSON, return raw text
                return {
                    "raw_analysis": response.text,
                    "sources": self._extract_sources(response)
                }
                
        except Exception as e:
            return {
                "error": f"Error generating market insights: {str(e)}"
            }
    
    def _extract_sources(self, response):
        """Extract sources from the grounding metadata if available"""
        sources = []
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    if hasattr(candidate.grounding_metadata, 'ground_chunks'):
                        for chunk in candidate.grounding_metadata.ground_chunks:
                            if hasattr(chunk, 'web') and chunk.web and hasattr(chunk.web, 'uri'):
                                sources.append({
                                    "uri": chunk.web.uri,
                                    "title": getattr(chunk.web, 'title', 'Unknown Source')
                                })
        except Exception:
            pass
        return sources
            
    def get_competitive_analysis(self, symbol, competitors=None):
        """
        Get competitive analysis comparing the target stock with its main competitors
        
        Args:
            symbol (str): Target stock symbol
            competitors (list, optional): List of competitor symbols
            
        Returns:
            dict: Competitive analysis
        """
        if not self.initialized:
            return {
                "error": "Gemini API key not found. Please add GEMINI_API_KEY to your .env file."
            }
        
        if not competitors:
            # Default competitors based on sector
            sector_competitors = {
                'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'NVDA'],
                'MSFT': ['AAPL', 'GOOGL', 'AMZN', 'NVDA'],
                'GOOGL': ['MSFT', 'AAPL', 'META', 'AMZN'],
                'AMZN': ['MSFT', 'GOOGL', 'AAPL', 'WMT'],
                'META': ['GOOGL', 'SNAP', 'PINS', 'TWTR'],
                'NVDA': ['AMD', 'INTC', 'TSLA', 'MU'],
                'TSLA': ['RIVN', 'LCID', 'GM', 'F']
            }
            competitors = sector_competitors.get(symbol, ['AAPL', 'MSFT', 'GOOGL'])
        
        competitor_str = ", ".join(competitors)
        
        try:
            prompt = f"""
            Provide a comprehensive competitive analysis comparing {symbol} with its competitors ({competitor_str}).
            
            Include:
            1. Relative market position
            2. Comparative valuation metrics (P/E, P/S, etc.)
            3. Performance comparison YTD
            4. Strengths and weaknesses vs competitors
            5. Who is gaining/losing market share
            
            Format response as JSON with these keys:
            - comparative_summary: brief summary of how {symbol} positions against competitors
            - valuation_comparison: comparison of key valuation metrics
            - performance_ranking: YTD performance ranking
            - competitive_advantages: list of {symbol}'s advantages
            - competitive_disadvantages: list of {symbol}'s disadvantages
            - market_share_trends: recent market share movement
            """
            
            # Configure Google Search as a tool
            google_search_tool = genai.types.Tool(
                google_search=genai.types.GoogleSearch()
            )
            
            # Generate competitive analysis
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                )
            )
            
            # Extract and parse as JSON
            try:
                import json
                result = json.loads(response.text)
                result['sources'] = self._extract_sources(response)
                return result
            except json.JSONDecodeError:
                return {
                    "raw_analysis": response.text,
                    "sources": self._extract_sources(response)
                }
                
        except Exception as e:
            return {
                "error": f"Error generating competitive analysis: {str(e)}"
            }
    
    def get_market_sentiment_score(self, symbol):
        """
        Get a numerical market sentiment score for a given stock
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Sentiment score and explanation
        """
        if not self.initialized:
            return {
                "error": "Gemini API key not found. Please add GEMINI_API_KEY to your .env file."
            }
        
        try:
            prompt = f"""
            Analyze the current market sentiment for {symbol} stock.
            
            Based on recent news, social media discussions, analyst reports, and institutional activity,
            generate a market sentiment score from -100 (extremely bearish) to +100 (extremely bullish).
            
            Format response as JSON with these keys:
            - sentiment_score: numerical score from -100 to +100
            - sentiment_label: one of [extremely bearish, bearish, slightly bearish, neutral, slightly bullish, bullish, extremely bullish]
            - key_factors: list of factors supporting this sentiment assessment
            - contrarian_indicators: factors that contradict the main sentiment
            """
            
            # Configure Google Search as a tool
            google_search_tool = genai.types.Tool(
                google_search=genai.types.GoogleSearch()
            )
            
            # Generate sentiment analysis
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                )
            )
            
            # Extract and parse as JSON
            try:
                import json
                result = json.loads(response.text)
                result['sources'] = self._extract_sources(response)
                return result
            except json.JSONDecodeError:
                return {
                    "raw_analysis": response.text,
                    "sources": self._extract_sources(response)
                }
                
        except Exception as e:
            return {
                "error": f"Error generating sentiment score: {str(e)}"
            }

# Creating Streamlit UI elements
def render_market_insights_ui():
    st.title("AI Market Insights")
    
    insights = MarketInsights()
    
    if not insights.initialized:
        st.warning("⚠️ Gemini API key not configured. Please add GEMINI_API_KEY to your .env file.")
        st.code("GEMINI_API_KEY=your_api_key_here")
        return
    
    # Selection for stock
    st.subheader("Stock Market Intelligence")
    symbol = st.text_input("Enter Stock Symbol", "AAPL").upper()
    
    tab1, tab2, tab3 = st.tabs(["Market Insights", "Competitive Analysis", "Sentiment Analysis"])
    
    with tab1:
        st.subheader(f"Market Insights for {symbol}")
        
        context = st.text_area("Additional Context or Questions (Optional)", 
                              placeholder="E.g., How might recent Fed decisions affect this stock?")
        
        if st.button("Generate Insights", key="insights_btn"):
            with st.spinner("Generating market insights with AI..."):
                results = insights.get_market_insights(symbol, context)
                
                if "error" in results:
                    st.error(results["error"])
                elif "raw_analysis" in results:
                    st.write(results["raw_analysis"])
                else:
                    if "sentiment" in results:
                        sentiment = results["sentiment"]
                        if sentiment.lower() == "bullish":
                            sentiment_color = "green"
                        elif sentiment.lower() == "bearish":
                            sentiment_color = "red"
                        else:
                            sentiment_color = "orange"
                        
                        st.markdown(f"**Market Sentiment:** :{sentiment_color}[{sentiment}]")
                    
                    if "summary" in results:
                        st.markdown(f"**Summary:** {results['summary']}")
                    
                    if "recent_news" in results:
                        st.subheader("Recent News")
                        for news in results["recent_news"]:
                            st.markdown(f"- **{news.get('date', 'Recent')}**: {news.get('headline', news)}")
                    
                    if "price_targets" in results:
                        st.subheader("Analyst Price Targets")
                        st.write(results["price_targets"])
                    
                    if "risk_factors" in results:
                        st.subheader("Risk Factors")
                        if isinstance(results["risk_factors"], list):
                            for risk in results["risk_factors"]:
                                st.markdown(f"- {risk}")
                        else:
                            st.write(results["risk_factors"])
                    
                    if "opportunities" in results:
                        st.subheader("Opportunities")
                        if isinstance(results["opportunities"], list):
                            for opp in results["opportunities"]:
                                st.markdown(f"- {opp}")
                        else:
                            st.write(results["opportunities"])
                    
                    if "sources" in results and results["sources"]:
                        st.subheader("Sources")
                        for source in results["sources"]:
                            st.markdown(f"- [{source.get('title', 'Source')}]({source.get('uri', '#')})")
    
    with tab2:
        st.subheader(f"Competitive Analysis for {symbol}")
        
        competitor_input = st.text_input("Competitor Symbols (comma separated)", 
                                        placeholder="E.g., MSFT,GOOGL,AMZN")
        
        competitors = None
        if competitor_input:
            competitors = [comp.strip().upper() for comp in competitor_input.split(",")]
        
        if st.button("Generate Competitive Analysis", key="comp_btn"):
            with st.spinner("Analyzing competitive landscape..."):
                results = insights.get_competitive_analysis(symbol, competitors)
                
                if "error" in results:
                    st.error(results["error"])
                elif "raw_analysis" in results:
                    st.write(results["raw_analysis"])
                else:
                    if "comparative_summary" in results:
                        st.markdown(f"**Comparative Summary:** {results['comparative_summary']}")
                    
                    if "valuation_comparison" in results:
                        st.subheader("Valuation Comparison")
                        st.write(results["valuation_comparison"])
                    
                    if "performance_ranking" in results:
                        st.subheader("YTD Performance Ranking")
                        st.write(results["performance_ranking"])
                    
                    if "competitive_advantages" in results:
                        st.subheader(f"Competitive Advantages for {symbol}")
                        if isinstance(results["competitive_advantages"], list):
                            for adv in results["competitive_advantages"]:
                                st.markdown(f"- {adv}")
                        else:
                            st.write(results["competitive_advantages"])
                    
                    if "competitive_disadvantages" in results:
                        st.subheader(f"Competitive Disadvantages for {symbol}")
                        if isinstance(results["competitive_disadvantages"], list):
                            for dis in results["competitive_disadvantages"]:
                                st.markdown(f"- {dis}")
                        else:
                            st.write(results["competitive_disadvantages"])
                    
                    if "market_share_trends" in results:
                        st.subheader("Market Share Trends")
                        st.write(results["market_share_trends"])
                    
                    if "sources" in results and results["sources"]:
                        st.subheader("Sources")
                        for source in results["sources"]:
                            st.markdown(f"- [{source.get('title', 'Source')}]({source.get('uri', '#')})")
    
    with tab3:
        st.subheader(f"Sentiment Analysis for {symbol}")
        
        if st.button("Generate Sentiment Analysis", key="sent_btn"):
            with st.spinner("Analyzing market sentiment..."):
                results = insights.get_market_sentiment_score(symbol)
                
                if "error" in results:
                    st.error(results["error"])
                elif "raw_analysis" in results:
                    st.write(results["raw_analysis"]) 
                else:
                    if "sentiment_score" in results:
                        score = results["sentiment_score"]
                        
                        if isinstance(score, str):
                            try:
                                score = float(score)
                            except:
                                score = 0
                        
                        # Create a gauge chart for sentiment
                        if score > 50:
                            color = "green"
                        elif score < -50:
                            color = "red"
                        elif score > 0:
                            color = "lightgreen"
                        elif score < 0:
                            color = "lightcoral"
                        else:
                            color = "gray"
                        
                        # Normalize for the progress bar (0-100 range)
                        normalized_score = (score + 100) / 2
                        
                        label = results.get("sentiment_label", "")
                        st.markdown(f"### Sentiment Score: **:{color}[{score}]** - {label}")
                        
                        st.progress(normalized_score/100)
                        
                        # Create a simple scale
                        col1, col2, col3, col5, col6 = st.columns(5)
                        with col1:
                            st.write("Extremely Bearish")
                        with col3:
                            st.write("Neutral")
                        with col6:
                            st.write("Extremely Bullish")
                    
                    if "key_factors" in results:
                        st.subheader("Key Factors")
                        if isinstance(results["key_factors"], list):
                            for factor in results["key_factors"]:
                                st.markdown(f"- {factor}")
                        else:
                            st.write(results["key_factors"])
                    
                    if "contrarian_indicators" in results:
                        st.subheader("Contrarian Indicators")
                        if isinstance(results["contrarian_indicators"], list):
                            for indicator in results["contrarian_indicators"]:
                                st.markdown(f"- {indicator}")
                        else:
                            st.write(results["contrarian_indicators"])
                    
                    if "sources" in results and results["sources"]:
                        st.subheader("Sources")
                        for source in results["sources"]:
                            st.markdown(f"- [{source.get('title', 'Source')}]({source.get('uri', '#')})")

if __name__ == "__main__":
    # If this file is run directly, render the UI
    render_market_insights_ui() 