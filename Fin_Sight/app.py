        # Fetch real-time sentiment from Alpha Vantage (hardcoded key)
        @st.cache_data(ttl=300)  # Cache for 5 minutes
        def fetch_real_time_sentiment(ticker):
            try:
                # Replace with your Alpha Vantage API key
                api_key = "D8VCWYUPOFJR8D52"  # Insert your key here (e.g., "ABC123XYZ")
                if not api_key or api_key == "your_alphavantage_key_here":
                    st.warning("Please replace 'your_alphavantage_key_here' with a valid Alpha Vantage API key. Get one at https://www.alphavantage.co/support/#api-key and restart the app after updating.")
                    raise ValueError("No valid API key provided.")
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
                response = requests.get(url)
                if response.status_code != 200:
                    raise ValueError(f"Alpha Vantage API error: Status code {response.status_code}")
                data = response.json()
                if 'feed' not in data:
                    raise ValueError("No sentiment data returned in API response.")
                
                # Extract sentiment scores and articles
                posts = []
                links = []
                sentiment_scores = []
                articles = data['feed'][:10]
                for article in articles:
                    title = article.get('title', 'No title')
                    summary = article.get('summary', '')
                    post = f"{title}: {summary[:100]}..."  # Shortened for display
                    posts.append(post)
                    links.append(article.get('url', '#'))
                    
                    # Extract and validate sentiment score
                    ticker_sent = article.get('ticker_sentiment', [])
                    score = 0.0
                    if ticker_sent:
                        for ts in ticker_sent:
                            if ts['ticker'] == ticker:
                                score_value = ts.get('ticker_sentiment_score', None)
                                if score_value is not None and isinstance(score_value, (int, float)):
                                    score = float(score_value)  # Ensure numeric
                    sentiment_scores.append(score)
                
                # Compute average only with valid numeric scores
                valid_scores = [s for s in sentiment_scores if isinstance(s, (int, float)) and not np.isnan(s)]
                avg_score = np.mean(valid_scores) if valid_scores else 0.0
                
                # Fallback if no articles
                if not posts:
                    posts = [f"Average sentiment score for {ticker}: {avg_score}"]
                    links = ['#']
                
                return posts, links, avg_score
            except Exception as e:
                st.warning(f"Alpha Vantage failed: {str(e)}. Using yfinance fallback.")
                # Fallback to yfinance news
                ticker_obj = yf.Ticker(ticker)
                news = ticker_obj.news[:10]
                posts = [f"{article.get('title', 'No title')} - {article.get('publisher', 'Unknown')}" for article in news if article.get('title')]
                links = [article.get('link', '#') for article in news]
                avg_score = 0.0  # Default for fallback
                return posts if posts else ["Sample: Neutral market news."], ['#'], avg_score

        sample_posts, links, avg_sentiment_from_api = fetch_real_time_sentiment(selected_ticker)

        # ... (Rest of the tabs remain unchanged)

        elif selected_tab == "Insights":
            st.header("Insights")
            avg_sentiment = avg_sentiment_from_api  # Use Alpha Vantage's direct score
            if avg_sentiment > 0.1:
                sentiment_note = "🟢 Bullish sentiment detected—consider long positions if predictions align."
            elif avg_sentiment < -0.1:
                sentiment_note = "🔴 Bearish sentiment—watch for downside risk in forecasts."
            else:
                sentiment_note = "⚪ Neutral sentiment—rely on technical indicators."
            if avg_sentiment == 0.0 and not sample_posts[0].startswith("Average sentiment score"):
                st.warning("Fallback data used; real-time sentiment unavailable. Please verify your API key is valid and restart the app.")
            st.write(f"💡 Average sentiment score: {avg_sentiment:.2f}")
            st.write(sentiment_note)
            st.write("Real-time news sentiment can amplify ML predictions—e.g., high positive scores narrow confidence intervals.")
