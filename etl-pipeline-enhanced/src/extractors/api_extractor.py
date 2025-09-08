"""
API Data Extractor
Extracts data from various APIs
"""

import requests
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
from urllib.parse import urljoin
import pandas as pd

logger = logging.getLogger(__name__)

class APIExtractor:
    """Extract data from REST APIs"""
    
    def __init__(self, base_url: str = None, auth_token: str = None):
        """Initialize API extractor"""
        self.base_url = base_url
        self.auth_token = auth_token
        self.session = requests.Session()
        
        if auth_token:
            self.session.headers.update({
                'Authorization': f'Bearer {auth_token}'
            })
    
    def extract(self, endpoint: str, params: Dict[str, Any] = None,
                method: str = 'GET', data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract data from API endpoint"""
        url = urljoin(self.base_url, endpoint) if self.base_url else endpoint
        
        logger.info(f"Extracting data from {url}")
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=30
            )
            
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def extract_paginated(self, endpoint: str, page_size: int = 100,
                         max_pages: int = None) -> List[Dict[str, Any]]:
        """Extract paginated data from API"""
        all_data = []
        page = 1
        
        while True:
            params = {
                'page': page,
                'page_size': page_size
            }
            
            response = self.extract(endpoint, params=params)
            
            # Handle different pagination response formats
            if isinstance(response, dict):
                data = response.get('results', response.get('data', []))
                total_pages = response.get('total_pages', 1)
            else:
                data = response
                total_pages = 1
            
            all_data.extend(data)
            
            if page >= total_pages or (max_pages and page >= max_pages):
                break
            
            page += 1
            time.sleep(0.1)  # Rate limiting
        
        logger.info(f"Extracted {len(all_data)} records from {page} pages")
        return all_data
    
    def extract_weather_data(self, city: str, api_key: str) -> Dict[str, Any]:
        """Extract weather data from OpenWeatherMap API"""
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': api_key,
            'units': 'metric'
        }
        
        return self.extract(url, params=params)
    
    def extract_financial_data(self, symbol: str, api_key: str) -> pd.DataFrame:
        """Extract financial data from Alpha Vantage API"""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': api_key,
            'outputsize': 'compact'
        }
        
        data = self.extract(url, params=params)
        
        # Convert to DataFrame
        time_series = data.get('Time Series (Daily)', {})
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.astype(float)
        
        return df
    
    def extract_news_data(self, query: str, api_key: str,
                         from_date: str = None) -> List[Dict[str, Any]]:
        """Extract news data from NewsAPI"""
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': api_key,
            'sortBy': 'publishedAt',
            'language': 'en'
        }
        
        if from_date:
            params['from'] = from_date
        
        response = self.extract(url, params=params)
        return response.get('articles', [])
    
    def extract_github_data(self, owner: str, repo: str) -> Dict[str, Any]:
        """Extract repository data from GitHub API"""
        endpoint = f"https://api.github.com/repos/{owner}/{repo}"
        
        repo_data = self.extract(endpoint)
        
        # Get additional data
        commits = self.extract(f"{endpoint}/commits", params={'per_page': 10})
        issues = self.extract(f"{endpoint}/issues", params={'state': 'open', 'per_page': 10})
        
        return {
            'repository': repo_data,
            'recent_commits': commits,
            'open_issues': issues
        }
    
    def extract_with_retry(self, endpoint: str, max_retries: int = 3,
                          backoff_factor: float = 2.0) -> Dict[str, Any]:
        """Extract data with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.extract(endpoint)
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                
                wait_time = backoff_factor ** attempt
                logger.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
