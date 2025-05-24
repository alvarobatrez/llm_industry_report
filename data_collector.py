import os
import serpapi
import requests
from query_parser import QueryParams, QueryParser

class DataCollector:

    def __init__(self):
        self.serpapi_api_key=os.getenv('SERPAPI_API_KEY')
        self.newsapi_api_key=os.getenv('NEWSAPI_API_KEY')

    def research_market(self, params: QueryParams) -> dict:
        data = {
            'market_news': self.get_news(params),
            'search_results': self.google_search(params)
        }
        
        return data
    
    def google_search(self, params):
        params_dict = {
            'engine': 'google',
            'q': f'{params.market} market trends {params.timeframe}'
        }
        client = serpapi.Client(api_key=os.environ.get('SERPAPI_API_KEY'))
        results = client.search(params_dict)
        return results
    
    def get_news(self, params):
        url = 'https://newsapi.org/v2/everything?'
        params = {
            'q': params.market,
            'apiKey': self.newsapi_api_key,
            'sortBy': 'relevancy'
        }
        return requests.get(url, params).json()
