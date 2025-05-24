from pydantic import BaseModel
from openai import OpenAI
import os
import json

class QueryParams(BaseModel):
    market: str
    companies: list[str]
    timeframe: str = '5 years'
    geography: str = 'global'

class QueryParser:

    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    def parse_query(self, query: str) -> QueryParams:
        sysmte_promt = """Get principal entities in JSON format:
        - market (str): Principal market
        - companies (list): Principal companies
        - timeframe (str): Time horizon
        - geography (str): Location"""

        response = self.client.chat.completions.create(
            model='gpt-4-1106-preview',
            messages=[{'role': 'system', 'content': sysmte_promt}, {'role': 'user', 'content': query}],
            response_format={'type': 'json_object'}
        )

        result = json.loads(response.choices[0].message.content)

        return QueryParams(**result)