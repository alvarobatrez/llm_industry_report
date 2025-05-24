from openai import OpenAI
import os
from typing import List, Union
from tenacity import retry, stop_after_attempt, wait_exponential
import json
from datetime import datetime, timezone

class AnalysisEngine:

    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.analysis_prompts = {
            'trends': 'Identify key market trends supported by quantitative and qualitative data',
            'competitors': '''Analyze key competitors considering:
            - Market share
            - Competitive advantages
            - Recent strategic moves
            - key financials''',
            'swot': '''Conduct a detailed SWOT analysis considering:
            1. Target market strengths
            2. Current weaknesses
            3. Emerging opportunities
            4. Competitive threats'''
        }

    def analyze_trends(self, data: dict) -> dict:
        processed_data = self.preprocess_data(data)

        analysis = {
            'trends': self.analyze_with_fallback(
                data=processed_data,
                prompt=self.analysis_prompts['trends'],
                analysis_type='trends'
            ),
            'competitors': self.analyze_competitors(processed_data),
            'swot': self.swot_analysis(processed_data)
        }

        return self.postprocess_analysis(analysis)

    def preprocess_data(self, raw_data: dict) -> dict:
        data = {
            'news': self.extract_text(raw_data.get('market_news', {}).get('articles', [])),
            'search_results': self.process_search_results(raw_data.get('search_results', {}))
        }
        return data

    def extract_text(self, articles: List[dict]) -> List[str]:
        return [f"{a['title']}: {a['description']}" for a in articles if a.get('description')]
    
    def process_search_results(self, results: dict) -> str:
        results = '\n'.join(f'{r.get("title", "")}: {r.get("snippet", "")}'
                            for r in results.get('organic_results', [])[:20])
        return results

    def analyze_with_fallback(self, data: dict, prompt: str, analysis_type: str) -> Union[dict, str]:
        return self.structured_analysis(
            data=data,
            system_prompt=self.get_system_prompt(analysis_type),
            user_prompt=prompt
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def structured_analysis(self, data: dict, system_prompt: str, user_prompt: str) -> dict:
        chunks = self.chunk_data(data)
        responses = []

        for chunk in chunks:
            response = self.client.chat.completions.create(
                model='gpt-4-1106-preview',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f'{user_prompt}\n\nRelevant data:\n{chunk}'}
                ],
                response_format={'type': 'json_object'},
                temperature=0.3
            )
            responses.append(json.loads(response.choices[0].message.content))

        return self.merge_responses(responses)

    def chunk_data(self, data: dict, max_items: int = 5) -> List[dict]:
        chunks = []
        current_chunk = []
        
        for article in data.get('news', [])[:20]:
            current_chunk.append({'news': article})
            if len(current_chunk) >= max_items:
                chunks.append({'news': current_chunk})
                current_chunk = []
        
        search_items = data.get('search_results', '').split('\n')[:10]
        for i in range(0, len(search_items), max_items):
            chunks.append({'search_results': search_items[i:i+max_items]})
        
        return chunks
    
    def merge_responses(self, responses: List) -> dict:
        merged = {}
        for response in responses:
            for key, value in response.items():
                if isinstance(value, list):
                    merged.setdefault(key, []).extend(value)
                elif isinstance(value, dict):
                    merged.setdefault(key, {}).update(value)
                else:
                    merged[key] = f'{merged.get(key, '')}\n{value}'.strip()
        
        return merged
    
    def get_system_prompt(self, analysis_type: str) -> str:
        prompts = {
            "trends": """Generates a trend analysis in JSON format:
            {
                "trends": [{
                    "name": str,
                    "description": str,
                    "evidence": [str],
                    "impact_score": float
                }],
                "summary": str
            }""",
            "competitors": """Returns JSON with:
            {
                "top_competitors": [{
                    "name": str,
                    "market_share": float,
                    "strengths": [str],
                    "weaknesses": [str],
                    "recent_activity": [str]
                }],
                "competitive_landscape": str
            }""",
            "swot": """Returns SWOT in JSON format:
            {
                "strengths": { "description": str, "evidence": [str] },
                "weaknesses": { "description": str, "evidence": [str] },
                "opportunities": { "description": str, "evidence": [str] },
                "threats": { "description": str, "evidence": [str] }
            }"""
        }

        return prompts.get(analysis_type, 'Parses data and returns results in JSON format')
    
    def analyze_competitors(self, data: dict) -> dict:
        system_prompt = """You are a senior strategic analyst. Return JSON with:
        {
            "top_competitors": [{
                "name": str,
                "market_share": float,
                "strengths": [str],
                "weaknesses": [str],
                "recent_activity": [str]
            }],
            "competitive_landscape": str
        }"""

        return self.structured_analysis(
            data=data,
            system_prompt=system_prompt,
            user_prompt=self.analysis_prompts['competitors']
        )
    
    def swot_analysis(self, data: dict) -> dict:
        system_prompt = """Returns SWOT in JSON format:
        {
            "strengths": { "description": str, "evidence": [str] },
            "weaknesses": { "description": str, "evidence": [str] },
            "opportunities": { "description": str, "evidence": [str] },
            "threats": { "description": str, "evidence": [str] }
        }"""
        
        return self.structured_analysis(
            data=data,
            system_prompt=system_prompt,
            user_prompt=self.analysis_prompts["swot"]
        )
    
    def postprocess_analysis(self, analysis: dict) -> dict:
        analysis = self.normalize_texts(analysis)
        analysis = self.remove_duplicates(analysis)
        self.validate_data_structure(analysis)
        analysis = self.strategic_sorting(analysis)
        self.validate_analysis_quality(analysis)
        analysis = self.add_metadata(analysis)
        return analysis
    
    def normalize_texts(self, analysis: dict) -> dict:
        def process_value(value):
            if isinstance(value, str):
                return value.strip().capitalize()
            if isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [process_value(item) for item in value]
            return value

        text_fields = ['trends', 'competitive_landscape', 'summary']
        for field in text_fields:
            if field in analysis:
                analysis[field] = process_value(analysis[field])
        return analysis
    
    def remove_duplicates(self, analysis: dict) -> dict:
        def create_hash(item):
            if isinstance(item, dict):
                return hash(frozenset(sorted(item.items())))
            return hash(str(item))

        for key in ['trends', 'strengths', 'weaknesses']:
            if key in analysis:
                seen = set()
                unique_list = []
                for item in analysis[key]:
                    item_hash = create_hash(item)
                    if item_hash not in seen:
                        seen.add(item_hash)
                        unique_list.append(item)
                analysis[key] = unique_list
        return analysis
    
    def validate_data_structure(self, analysis: dict):
        required_competitor_fields = {'name', 'market_share'}
        
        if 'competitors' in analysis:
            for competitor in analysis['competitors'].get('top_competitors', []):
                missing = required_competitor_fields - competitor.keys()
                if missing:
                    print(f"Missing field in competitors {competitor.get('name')}: {missing}")
                
                if not isinstance(competitor.get('market_share', 0), (int, float)):
                    print(f"Invalid value in market_share for {competitor.get('name')}")
    
    def strategic_sorting(self, analysis: dict) -> dict:
        def safe_get(value, key, default=0.0):
            
            if isinstance(value, dict):
                val = value.get(key, default)
            else:
                val = default
            
            try:
                return float(val) if val is not None else default
            except (TypeError, ValueError):
                return default

        if 'trends' in analysis:
            
            valid_trends = [t for t in analysis['trends'] if isinstance(t, dict) and 'impact_score' in t]
            
            valid_trends.sort(key=lambda x: safe_get(x, 'impact_score'), reverse=True)
            
            invalid_trends = [t for t in analysis['trends'] if not isinstance(t, dict) or 'impact_score' not in t]
            
            analysis['trends'] = valid_trends + invalid_trends

        return analysis
    
    def validate_analysis_quality(self, analysis: dict):
        metrics = {
            'min_competitors': 2,
            'min_trends': 2,
            'max_weaknesses': 5
        }
        
        competitors = analysis.get('competitors', {}).get('top_competitors', [])
        if len(competitors) < metrics['min_competitors']:
            print(f"Warning: Only {len(competitors)} competitors identified")
        
        trends = analysis.get('trends', [])
        if len(trends) < metrics['min_trends']:
            print(f"Warning: Only {len(trends)} trends identified")
        
        weaknesses = analysis.get('swot', {}).get('weaknesses', {}).get('evidence', [])
        if len(weaknesses) > metrics['max_weaknesses']:
            print(f"Warning: {len(weaknesses)} weaknesses found")

    def add_metadata(self, analysis: dict) -> dict:
        analysis['metadata'] = {
            'processing_date': datetime.now(timezone.utc).isoformat(),
            'data_sources': ['web_search'],
            'model_version': 'gpt-4-1106-preview',
            'quality_score': self.calculate_quality_score(analysis)
        }
        return analysis

    def calculate_quality_score(self, analysis: dict) -> float:
        score = 0.3
        trend_count = len(analysis.get('trends', []))
        score += min(trend_count * 0.1, 0.3)
        return min(score, 1.0)