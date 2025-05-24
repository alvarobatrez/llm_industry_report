from openai import OpenAI
from datetime import datetime
import os
from typing import Dict, Any

class ReportGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4-1106-preview"

    def create_report(self, analysis: Dict[str, Any]) -> str:
        structured_prompt = self._build_prompt(analysis)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": structured_prompt}],
            temperature=0.3,
            top_p=0.95
        )
        
        return response.choices[0].message.content

    def _build_prompt(self, data: Dict[str, Any]) -> str:
        return f"""
As a senior strategic director, analyze this market data and generate an executive report in Markdown:

# Required Structure:
## 🎯 Executive Summary (1 paragraph)
- Key market dynamics
- Main strategic opportunities
- Critical risks to mitigate

## 📊 Trend Analysis
- Top 3 disruptive trends
- Sector growth projection
- Relevant emerging technologies

## 🥇 Competitive Landscape
### Comparison Table (Top 5)
{self._format_competitors_table(data['competitors']['top_competitors'][:5])}

### Strategic Map
- Positioning by market segment
- Key competitive advantages

## 🚀 Strategic Recommendations
- Investment priorities (short/medium term)
- Recommended strategic alliances
- Technological innovations to be developed

## ✅ Action Plan
- Key initiatives for the next 90 days
- Success metrics (KPIs)
- Recommended resource allocation

# Input Data:
{self._format_input_data(data)}

# Formatting Instructions:
1. Use executive but concise language
2. Highlight key figures in bold
3. Include a condensed SWOT analysis
4. Prioritize actionable insights
"""

    def _format_competitors_table(self, competitors: list) -> str:
        table = "| Company | Participation | Key Strengths | Critical Weaknesses |\n"
        table += "|---------|---------------|------------------|----------------------|\n"
        
        for comp in competitors:
            table += (
                f"| {comp['name']} "
                f"| {self._format_market_share(comp['market_share'])} "
                f"| {', '.join(comp['strengths'][:2])} "
                f"| {', '.join(comp['weaknesses'][:2])} |\n"
            )
        return table

    def _format_input_data(self, data: Dict[str, Any]) -> str:
        return f"""
### Market Context
- Analysis Date: {datetime.fromisoformat(data['metadata']['processing_date']).strftime('%d/%m/%Y')}
- Data Sources: {', '.join(data['metadata']['data_sources'])}
- Analysis Quality: {data['metadata']['quality_score']}/1.0

### SWOT Analysis
{self._format_swot(data['swot'])}
"""

    def _format_trends(self, trends: Any) -> str:
        if isinstance(trends, list):
            return "\n".join(f"- {t}" for t in trends if isinstance(t, str))
        return "- No clear trends were detected"

    def _format_swot(self, swot: Dict[str, Any]) -> str:
        return "\n".join(
            f"**{category.title()}:** {swot[category]['description']}\n"
            f"Evidencia: {', '.join(swot[category]['evidence'][:2])}\n"
            for category in ['strengths', 'weaknesses', 'opportunities', 'threats']
        )

    def _format_financials(self, financials: Dict[str, float]) -> str:
        if not financials:
            return "No financial data available"
        return "\n".join(f"- {k}: {v:.2f}%" if '%' in k else f"- {k}: ${v:.2f}B" 
                       for k, v in financials.items())

    def _format_market_share(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{value}%"
        return "N/A"