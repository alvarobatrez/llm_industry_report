import streamlit as st
import os
from dotenv import load_dotenv
from query_parser import QueryParser
from data_collector import DataCollector
from analysis import AnalysisEngine
from report_generator import ReportGenerator

load_dotenv()

def main():
    st.set_page_config(page_title='Autonomous Industry Intelligence Report Generation', layout='wide')
    st.title('Autonomous Industry Intelligence Report Generation')
    query = st.text_input('Enter your market research query (e.g. EV market in Spain in 2024)')
    report = None

    if st.button('Generate Report'):
        if query:
            with st.status('Processing query...', expanded=True) as status:
                try:
                    st.write('Parsing query...')
                    parser  = QueryParser()
                    params = parser.parse_query(query=query)

                    st.write('Collecting data...')
                    collector = DataCollector()
                    market_data = collector.research_market(params)

                    st.write('Analyzing trends...')
                    analyzer = AnalysisEngine()
                    analysis = analyzer.analyze_trends(market_data)
                    
                    st.write('Generating report...')
                    generator = ReportGenerator()
                    report = generator.create_report(analysis)

                    status.update(label="Complete!", state="complete")

                except Exception as e:
                    st.error(f'Error in processing: {e}')

        if report is not None:
            st.markdown(report, unsafe_allow_html=True)
        else:
            st.info("Enter a query to generate report")

if __name__ == '__main__':
    main()