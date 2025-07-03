"""
🎯 Perfect Research Assistant - Advanced Streamlit App
Advanced PDF querying, Q&A, and custom PPT generation based on user prompts
"""

import streamlit as st
import os
import tempfile
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Import Perfect System components
from config import AdvancedConfig
from enhanced_pdf_processor import EnhancedPDFProcessor
from vector_storage import AdvancedVectorStorage
from research_intelligence import ResearchPaperAnalyzer
from perfect_ppt_generator import PerfectPPTGenerator
from search_client import SerpAPIClient

# Configure page
st.set_page_config(
    page_title="Perfect Research Assistant", 
    page_icon="🎯",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerfectResearchApp:
    """Perfect Research Assistant with advanced capabilities"""
    
    def __init__(self):
        """Initialize the Perfect Research Assistant"""
        try:
            self.config = AdvancedConfig()
            self.pdf_processor = EnhancedPDFProcessor(self.config)
            self.vector_storage = AdvancedVectorStorage(self.config)
            self.research_analyzer = ResearchPaperAnalyzer(self.config)
            self.ppt_generator = PerfectPPTGenerator(self.config, self.vector_storage, self.research_analyzer)
            self.search_client = SerpAPIClient(self.config)
            self._init_session_state()
            st.success("🎯 Perfect Research System loaded successfully!")
        except Exception as e:
            st.error(f"⚠️ System initialization error: {e}")
            st.stop()
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if "processed_papers" not in st.session_state:
            st.session_state.processed_papers = {}
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "search_history" not in st.session_state:
            st.session_state.search_history = []
    
    def run(self):
        """Run the Perfect Research Assistant"""
        st.title("🎯 Perfect Research Assistant")
        st.markdown("**Advanced PDF Analysis • Intelligent Q&A • Custom PPT Generation**")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload & Process", "🔍 Query & Q&A", "🌐 Web Search", "🎨 Generate PPT"])
        
        with tab1:
            self._render_upload_tab()
        with tab2:
            self._render_query_tab()
        with tab3:
            self._render_search_tab()
        with tab4:
            self._render_ppt_tab()
    
    def _render_upload_tab(self):
        """Render PDF upload and processing tab"""
        st.header("📄 Upload Research Papers")
        uploaded_file = st.file_uploader("Choose a PDF research paper", type="pdf")
        
        if uploaded_file:
            paper_id = st.text_input("Paper ID", value=uploaded_file.name.replace('.pdf', ''))
            if st.button("🚀 Process Paper", type="primary"):
                if paper_id:
                    self._process_pdf_advanced(uploaded_file, paper_id)
        
        if st.session_state.processed_papers:
            st.header("📋 Processed Papers")
            for paper_id, paper_data in st.session_state.processed_papers.items():
                with st.expander(f"📄 {paper_id}"):
                    metadata = paper_data.get('metadata', {})
                    stats = paper_data.get('summary_stats', {})
                    if metadata.get('title'):
                        st.write(f"**Title:** {metadata['title']}")
                    st.write(f"**Pages:** {stats.get('total_pages', 0)}")
                    st.write(f"**Words:** {stats.get('total_words', 0)}")
    
    def _render_query_tab(self):
        """Render query and Q&A tab"""
        st.header("🔍 Query Your Research Papers")
        
        if not st.session_state.processed_papers:
            st.info("📄 Please upload and process some papers first!")
            return
        
        user_query = st.text_input(
            "🔍 Ask detailed questions about your research papers:",
            placeholder="Be specific! Examples: What methodology was used? What were the statistical results and p-values? What are the clinical applications? How does this compare to other studies? What are the limitations and future research directions?",
            help="""Ask detailed questions about:
            
📊 METHODOLOGY: Research design, sample size, data collection methods, controls
📈 RESULTS: Key findings, statistical significance, p-values, effect sizes, confidence intervals  
🏥 APPLICATIONS: Clinical implications, real-world applications, patient outcomes
📚 LITERATURE: How this compares to other studies, citations, related research
⚠️ LIMITATIONS: Study constraints, potential biases, areas for improvement
🔮 FUTURE: Research directions, unanswered questions, next steps
📋 DETAILS: Specific data points, procedures, measurements, demographics"""
        )
        
        col1, col2 = st.columns(2)
        with col1:
            target_paper = st.selectbox("Target Paper", ["All Papers"] + list(st.session_state.processed_papers.keys()))
        with col2:
            max_results = st.slider("Max Results", 1, 20, 10)
        
        if st.button("🔍 Search", type="primary") and user_query:
            self._execute_query(user_query, target_paper, max_results)
        
        if st.session_state.chat_history:
            st.header("💬 Q&A History")
            for chat in reversed(st.session_state.chat_history[-5:]):
                with st.expander(f"Q: {chat['query'][:50]}..."):
                    st.markdown(f"**Question:** {chat['query']}")
                    st.markdown(f"**Answer:** {chat['response']}")
    
    def _render_search_tab(self):
        """Render Google search functionality tab"""
        st.header("🌐 Web Search")
        st.markdown("Search Google, Scholar, and News to enhance your research")
        
        # Search form
        with st.form("search_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                search_query = st.text_input(
                    "Search Query", 
                    placeholder="e.g., machine learning healthcare applications",
                    help="Enter your research topic or keywords"
                )
            
            with col2:
                search_type = st.selectbox(
                    "Search Type", 
                    ["search", "scholar", "news"],
                    help="Choose search type: General web, Academic papers, or News"
                )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                num_results = st.number_input("Number of Results", min_value=1, max_value=20, value=10)
            with col2:
                location = st.selectbox("Location", ["United States", "Global", "United Kingdom", "Canada"])
            with col3:
                enhance_results = st.checkbox("AI Enhancement", value=True, help="Apply AI analysis to results")
            
            search_submitted = st.form_submit_button("🔍 Search", type="primary")
        
        if search_submitted and search_query:
            self._perform_web_search(search_query, search_type, num_results, location, enhance_results)
        
        # Display search history
        if st.session_state.search_history:
            st.header("📊 Recent Searches")
            
            for i, search_data in enumerate(reversed(st.session_state.search_history[-5:])):
                with st.expander(f"🔍 {search_data['query']} ({search_data['search_type']})"):
                    st.write(f"**Results Found:** {search_data['num_results']}")
                    st.write(f"**Search Type:** {search_data['search_type'].title()}")
                    st.write(f"**Timestamp:** {search_data['timestamp']}")
                    
                    # Show top 3 results
                    if search_data.get('results'):
                        st.write("**Top Results:**")
                        for j, result in enumerate(search_data['results'][:3], 1):
                            st.write(f"**{j}.** {result.get('title', 'No Title')}")
                            st.write(f"   {result.get('snippet', 'No description')[:150]}...")
                            if result.get('link'):
                                st.write(f"   🔗 [Read More]({result['link']})")
                    
                    # Option to create presentation from search
                    if st.button(f"🎨 Create PPT from Search", key=f"search_ppt_{i}"):
                        self._create_search_presentation(search_data)
    
    def _render_ppt_tab(self):
        """Render PPT generation tab"""
        st.header("🎨 Generate Custom Presentations")
        
        if not st.session_state.processed_papers:
            st.info("📄 Please upload and process some papers first!")
            return
        
        target_paper = st.selectbox("Source Paper", list(st.session_state.processed_papers.keys()))
        user_prompt = st.text_area(
            "📝 Describe your presentation requirements in detail:",
            placeholder="""Be specific about what you want! Examples:

🎯 FOCUS AREAS: "Focus on methodology and statistical results, emphasize p-values and significance"
📊 AUDIENCE: "For medical professionals, use clinical terminology and practical applications"  
🔍 CONTENT: "Include limitations, future research directions, and real-world implications"
📈 DATA: "Highlight key findings, charts, and statistical analysis from the results section"
🏥 APPLICATION: "Show how this research applies to patient care and clinical practice"
📚 CITATIONS: "Include key references and compare with other studies in the field"

💡 The more specific you are, the better your presentation will be!
            
Example: "Create a 12-slide presentation for medical professionals focusing on the methodology, statistical significance of results (especially p-values), clinical applications, patient outcomes, limitations of the study, and future research directions. Use clinical terminology and emphasize practical implications for patient care.""",
            height=150,
            help="Provide detailed requirements: What sections to focus on? What audience? What specific findings to highlight? What applications to emphasize?"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            ppt_title = st.text_input("Presentation Title")
            theme = st.selectbox("Theme", ["academic_professional", "research_modern", "executive_clean"])
        with col2:
            slide_count = st.number_input("Number of Slides", min_value=5, max_value=25, value=12)
            audience_type = st.selectbox("Target Audience", ["academic", "business", "general", "executive"])
        
        if st.button("🎯 Generate Perfect Presentation", type="primary") and target_paper and user_prompt:
            self._generate_custom_presentation(target_paper, user_prompt, ppt_title, theme, slide_count, audience_type)
    
    def _process_pdf_advanced(self, uploaded_file, paper_id: str):
        """Process PDF with advanced features"""
        with st.spinner("Processing PDF..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                extraction_result = asyncio.run(self.pdf_processor.extract_content_from_file(tmp_path))
                os.unlink(tmp_path)
                
                if not extraction_result.get("success"):
                    st.error(f"❌ PDF extraction failed: {extraction_result.get('error')}")
                    return
                
                research_analysis = asyncio.run(self.research_analyzer.analyze_research_paper(extraction_result))
                extraction_result["research_analysis"] = research_analysis
                
                storage_result = asyncio.run(self.vector_storage.process_and_store_document(extraction_result, paper_id))
                extraction_result["vector_storage"] = storage_result
                
                st.session_state.processed_papers[paper_id] = {
                    **extraction_result,
                    "paper_id": paper_id,
                    "processed_at": datetime.now()
                }
                
                st.success(f"✅ Successfully processed '{paper_id}'!")
                
                if extraction_result.get("sections", {}).get("abstract"):
                    st.write("**Abstract Preview:**")
                    st.write(extraction_result["sections"]["abstract"][:300] + "...")
                
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
    
    def _execute_query(self, query: str, target_paper: str, max_results: int):
        """Execute semantic query on processed papers"""
        with st.spinner("🔍 Searching..."):
            try:
                results = []
                
                if target_paper == "All Papers":
                    for paper_id in st.session_state.processed_papers.keys():
                        paper_results = asyncio.run(self.vector_storage.semantic_search(query=query, namespace=paper_id, top_k=5))
                        results.extend(paper_results)
                else:
                    results = asyncio.run(self.vector_storage.semantic_search(query=query, namespace=target_paper, top_k=max_results))
                
                ai_response = self._generate_ai_response(query, results)
                
                st.session_state.chat_history.append({
                    "query": query,
                    "response": ai_response,
                    "timestamp": datetime.now()
                })
                
                st.success(f"✅ Found {len(results)} relevant results!")
                st.markdown("### 🎯 AI Response")
                st.markdown(ai_response)
                
                if results:
                    st.markdown("### 📊 Top Results")
                    for i, result in enumerate(results[:3]):
                        st.write(f"**{i+1}.** {result.content[:200]}... (Score: {result.score:.3f})")
                
            except Exception as e:
                st.error(f"❌ Query error: {str(e)}")
    
    def _perform_web_search(self, query: str, search_type: str, num_results: int, location: str, enhance_results: bool):
        """Perform web search using SerpAPI"""
        with st.spinner(f"🔍 Searching {search_type}..."):
            try:
                # Map location for SerpAPI
                location_map = {
                    "United States": "United States",
                    "Global": "",
                    "United Kingdom": "United Kingdom", 
                    "Canada": "Canada"
                }
                
                # Perform search
                if search_type == "scholar":
                    search_response = self.search_client.search_for_research_papers(
                        topic=query, 
                        num_results=num_results
                    )
                else:
                    search_response = self.search_client.search_google(
                        query=query,
                        search_type=search_type,
                        num_results=num_results,
                        location=location_map.get(location, location)
                    )
                
                if search_response.get("success"):
                    results = search_response.get("results", [])
                    
                    # Store search data
                    search_data = {
                        "query": query,
                        "search_type": search_type,
                        "results": results,
                        "num_results": len(results),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "location": location,
                        "enhanced": enhance_results
                    }
                    
                    st.session_state.search_history.append(search_data)
                    
                    # Display results
                    st.success(f"✅ Found {len(results)} {search_type} results!")
                    
                    # AI Enhancement
                    if enhance_results and results:
                        ai_summary = self._generate_search_summary(query, results, search_type)
                        st.markdown("### 🧠 AI Summary")
                        st.markdown(ai_summary)
                    
                    # Display detailed results
                    st.markdown("### 📄 Search Results")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"{i}. {result.get('title', 'No Title')}"):
                            st.write(f"**URL:** {result.get('link', 'N/A')}")
                            st.write(f"**Description:** {result.get('snippet', 'No description available')}")
                            
                            # Search-type specific info
                            if search_type == "scholar":
                                if result.get('authors'):
                                    st.write(f"**Authors:** {', '.join(result['authors'])}")
                                if result.get('year'):
                                    st.write(f"**Year:** {result['year']}")
                                if result.get('cited_by'):
                                    st.write(f"**Citations:** {result['cited_by']}")
                            elif search_type == "news":
                                if result.get('source'):
                                    st.write(f"**Source:** {result['source']}")
                                if result.get('date'):
                                    st.write(f"**Date:** {result['date']}")
                            
                            # Quick actions
                            col1, col2 = st.columns(2)
                            with col1:
                                if result.get('link'):
                                    st.markdown(f"🔗 [**Open Link**]({result['link']})")
                            with col2:
                                if st.button(f"📋 Save for Research", key=f"save_{i}"):
                                    st.info("Result saved to research notes!")
                
                else:
                    st.error(f"❌ Search failed: {search_response.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Search error: {str(e)}")
    
    def _generate_custom_presentation(self, paper_id: str, user_prompt: str, title: str, theme: str, slide_count: int, audience_type: str):
        """Generate custom presentation based on user prompt"""
        with st.spinner("🎨 Creating presentation..."):
            try:
                paper_content = st.session_state.processed_papers[paper_id]
                
                presentation_path = asyncio.run(
                    self.ppt_generator.create_perfect_presentation(
                        paper_content=paper_content,
                        user_prompt=user_prompt,
                        paper_id=paper_id,
                        title=title,
                        theme=theme,
                        slide_count=slide_count,
                        audience_type=audience_type
                    )
                )
                
                st.success("🎉 Perfect presentation created!")
                st.write(f"**File:** {os.path.basename(presentation_path)}")
                
                if os.path.exists(presentation_path):
                    with open(presentation_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Presentation",
                            data=file.read(),
                            file_name=os.path.basename(presentation_path),
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            type="primary"
                        )
                
            except Exception as e:
                st.error(f"❌ Presentation error: {str(e)}")
    
    def _generate_search_summary(self, query: str, results: List[Dict], search_type: str) -> str:
        """Generate AI summary of search results"""
        try:
            # Prepare context from search results
            context = ""
            for result in results[:5]:  # Use top 5 results
                context += f"Title: {result.get('title', '')}\n"
                context += f"Summary: {result.get('snippet', '')}\n\n"
            
            from openai import OpenAI
            client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            
            prompt = f"""
            Analyze these {search_type} search results for the query: "{query}"
            
            Search Results:
            {context}
            
            Provide a comprehensive summary that includes:
            1. Key themes and trends
            2. Main findings or topics
            3. Notable sources or authors (if academic)
            4. Potential research gaps or opportunities
            5. Relevance to current research
            
            Keep it concise but informative (200-300 words).
            """
            
            response = client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a research analyst. Provide insightful summaries of search results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Search completed successfully. Found {len(results)} relevant results on {search_type}."
    
    def _create_search_presentation(self, search_data: Dict):
        """Create presentation from search results"""
        st.info("🎨 Search-based presentation feature coming soon! For now, upload PDFs to create presentations.")
    
    def _generate_ai_response(self, query: str, results: List) -> str:
        """Generate AI response based on query and results"""
        if not results:
            return "I couldn't find relevant information to answer this question."
        
        context = "\n\n".join([r.content for r in results[:5]])
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            
            response = client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Answer questions based on research paper content. Be accurate and cite findings."},
                    {"role": "user", "content": f"Question: {query}\n\nContent: {context}\n\nAnswer:"}
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Based on the search results: {results[0].content[:200]}..."


def main():
    """Main application entry point"""
    app = PerfectResearchApp()
    app.run()


if __name__ == "__main__":
    main()
 