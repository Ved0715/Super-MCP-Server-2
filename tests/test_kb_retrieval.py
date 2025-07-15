#!/usr/bin/env python3
"""
🧪 Test Script for Advanced Knowledge Base Retrieval System
Tests the new intelligent KB retrieval API with various query types
"""

import asyncio
import json
import time
import sys
from typing import Dict, Any, List

# Test the new KB retrieval system
try:
    from knowledge_base_retrieval import AdvancedKnowledgeBaseRetriever
    print("✅ Successfully imported AdvancedKnowledgeBaseRetriever")
except ImportError as e:
    print(f"❌ Failed to import AdvancedKnowledgeBaseRetriever: {e}")
    sys.exit(1)

class KnowledgeBaseRetrievalTester:
    """Comprehensive test suite for the advanced knowledge base retrieval system"""
    
    def __init__(self):
        self.retriever = None
        self.test_results = []
        
    async def setup(self):
        """Initialize the retriever for testing"""
        try:
            print("🔄 Initializing Advanced Knowledge Base Retriever...")
            self.retriever = AdvancedKnowledgeBaseRetriever()
            print("✅ Retriever initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize retriever: {e}")
            return False
    
    async def test_query_understanding(self):
        """Test query understanding and classification"""
        print("\n🧠 Testing Query Understanding...")
        
        test_queries = [
            "What is machine learning?",
            "Create a study plan for deep learning",
            "Analyze books about algorithms",
            "Show chapters from machine learning book",
            "How to implement neural networks?",
            "Compare supervised vs unsupervised learning"
        ]
        
        try:
            for query in test_queries:
                print(f"  📝 Testing: '{query}'")
                analysis = self.retriever.query_processor.understand_query(query)
                print(f"     Type: {analysis.get('type', 'unknown')}")
                print(f"     Difficulty: {analysis.get('difficulty', 'unknown')}")
                print(f"     Mathematical: {analysis.get('mathematical', False)}")
                
            self.test_results.append({
                "test": "query_understanding",
                "status": "passed",
                "message": "Query understanding working correctly"
            })
            print("✅ Query understanding tests passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "query_understanding",
                "status": "failed",
                "message": f"Query understanding failed: {e}"
            })
            print(f"❌ Query understanding test failed: {e}")
    
    async def test_knowledge_base_inventory(self):
        """Test knowledge base inventory retrieval"""
        print("\n📊 Testing Knowledge Base Inventory...")
        
        try:
            inventory = await self.retriever.vector_storage.get_knowledge_base_inventory()
            
            print(f"  📚 Books found: {len(inventory.get('books', []))}")
            print(f"  📄 Total chunks: {inventory.get('total_chunks', 0)}")
            print(f"  🏷️ Namespace: {inventory.get('namespace', 'unknown')}")
            print(f"  📇 Index: {inventory.get('index_name', 'unknown')}")
            
            if len(inventory.get('books', [])) > 0:
                print("  📖 Available books:")
                for book in inventory.get('books', []):
                    print(f"    - {book}")
            
            self.test_results.append({
                "test": "knowledge_base_inventory",
                "status": "passed",
                "message": f"Found {len(inventory.get('books', []))} books with {inventory.get('total_chunks', 0)} chunks"
            })
            print("✅ Knowledge base inventory test passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "knowledge_base_inventory",
                "status": "failed",
                "message": f"Knowledge base inventory failed: {e}"
            })
            print(f"❌ Knowledge base inventory test failed: {e}")
    
    async def test_basic_search(self):
        """Test basic search functionality"""
        print("\n🔍 Testing Basic Search...")
        
        test_queries = [
            "machine learning",
            "neural networks",
            "algorithms",
            "data science"
        ]
        
        try:
            for query in test_queries:
                print(f"  🔎 Searching: '{query}'")
                results = await self.retriever.vector_storage.enhanced_knowledge_base_search(
                    query=query,
                    namespace="knowledge-base",
                    top_k=3
                )
                
                print(f"    📊 Found {len(results)} results")
                if results:
                    print(f"    🎯 Top result score: {results[0].get('score', 0):.3f}")
                    print(f"    📖 From book: {results[0].get('book_name', 'Unknown')}")
                
            self.test_results.append({
                "test": "basic_search",
                "status": "passed",
                "message": "Basic search functionality working"
            })
            print("✅ Basic search tests passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "basic_search",
                "status": "failed",
                "message": f"Basic search failed: {e}"
            })
            print(f"❌ Basic search test failed: {e}")
    
    async def test_intelligent_search(self):
        """Test intelligent search with different query types"""
        print("\n🧠 Testing Intelligent Search...")
        
        test_scenarios = [
            {
                "query": "What is machine learning?",
                "expected_type": "search"
            },
            {
                "query": "Create a study plan for deep learning",
                "expected_type": "study_plan"
            },
            {
                "query": "Analyze books about algorithms",
                "expected_type": "book_analysis"
            },
            {
                "query": "Show chapters from machine learning book",
                "expected_type": "chapters"
            }
        ]
        
        try:
            for scenario in test_scenarios:
                query = scenario["query"]
                expected_type = scenario["expected_type"]
                
                print(f"  🔍 Testing: '{query}'")
                result = await self.retriever.intelligent_search(query)
                
                if result.get("success", False):
                    response_type = result.get("response_type", "unknown")
                    print(f"    ✅ Success - Type: {response_type}")
                    
                    if response_type == "search":
                        print(f"    📊 Results: {result.get('total_results', 0)}")
                    elif response_type == "study_plan":
                        print(f"    📚 Study plan generated for: {result.get('topic', 'unknown')}")
                    elif response_type == "book_analysis":
                        print(f"    📖 Analyzed {result.get('total_books', 0)} books")
                    elif response_type == "chapters":
                        print(f"    📑 Chapter extraction completed")
                else:
                    print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
                
            self.test_results.append({
                "test": "intelligent_search",
                "status": "passed",
                "message": "Intelligent search routing working correctly"
            })
            print("✅ Intelligent search tests passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "intelligent_search",
                "status": "failed",
                "message": f"Intelligent search failed: {e}"
            })
            print(f"❌ Intelligent search test failed: {e}")
    
    async def test_study_plan_generation(self):
        """Test study plan generation"""
        print("\n📚 Testing Study Plan Generation...")
        
        study_queries = [
            "Create a study plan for machine learning",
            "I want to learn data science from scratch",
            "Help me plan my deep learning studies"
        ]
        
        try:
            for query in study_queries:
                print(f"  📝 Generating study plan for: '{query}'")
                result = await self.retriever.intelligent_search(query)
                
                if result.get("success", False) and result.get("response_type") == "study_plan":
                    study_plan = result.get("study_plan", {})
                    print(f"    📖 Topic: {study_plan.get('topic', 'unknown')}")
                    print(f"    🎯 Difficulty: {study_plan.get('difficulty', 'unknown')}")
                    print(f"    ⏱️ Duration: {study_plan.get('estimated_duration', 'unknown')}")
                    print(f"    📚 Books: {len(study_plan.get('relevant_books', []))}")
                else:
                    print(f"    ❌ Failed to generate study plan")
                
            self.test_results.append({
                "test": "study_plan_generation",
                "status": "passed",
                "message": "Study plan generation working"
            })
            print("✅ Study plan generation tests passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "study_plan_generation",
                "status": "failed",
                "message": f"Study plan generation failed: {e}"
            })
            print(f"❌ Study plan generation test failed: {e}")
    
    async def test_book_analysis(self):
        """Test book analysis functionality"""
        print("\n📖 Testing Book Analysis...")
        
        analysis_queries = [
            "Analyze books about machine learning",
            "Which books cover neural networks?",
            "Compare available data science books"
        ]
        
        try:
            for query in analysis_queries:
                print(f"  🔍 Analyzing: '{query}'")
                result = await self.retriever.intelligent_search(query)
                
                if result.get("success", False) and result.get("response_type") == "book_analysis":
                    book_analysis = result.get("book_analysis", {})
                    print(f"    📚 Books analyzed: {len(book_analysis)}")
                    for book_name, analysis in book_analysis.items():
                        print(f"    📖 {book_name}:")
                        print(f"      📊 Chunks: {analysis.get('total_chunks', 0)}")
                        print(f"      🔢 Mathematical: {analysis.get('has_mathematical_content', False)}")
                        print(f"      📈 Relevance: {analysis.get('relevance_score', 0):.3f}")
                else:
                    print(f"    ❌ Failed to analyze books")
                
            self.test_results.append({
                "test": "book_analysis",
                "status": "passed",
                "message": "Book analysis working correctly"
            })
            print("✅ Book analysis tests passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "book_analysis",
                "status": "failed",
                "message": f"Book analysis failed: {e}"
            })
            print(f"❌ Book analysis test failed: {e}")
    
    async def test_chapter_extraction(self):
        """Test chapter extraction functionality"""
        print("\n📑 Testing Chapter Extraction...")
        
        chapter_queries = [
            "Show chapters from machine learning book",
            "List table of contents for data science books",
            "Get chapter structure of available books"
        ]
        
        try:
            for query in chapter_queries:
                print(f"  📋 Extracting chapters for: '{query}'")
                result = await self.retriever.intelligent_search(query)
                
                if result.get("success", False) and result.get("response_type") == "chapters":
                    chapters = result.get("chapters", {})
                    if isinstance(chapters, dict):
                        for book_name, book_chapters in chapters.items():
                            print(f"    📖 {book_name}: {len(book_chapters)} chapters")
                    else:
                        print(f"    📑 Chapters extracted: {len(chapters)}")
                else:
                    print(f"    ❌ Failed to extract chapters")
                
            self.test_results.append({
                "test": "chapter_extraction",
                "status": "passed",
                "message": "Chapter extraction working correctly"
            })
            print("✅ Chapter extraction tests passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "chapter_extraction",
                "status": "failed",
                "message": f"Chapter extraction failed: {e}"
            })
            print(f"❌ Chapter extraction test failed: {e}")
    
    async def test_system_components(self):
        """Test individual system components"""
        print("\n🔧 Testing System Components...")
        
        components = {
            "retriever": self.retriever is not None,
            "vector_storage": self.retriever.vector_storage is not None,
            "openai_client": self.retriever.openai_client is not None,
            "query_processor": self.retriever.query_processor is not None,
            "reranker": self.retriever.reranker is not None,
            "book_structure": len(self.retriever.book_structure) > 0
        }
        
        try:
            for component, status in components.items():
                print(f"  {'✅' if status else '❌'} {component}: {'OK' if status else 'FAILED'}")
            
            passed = sum(1 for status in components.values() if status)
            total = len(components)
            
            self.test_results.append({
                "test": "system_components",
                "status": "passed" if passed == total else "partial",
                "message": f"Components working: {passed}/{total}"
            })
            print(f"✅ System components test: {passed}/{total} passed")
            
        except Exception as e:
            self.test_results.append({
                "test": "system_components",
                "status": "failed",
                "message": f"System components test failed: {e}"
            })
            print(f"❌ System components test failed: {e}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("🎯 TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for result in self.test_results if result["status"] == "passed")
        partial = sum(1 for result in self.test_results if result["status"] == "partial")
        failed = sum(1 for result in self.test_results if result["status"] == "failed")
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"⚠️  Partial: {partial}")
        print(f"❌ Failed: {failed}")
        print()
        
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "passed" else "⚠️" if result["status"] == "partial" else "❌"
            print(f"{status_icon} {result['test']}: {result['message']}")
        
        print()
        if failed == 0:
            print("🎉 ALL TESTS PASSED! Knowledge Base Retrieval System is working correctly.")
        elif failed <= 2:
            print("⚠️  MOSTLY WORKING with minor issues. System is usable.")
        else:
            print("❌ SIGNIFICANT ISSUES detected. System may need troubleshooting.")

async def main():
    """Run the comprehensive test suite"""
    print("🚀 Starting Advanced Knowledge Base Retrieval System Tests")
    print("="*60)
    
    tester = KnowledgeBaseRetrievalTester()
    
    # Setup
    if not await tester.setup():
        print("❌ Failed to initialize system. Exiting.")
        return
    
    # Run all tests
    await tester.test_system_components()
    await tester.test_query_understanding()
    await tester.test_knowledge_base_inventory()
    await tester.test_basic_search()
    await tester.test_intelligent_search()
    await tester.test_study_plan_generation()
    await tester.test_book_analysis()
    await tester.test_chapter_extraction()
    
    # Print summary
    tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main()) 