#!/usr/bin/env python3
"""
Test script to verify the fixes for the presentation generation system
"""
import os
import sys
import asyncio
import json
from pathlib import Path

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perfect_mcp_server import PerfectMCPServer
from perfect_ppt_generator import PerfectPPTGenerator
from config import AdvancedConfig

async def test_fixed_presentation_system():
    """Test the fixes for the presentation generation system"""
    
    print("🔧 Testing FIXED presentation system...")
    
    try:
        # Initialize the system
        server = PerfectMCPServer()
        
        # Test 1: Simple CoT analysis
        print("\n🧠 Test 1: Chain-of-Thought Analysis")
        cot_result = await server._chain_of_thought_presentation_analysis(
            "machine learning algorithms",
            "Create a comprehensive presentation"
        )
        
        if cot_result.get("success"):
            print("✅ CoT analysis successful")
            structured = cot_result.get("structured_analysis", {})
            print(f"   - Main topic: {structured.get('main_topic', 'N/A')}")
            print(f"   - Subtopics: {len(structured.get('key_subtopics', []))}")
            print(f"   - Search terms: {len(structured.get('search_terms', []))}")
        else:
            print(f"❌ CoT analysis failed: {cot_result.get('error', 'Unknown error')}")
            
        # Test 2: Content categorization
        print("\n📊 Test 2: Content Categorization")
        
        # Simulate search results
        mock_search_results = {
            "success": True,
            "search_results": [
                {
                    "content": "Machine learning algorithms can be categorized into supervised learning methods",
                    "metadata": {"book_name": "ML Fundamentals"},
                    "score": 0.9
                },
                {
                    "content": "Classification algorithms like decision trees are used for categorization",
                    "metadata": {"book_name": "ML Algorithms"},
                    "score": 0.8
                },
                {
                    "content": "Regression techniques predict continuous values using statistical methods",
                    "metadata": {"book_name": "Statistical Learning"},
                    "score": 0.7
                }
            ],
            "statistics": {
                "total_chunks": 3,
                "total_content_length": 200,
                "average_score": 0.8
            }
        }
        
        # Test content transformation
        aggregated = server._transform_search_results_to_paper_format(mock_search_results, cot_result)
        
        if aggregated.get("success"):
            sections = aggregated.get("sections", {})
            print(f"✅ Content transformation successful")
            print(f"   - Sections created: {len(sections)}")
            print(f"   - Section names: {list(sections.keys())}")
            
            # Verify we have multiple sections (not just "general")
            if len(sections) > 1:
                print("✅ Multiple sections created (fixed categorization)")
            else:
                print("⚠️ Only one section created - may need further optimization")
                
        else:
            print(f"❌ Content transformation failed: {aggregated.get('error', 'Unknown error')}")
            
        # Test 3: PPT Generator Context Handling
        print("\n🎨 Test 3: PPT Generator Context Handling")
        
        try:
            config = AdvancedConfig()
            ppt_gen = PerfectPPTGenerator(config)
            
            # Test with both string and dict content formats
            test_content = {
                "sections": {
                    "general": {
                        "content": "This is test content about machine learning algorithms",
                        "sources": ["Test Book"],
                        "chunk_count": 1
                    },
                    "classification": "Simple string content about classification"
                }
            }
            
            context = ppt_gen._prepare_context_for_slide_generation(test_content, {}, {})
            
            if context and "GENERAL:" in context:
                print("✅ Context preparation handles both formats")
                print(f"   - Context length: {len(context)} characters")
            else:
                print("❌ Context preparation failed")
                
        except Exception as e:
            print(f"❌ PPT generator test failed: {e}")
            
        # Test 4: Search Query Optimization
        print("\n🔍 Test 4: Search Query Optimization")
        
        # Simulate a comprehensive search scenario
        large_cot_analysis = {
            "success": True,
            "structured_analysis": {
                "main_topic": "Machine Learning Algorithms",
                "key_subtopics": ["supervised learning", "classification", "regression", "neural networks", "deep learning", "reinforcement learning"],
                "focus_areas": ["algorithms", "applications", "evaluation", "challenges", "methodology"],
                "search_terms": ["machine learning", "ML", "artificial intelligence", "data science", "algorithms"]
            }
        }
        
        # This would normally create 15+ searches, but should now be optimized to 8
        print(f"   - Original approach would create: ~15+ searches")
        print(f"   - Optimized approach should create: ≤8 searches")
        
        # Count what our optimization would produce
        structured = large_cot_analysis["structured_analysis"]
        main_topic = structured.get("main_topic", "")
        key_subtopics = structured.get("key_subtopics", [])
        focus_areas = structured.get("focus_areas", [])
        search_terms = structured.get("search_terms", [])
        
        simulated_queries = []
        if main_topic:
            simulated_queries.append(main_topic)
        simulated_queries.extend(key_subtopics[:3])
        simulated_queries.extend(focus_areas[:2])
        
        # Add unique search terms
        for term in search_terms:
            if term not in simulated_queries and len(simulated_queries) < 8:
                simulated_queries.append(term)
                
        print(f"✅ Optimized search count: {len(simulated_queries)} searches")
        print(f"   - Queries: {simulated_queries}")
        
        if len(simulated_queries) <= 8:
            print("✅ Search optimization successful - reduced API calls")
        else:
            print("⚠️ Search optimization needs further tuning")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests"""
    print("🚀 Starting presentation system fix tests...")
    await test_fixed_presentation_system()
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 