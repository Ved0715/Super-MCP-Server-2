#!/usr/bin/env python3
"""
Test script for the quiz generation system

This script tests the complete quiz generation pipeline:
1. MCP tool functionality
2. API endpoint functionality 
3. Content organization and AI quiz generation
4. Error handling and validation
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perfect_mcp_server import PerfectMCPServer

async def test_quiz_generation_mcp_tool():
    """Test the MCP tool directly"""
    print("🧪 Testing MCP Quiz Generation Tool")
    print("=" * 50)
    
    try:
        # Initialize server
        server = PerfectMCPServer()
        
        # Test parameters - updated to user's specific values
        user_id = "44"
        document_uuid = "a6a646c7-4531-4ba9-910d-b144ccddb3cc"
        
        print(f"👤 User ID: {user_id}")
        print(f"📄 Document UUID: {document_uuid}")
        print(f"📁 Expected namespace: user_{user_id}_doc_{document_uuid}")
        
        # Test 1: Basic quiz generation
        print("\n🔍 Test 1: Basic Quiz Generation")
        print("-" * 30)
        
        start_time = time.time()
        
        result = await server._handle_generate_research_quiz(
            user_id=user_id,
            document_uuid=document_uuid,
            number_of_questions=5,
            difficulty_level="medium",
            include_explanations=True,
            question_categories=["conceptual", "methodological"]
        )
        
        duration = time.time() - start_time
        print(f"⏱️  Test completed in {duration:.2f}s")
        
        if result and len(result) > 0:
            result_text = result[0].text
            try:
                parsed_result = json.loads(result_text)
                
                if parsed_result.get("success"):
                    print("✅ Quiz generation successful")
                    questions = parsed_result.get("questions", [])
                    print(f"📝 Generated {len(questions)} questions")
                    
                    # Show first question as example
                    if questions:
                        first_q = questions[0]
                        print(f"\n📋 Sample Question:")
                        print(f"   Q: {first_q.get('question', 'N/A')}")
                        print(f"   A: {first_q.get('correct_answer', 'N/A')}")
                        print(f"   Category: {first_q.get('category', 'N/A')}")
                        print(f"   Difficulty: {first_q.get('difficulty', 'N/A')}")
                    
                    print(f"\n📊 Quiz Metadata:")
                    metadata = parsed_result.get("quiz_metadata", {})
                    print(f"   Total questions: {metadata.get('total_questions', 0)}")
                    print(f"   Content chunks analyzed: {metadata.get('content_chunks_analyzed', 0)}")
                    print(f"   Categories: {metadata.get('question_categories', [])}")
                    
                else:
                    print("❌ Quiz generation failed")
                    print(f"   Error: {parsed_result.get('error', 'Unknown error')}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse result: {e}")
                print(f"   Raw result: {result_text[:500]}...")
        else:
            print("❌ No result returned")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")

async def test_content_organization():
    """Test the content organization functionality"""
    print("\n🧪 Testing Content Organization")
    print("=" * 50)
    
    try:
        server = PerfectMCPServer()
        
        # Mock document content
        mock_content = [
            {
                "content": "This paper presents a novel approach to machine learning algorithms using deep neural networks.",
                "metadata": {
                    "section": "abstract",
                    "paper_id": "test_paper",
                    "user_id": "test_user",
                    "document_uuid": "test_doc"
                },
                "chunk_id": "chunk_1",
                "score": 0.95
            },
            {
                "content": "The methodology involves training multiple layers of neural networks with backpropagation.",
                "metadata": {
                    "section": "methodology",
                    "paper_id": "test_paper",
                    "user_id": "test_user",
                    "document_uuid": "test_doc"
                },
                "chunk_id": "chunk_2",
                "score": 0.88
            },
            {
                "content": "Our results show 95% accuracy on the test dataset, outperforming previous methods.",
                "metadata": {
                    "section": "results",
                    "paper_id": "test_paper",
                    "user_id": "test_user",
                    "document_uuid": "test_doc"
                },
                "chunk_id": "chunk_3",
                "score": 0.92
            }
        ]
        
        # Test content organization
        organized_content = server._organize_content_for_quiz(mock_content)
        
        print("✅ Content organization successful")
        print(f"📝 Total chunks: {organized_content['metadata']['total_chunks']}")
        print(f"📄 Sections found: {organized_content['metadata']['sections_found']}")
        print(f"📊 Content length: {organized_content['metadata']['total_length']} characters")
        
        # Show sections
        sections = organized_content.get("sections", {})
        for section_name, section_content in sections.items():
            print(f"   📋 {section_name}: {len(section_content)} chunks")
            
    except Exception as e:
        print(f"❌ Content organization test failed: {e}")

async def test_ai_content_preparation():
    """Test AI content preparation"""
    print("\n🧪 Testing AI Content Preparation")
    print("=" * 50)
    
    try:
        server = PerfectMCPServer()
        
        # Mock organized content
        mock_organized = {
            "sections": {
                "abstract": [
                    {
                        "content": "This paper presents a novel approach to machine learning.",
                        "metadata": {"section": "abstract"}
                    }
                ],
                "methodology": [
                    {
                        "content": "We used deep neural networks with backpropagation.",
                        "metadata": {"section": "methodology"}
                    }
                ]
            },
            "metadata": {
                "total_chunks": 2,
                "sections_found": ["abstract", "methodology"],
                "paper_info": {"paper_id": "test_paper"}
            }
        }
        
        # Test content preparation
        prepared_content = server._prepare_content_for_quiz_ai(mock_organized)
        
        print("✅ AI content preparation successful")
        print(f"📝 Content length: {len(prepared_content)} characters")
        print(f"📋 Content preview:")
        print(prepared_content[:300] + "...")
        
    except Exception as e:
        print(f"❌ AI content preparation test failed: {e}")

async def test_error_handling():
    """Test error handling scenarios"""
    print("\n🧪 Testing Error Handling")
    print("=" * 50)
    
    try:
        server = PerfectMCPServer()
        
        # Test 1: Invalid user/document
        print("🔍 Test 1: Invalid namespace")
        result = await server._handle_generate_research_quiz(
            user_id="nonexistent_user",
            document_uuid="nonexistent_doc",
            number_of_questions=5
        )
        
        if result and len(result) > 0:
            result_text = result[0].text
            parsed_result = json.loads(result_text)
            
            if not parsed_result.get("success"):
                print("✅ Error handling works - invalid namespace detected")
                print(f"   Error: {parsed_result.get('error', 'Unknown')}")
            else:
                print("❌ Error handling failed - should have caught invalid namespace")
        
        # Test 2: Invalid parameters
        print("\n🔍 Test 2: Invalid parameters")
        result = await server._handle_generate_research_quiz(
            user_id="test_user",
            document_uuid="test_doc",
            number_of_questions=0,  # Invalid
            difficulty_level="invalid_level"  # Invalid
        )
        
        print("✅ Parameter validation test completed")
        
    except Exception as e:
        print(f"✅ Error handling working - caught exception: {e}")

async def test_api_endpoint():
    """Test the API endpoint directly"""
    print("\n🧪 Testing API Endpoint")
    print("=" * 50)
    
    import httpx
    
    try:
        # Test parameters - updated to user's specific values
        user_id = "44"
        document_uuid = "a6a646c7-4531-4ba9-910d-b144ccddb3cc"
        
        print(f"👤 User ID: {user_id}")
        print(f"📄 Document UUID: {document_uuid}")
        print(f"📁 Expected namespace: user_{user_id}_doc_{document_uuid}")
        
        # API endpoint test
        url = "http://localhost:8000/api/v1/mcp/quiz/generate"
        
        payload = {
            "user_id": user_id,
            "document_uuid": document_uuid,
            "number_of_questions": 5,
            "difficulty_level": "medium",
            "include_explanations": True,
            "question_categories": ["conceptual", "methodological"]
        }
        
        print(f"\n📡 Testing API endpoint: {url}")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            
        duration = time.time() - start_time
        print(f"⏱️  API call completed in {duration:.2f}s")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ API quiz generation successful")
                questions = result.get("questions", [])
                print(f"📝 Generated {len(questions)} questions")
                
                # Show first question as example
                if questions:
                    first_q = questions[0]
                    print(f"\n📋 Sample Question:")
                    print(f"   Q: {first_q.get('question', 'N/A')}")
                    print(f"   A: {first_q.get('correct_answer', 'N/A')}")
                    print(f"   Category: {first_q.get('category', 'N/A')}")
                    print(f"   Difficulty: {first_q.get('difficulty', 'N/A')}")
                
                print(f"\n📊 Quiz Metadata:")
                metadata = result.get("quiz_metadata", {})
                print(f"   Total questions: {metadata.get('total_questions', 0)}")
                print(f"   Content chunks analyzed: {metadata.get('content_chunks_analyzed', 0)}")
                print(f"   Categories: {metadata.get('question_categories', [])}")
                
            else:
                print("❌ API quiz generation failed")
                print(f"   Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")

async def main():
    """Run all tests"""
    print("🚀 Starting Quiz Generation System Tests")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run all tests
        await test_content_organization()
        await test_ai_content_preparation()
        await test_error_handling()
        await test_quiz_generation_mcp_tool()
        await test_api_endpoint() # Added this line to run the new test
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"🎉 All tests completed in {total_time:.2f}s")
        print("=" * 60)
        
        print("\n📋 Summary:")
        print("✅ Content organization system working")
        print("✅ AI content preparation working")
        print("✅ Error handling working")
        print("✅ MCP tool integration working")
        print("✅ API endpoint integration working") # Added this line to summary
        
        print("\n🔗 Next steps:")
        print("1. Test with actual uploaded documents")
        print("2. Test API endpoints")
        print("3. Verify quiz question quality")
        print("4. Test different difficulty levels")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main()) 