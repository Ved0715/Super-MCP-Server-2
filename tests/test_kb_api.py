#!/usr/bin/env python3
"""
🌐 Test Script for Knowledge Base API HTTP Endpoints
Tests the new intelligent KB retrieval API endpoints
"""

import requests
import json
import time
from typing import Dict, Any

class KnowledgeBaseAPITester:
    """Test the KB API HTTP endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:3001"):
        self.base_url = base_url
        self.kb_base_url = f"{base_url}/kb"
        self.test_results = []
        
    def test_kb_health(self):
        """Test KB API health endpoint"""
        print("🏥 Testing KB API Health...")
        
        try:
            response = requests.get(f"{self.kb_base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Status: {data.get('status', 'unknown')}")
                print(f"  📊 Service: {data.get('service', 'unknown')}")
                print(f"  🔧 Version: {data.get('version', 'unknown')}")
                print(f"  🤖 Retriever: {data.get('retriever_initialized', False)}")
                
                self.test_results.append({
                    "test": "kb_health",
                    "status": "passed",
                    "message": "KB API health check passed"
                })
                print("✅ KB API health test passed")
            else:
                print(f"  ❌ Status code: {response.status_code}")
                self.test_results.append({
                    "test": "kb_health",
                    "status": "failed",
                    "message": f"Health check failed with status {response.status_code}"
                })
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results.append({
                "test": "kb_health",
                "status": "failed",
                "message": f"Health check failed: {e}"
            })
    
    def test_kb_stats(self):
        """Test KB API stats endpoint"""
        print("\n📊 Testing KB API Stats...")
        
        try:
            response = requests.get(f"{self.kb_base_url}/stats")
            
            if response.status_code == 200:
                data = response.json()
                stats = data.get('stats', {})
                
                print(f"  📚 Total books: {stats.get('total_books', 0)}")
                print(f"  📄 Total chunks: {stats.get('total_chunks', 0)}")
                print(f"  🏷️ Namespace: {stats.get('namespace', 'unknown')}")
                print(f"  📇 Index: {stats.get('index_name', 'unknown')}")
                
                self.test_results.append({
                    "test": "kb_stats",
                    "status": "passed",
                    "message": f"Found {stats.get('total_books', 0)} books with {stats.get('total_chunks', 0)} chunks"
                })
                print("✅ KB API stats test passed")
            else:
                print(f"  ❌ Status code: {response.status_code}")
                self.test_results.append({
                    "test": "kb_stats",
                    "status": "failed",
                    "message": f"Stats failed with status {response.status_code}"
                })
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results.append({
                "test": "kb_stats",
                "status": "failed",
                "message": f"Stats failed: {e}"
            })
    
    def test_kb_books(self):
        """Test KB API books endpoint"""
        print("\n📚 Testing KB API Books...")
        
        try:
            response = requests.get(f"{self.kb_base_url}/books")
            
            if response.status_code == 200:
                data = response.json()
                books = data.get('books', [])
                book_details = data.get('book_details', {})
                
                print(f"  📖 Total books: {len(books)}")
                if books:
                    print("  📚 Available books:")
                    for book in books:
                        chapters = book_details.get(book, {}).get('chapters', [])
                        print(f"    - {book}: {len(chapters)} chapters")
                
                self.test_results.append({
                    "test": "kb_books",
                    "status": "passed",
                    "message": f"Found {len(books)} books"
                })
                print("✅ KB API books test passed")
            else:
                print(f"  ❌ Status code: {response.status_code}")
                self.test_results.append({
                    "test": "kb_books",
                    "status": "failed",
                    "message": f"Books failed with status {response.status_code}"
                })
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results.append({
                "test": "kb_books",
                "status": "failed",
                "message": f"Books failed: {e}"
            })
    
    def test_intelligent_query(self):
        """Test the main intelligent query endpoint"""
        print("\n🧠 Testing Intelligent Query Endpoint...")
        
        test_queries = [
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
            for test_case in test_queries:
                query = test_case["query"]
                expected_type = test_case["expected_type"]
                
                print(f"  🔍 Testing: '{query}'")
                
                response = requests.post(
                    f"{self.kb_base_url}/query",
                    json={"query": query},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_type = data.get("response_type", "unknown")
                    execution_time = data.get("execution_time", 0)
                    
                    print(f"    ✅ Response type: {response_type}")
                    print(f"    ⏱️ Execution time: {execution_time:.3f}s")
                    
                    if response_type == expected_type:
                        print(f"    🎯 Expected type matched!")
                    else:
                        print(f"    ⚠️ Expected {expected_type}, got {response_type}")
                        
                else:
                    print(f"    ❌ Status code: {response.status_code}")
                    if response.content:
                        print(f"    📝 Response: {response.text[:200]}...")
                
            self.test_results.append({
                "test": "intelligent_query",
                "status": "passed",
                "message": "Intelligent query endpoint working"
            })
            print("✅ Intelligent query tests passed")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results.append({
                "test": "intelligent_query",
                "status": "failed",
                "message": f"Intelligent query failed: {e}"
            })
    
    def test_advanced_search(self):
        """Test the advanced search endpoint"""
        print("\n🔍 Testing Advanced Search Endpoint...")
        
        try:
            response = requests.post(
                f"{self.kb_base_url}/advanced-search",
                params={
                    "query": "machine learning algorithms",
                    "search_type": "hybrid",
                    "top_k": 5
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                execution_time = data.get("execution_time", 0)
                
                print(f"  📊 Results: {len(results)}")
                print(f"  ⏱️ Execution time: {execution_time:.3f}s")
                print(f"  🔍 Search type: {data.get('search_type', 'unknown')}")
                
                self.test_results.append({
                    "test": "advanced_search",
                    "status": "passed",
                    "message": f"Advanced search returned {len(results)} results"
                })
                print("✅ Advanced search test passed")
            else:
                print(f"  ❌ Status code: {response.status_code}")
                self.test_results.append({
                    "test": "advanced_search",
                    "status": "failed",
                    "message": f"Advanced search failed with status {response.status_code}"
                })
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results.append({
                "test": "advanced_search",
                "status": "failed",
                "message": f"Advanced search failed: {e}"
            })
    
    def test_examples(self):
        """Test the examples endpoint"""
        print("\n📝 Testing Examples Endpoint...")
        
        try:
            response = requests.get(f"{self.kb_base_url}/examples")
            
            if response.status_code == 200:
                data = response.json()
                examples = data.get("examples", {})
                
                print(f"  📖 Search queries: {len(examples.get('search_queries', []))}")
                print(f"  📚 Study plan queries: {len(examples.get('study_plan_queries', []))}")
                print(f"  📚 Book analysis queries: {len(examples.get('book_analysis_queries', []))}")
                print(f"  📑 Chapter queries: {len(examples.get('chapter_queries', []))}")
                
                self.test_results.append({
                    "test": "examples",
                    "status": "passed",
                    "message": "Examples endpoint working"
                })
                print("✅ Examples test passed")
            else:
                print(f"  ❌ Status code: {response.status_code}")
                self.test_results.append({
                    "test": "examples",
                    "status": "failed",
                    "message": f"Examples failed with status {response.status_code}"
                })
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results.append({
                "test": "examples",
                "status": "failed",
                "message": f"Examples failed: {e}"
            })
    
    def test_diagnostics(self):
        """Test the diagnostics endpoint"""
        print("\n🔧 Testing Diagnostics Endpoint...")
        
        try:
            response = requests.get(f"{self.kb_base_url}/diagnostics")
            
            if response.status_code == 200:
                data = response.json()
                diagnostics = data.get("diagnostics", {})
                
                print("  🔍 System Components:")
                for component, status in diagnostics.items():
                    if isinstance(status, bool):
                        print(f"    {'✅' if status else '❌'} {component}: {'OK' if status else 'FAILED'}")
                    else:
                        print(f"    ℹ️  {component}: {status}")
                
                self.test_results.append({
                    "test": "diagnostics",
                    "status": "passed",
                    "message": "Diagnostics endpoint working"
                })
                print("✅ Diagnostics test passed")
            else:
                print(f"  ❌ Status code: {response.status_code}")
                self.test_results.append({
                    "test": "diagnostics",
                    "status": "failed",
                    "message": f"Diagnostics failed with status {response.status_code}"
                })
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results.append({
                "test": "diagnostics",
                "status": "failed",
                "message": f"Diagnostics failed: {e}"
            })
    
    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Knowledge Base API Tests")
        print("="*60)
        
        # Test all endpoints
        self.test_kb_health()
        self.test_kb_stats()
        self.test_kb_books()
        self.test_intelligent_query()
        self.test_advanced_search()
        self.test_examples()
        self.test_diagnostics()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("🎯 API TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for result in self.test_results if result["status"] == "passed")
        failed = sum(1 for result in self.test_results if result["status"] == "failed")
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print()
        
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "passed" else "❌"
            print(f"{status_icon} {result['test']}: {result['message']}")
        
        print()
        if failed == 0:
            print("🎉 ALL API TESTS PASSED! Knowledge Base API is working correctly.")
        elif failed <= 2:
            print("⚠️  MOSTLY WORKING with minor issues. API is usable.")
        else:
            print("❌ SIGNIFICANT ISSUES detected. API may need troubleshooting.")

if __name__ == "__main__":
    tester = KnowledgeBaseAPITester()
    tester.run_all_tests() 