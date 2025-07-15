#!/usr/bin/env python3
"""
Simple Interactive Test Script for Knowledge Base System
Run this to test your MCP server with OpenAI-powered responses
"""

import requests
import json
import sys
import time

class KnowledgeBaseTestClient:
    def __init__(self, base_url="http://localhost:3001"):
        self.base_url = base_url
        
    def test_health(self):
        """Test if the MCP server is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print("✅ MCP Server is healthy!")
                print(f"   Version: {health_data.get('version', 'Unknown')}")
                print(f"   Uptime: {health_data.get('uptime', 0):.1f} seconds")
                print(f"   Tools: {health_data.get('tools_count', 0)}")
                print(f"   Memory: {health_data.get('memory_usage', 'Unknown')}")
                return True
            else:
                print("❌ MCP Server health check failed")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to MCP server: {e}")
            return False
    
    def search_knowledge_base(self, query):
        """Search the knowledge base with OpenAI-powered responses"""
        try:
            print(f"🧠 Searching knowledge base with AI-powered response generation...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/mcp/call",
                json={
                    "tool": "search_knowledge_base",
                    "arguments": {"query": query}
                }
            )
            
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    content = json.loads(result['result']['content'])
                    
                    # Check if we got an AI-generated response
                    if content.get('search_type') == 'enhanced' and 'ai_response' in content:
                        print(f"🎯 Query: '{query}'")
                        print(f"⚡ Response time: {execution_time:.2f}s")
                        print(f"📊 Sources used: {content.get('total_sources', 'Unknown')}")
                        print(f"🧠 AI-Generated Response:\n")
                        print("=" * 80)
                        print(content['ai_response'])
                        print("=" * 80)
                        print(f"\n✨ Enhanced with OpenAI {content.get('search_type', 'enhanced')} search")
                        
                    else:
                        # Handle other response types or fallbacks
                        search_type = content.get('search_type', 'basic')
                        print(f"🔍 Search Results for: '{query}' ({search_type})")
                        print(f"⚡ Response time: {execution_time:.2f}s")
                        print(f"📚 Results:")
                        
                        if 'results' in content:
                            results_text = content['results']
                            if isinstance(results_text, str):
                                # Display first part of results
                                print(results_text[:1000] + "..." if len(results_text) > 1000 else results_text)
                            else:
                                print(json.dumps(results_text, indent=2)[:1000] + "...")
                    
                    return True
                else:
                    print(f"❌ Search failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Request failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Search error: {e}")
            return False
    
    def get_knowledge_inventory(self):
        """Get knowledge base inventory"""
        try:
            response = requests.post(
                f"{self.base_url}/mcp/call",
                json={
                    "tool": "get_knowledge_base_inventory",
                    "arguments": {}
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    content = json.loads(result['result']['content'])
                    inventory = content.get('inventory', {})
                    
                    print("📚 Knowledge Base Inventory:")
                    print(f"   Total Chunks: {inventory.get('total_chunks', 0)}")
                    print(f"   Books Available: {len(inventory.get('books', []))}")
                    
                    books_structure = inventory.get('books_structure', {})
                    for book_name, book_info in books_structure.items():
                        print(f"\n   📖 {book_name}")
                        print(f"      Chapters: {book_info.get('chunk_count', 0)}")
                        print(f"      Words: {book_info.get('total_words', 0):,.0f}")
                        print(f"      Mathematical Content: {book_info.get('mathematical_content', 0)}")
                        
                        chunk_types = book_info.get('chunk_types', {})
                        if chunk_types:
                            types_str = ", ".join([f"{k}: {v}" for k, v in chunk_types.items()])
                            print(f"      Types: {types_str}")
                    
                    return True
                else:
                    print(f"❌ Inventory failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Request failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Inventory error: {e}")
            return False
    
    def find_books_covering_topic(self, topic):
        """Find books covering a specific topic"""
        try:
            response = requests.post(
                f"{self.base_url}/mcp/call",
                json={
                    "tool": "find_books_covering_topic",
                    "arguments": {"topic": topic}
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    content = json.loads(result['result']['content'])
                    
                    print(f"📖 Books covering '{topic}':")
                    books = content.get('books', [])
                    if books:
                        for book in books:
                            print(f"   • {book.get('book_name', 'Unknown')}")
                            print(f"     Relevance: {book.get('relevance_score', 0):.2f}")
                            print(f"     Chunks: {book.get('chunk_count', 0)}")
                    else:
                        print("   No books found for this topic")
                    
                    return True
                else:
                    print(f"❌ Search failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Request failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Search error: {e}")
            return False

def main():
    """Interactive test interface"""
    print("🧠 Knowledge Base System Tester")
    print("=" * 50)
    
    client = KnowledgeBaseTestClient()
    
    # Test health first
    if not client.test_health():
        print("\n❌ Cannot connect to MCP server. Make sure it's running:")
        print("   python start_mcp_server.py --host localhost --port 3001")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. Search knowledge base")
        print("2. View knowledge inventory")
        print("3. Find books covering topic")
        print("4. Test health again")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            query = input("\n🔍 Enter your search query: ").strip()
            if query:
                print()
                client.search_knowledge_base(query)
            else:
                print("❌ Please enter a valid query")
                
        elif choice == "2":
            print()
            client.get_knowledge_inventory()
            
        elif choice == "3":
            topic = input("\n📖 Enter topic to find books for: ").strip()
            if topic:
                print()
                client.find_books_covering_topic(topic)
            else:
                print("❌ Please enter a valid topic")
                
        elif choice == "4":
            print()
            client.test_health()
            
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main() 