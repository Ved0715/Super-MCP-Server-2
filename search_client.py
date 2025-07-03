"""
SerpAPI Search Client
Provides Google search functionality using SerpAPI
"""

import logging
import requests
from typing import List, Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)

class SerpAPIClient:
    """Client for Google search using SerpAPI"""
    
    def __init__(self, config: Config = None):
        """Initialize SerpAPI client"""
        self.config = config or Config()
        self.api_key = self.config.SERPAPI_KEY
        self.base_url = "https://serpapi.com/search"
        
    def search_google(self, 
                      query: str, 
                      num_results: int = None,
                      location: str = None,
                      search_type: str = "search") -> Dict[str, Any]:
        """
        Perform Google search using SerpAPI
        
        Args:
            query: Search query string
            num_results: Number of results to return (default from config)
            location: Search location (default from config)
            search_type: Type of search ('search', 'news', 'scholar')
            
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            # Set default values
            num_results = num_results or self.config.MAX_SEARCH_RESULTS
            location = location or self.config.SEARCH_LOCATION
            
            # Build search parameters
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "num": num_results,
                "location": location,
                "hl": "en",
                "gl": "us"
            }
            
            # Add search type specific parameters
            if search_type == "news":
                params["tbm"] = "nws"
            elif search_type == "scholar":
                params["engine"] = "google_scholar"
            
            # Make API request
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse results
            results = self._parse_search_results(data, search_type)
            
            return {
                "success": True,
                "query": query,
                "num_results": len(results),
                "results": results,
                "search_type": search_type,
                "location": location,
                "raw_data": data  # Include raw data for debugging
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"SerpAPI request failed: {e}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "query": query,
                "results": []
            }
        except Exception as e:
            logger.error(f"SerpAPI search error: {e}")
            return {
                "success": False,
                "error": f"Search error: {str(e)}",
                "query": query,
                "results": []
            }
    
    def _parse_search_results(self, data: Dict[str, Any], search_type: str) -> List[Dict[str, Any]]:
        """Parse SerpAPI response into structured results"""
        results = []
        
        try:
            if search_type == "scholar":
                # Google Scholar results
                organic_results = data.get("organic_results", [])
                for result in organic_results:
                    parsed_result = {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "publication_info": result.get("publication_info", {}),
                        "authors": result.get("publication_info", {}).get("authors", []),
                        "year": result.get("publication_info", {}).get("year", ""),
                        "cited_by": result.get("inline_links", {}).get("cited_by", {}).get("total", 0),
                        "type": "scholar"
                    }
                    results.append(parsed_result)
                    
            elif search_type == "news":
                # Google News results
                news_results = data.get("news_results", [])
                for result in news_results:
                    parsed_result = {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "source": result.get("source", ""),
                        "date": result.get("date", ""),
                        "thumbnail": result.get("thumbnail", ""),
                        "type": "news"
                    }
                    results.append(parsed_result)
                    
            else:
                # Regular Google search results
                organic_results = data.get("organic_results", [])
                for result in organic_results:
                    parsed_result = {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "displayed_link": result.get("displayed_link", ""),
                        "position": result.get("position", 0),
                        "type": "web"
                    }
                    results.append(parsed_result)
                
                # Add featured snippet if available
                featured_snippet = data.get("answer_box")
                if featured_snippet:
                    snippet_result = {
                        "title": featured_snippet.get("title", "Featured Snippet"),
                        "link": featured_snippet.get("link", ""),
                        "snippet": featured_snippet.get("snippet", ""),
                        "source": featured_snippet.get("source", ""),
                        "type": "featured_snippet",
                        "position": 0
                    }
                    results.insert(0, snippet_result)
                    
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            
        return results
    
    def search_for_research_papers(self, topic: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Search for research papers on a specific topic using Google Scholar
        
        Args:
            topic: Research topic to search for
            num_results: Number of papers to return
            
        Returns:
            Dictionary containing research paper results
        """
        # Enhance query for academic content
        academic_query = f"{topic} filetype:pdf academic research paper"
        
        return self.search_google(
            query=academic_query,
            num_results=num_results,
            search_type="scholar"
        )
    
    def search_recent_news(self, topic: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Search for recent news articles on a topic
        
        Args:
            topic: Topic to search news for
            num_results: Number of news articles to return
            
        Returns:
            Dictionary containing news results
        """
        return self.search_google(
            query=topic,
            num_results=num_results,
            search_type="news"
        )
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """
        Get search suggestions for a partial query
        This is a simple implementation - SerpAPI doesn't provide suggestions directly
        """
        suggestions = [
            f"{partial_query} research",
            f"{partial_query} papers",
            f"{partial_query} study",
            f"{partial_query} analysis",
            f"{partial_query} methodology"
        ]
        return suggestions[:5]
    
    def format_search_results(self, search_response: Dict[str, Any]) -> str:
        """
        Format search results into a readable string
        
        Args:
            search_response: Response from search_google method
            
        Returns:
            Formatted string of search results
        """
        if not search_response.get("success"):
            return f"Search failed: {search_response.get('error', 'Unknown error')}"
        
        results = search_response.get("results", [])
        if not results:
            return f"No results found for query: {search_response.get('query')}"
        
        formatted_results = []
        formatted_results.append(f"Search Results for: '{search_response.get('query')}'")
        formatted_results.append(f"Found {len(results)} results\n")
        
        for i, result in enumerate(results, 1):
            formatted_results.append(f"{i}. **{result.get('title', 'No Title')}**")
            formatted_results.append(f"   URL: {result.get('link', 'No URL')}")
            formatted_results.append(f"   {result.get('snippet', 'No description available')}")
            
            # Add type-specific information
            if result.get('type') == 'scholar':
                if result.get('authors'):
                    formatted_results.append(f"   Authors: {', '.join(result['authors'])}")
                if result.get('year'):
                    formatted_results.append(f"   Year: {result['year']}")
                if result.get('cited_by'):
                    formatted_results.append(f"   Cited by: {result['cited_by']}")
            elif result.get('type') == 'news':
                if result.get('source'):
                    formatted_results.append(f"   Source: {result['source']}")
                if result.get('date'):
                    formatted_results.append(f"   Date: {result['date']}")
            
            formatted_results.append("")  # Empty line between results
        
        return "\n".join(formatted_results) 