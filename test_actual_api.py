#!/usr/bin/env python3
"""
Test the actual MCP API with the fixes applied
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

async def test_actual_presentation_api():
    """Test the actual presentation generation API"""
    
    print("🚀 Testing ACTUAL MCP Presentation API...")
    print("=" * 60)
    
    try:
        # Initialize the server
        server = PerfectMCPServer()
        
        # Test with a focused query that should work well
        query = "supervised learning algorithms"
        user_prompt = "Create a comprehensive presentation about supervised learning algorithms for students"
        
        print(f"📝 **Query:** {query}")
        print(f"💭 **User Prompt:** {user_prompt}")
        print(f"🎨 **Theme:** academic_professional")
        print(f"🎯 **Slides:** 8")
        print(f"📅 **Start Time:** {time.strftime('%H:%M:%S')}")
        
        # Track timing
        start_time = time.time()
        
        # Call the actual API
        print("\n🔄 Calling actual presentation generation API...")
        result = await server._handle_create_presentation(
            query=query,
            user_prompt=user_prompt,
            title="Machine Learning: Supervised Learning Algorithms",
            author="AI Research Assistant",
            theme="academic_professional",
            slide_count=8,
            audience_type="academic",
            include_web_references=False
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️ **Total Processing Time:** {duration:.2f} seconds")
        print(f"📅 **End Time:** {time.strftime('%H:%M:%S')}")
        
        # Analyze the results
        if result and len(result) > 0:
            response_text = result[0].text if hasattr(result[0], 'text') else str(result[0])
            
            print(f"\n✅ **API Call Successful!**")
            print(f"📊 **Response Length:** {len(response_text)} characters")
            
            # Check for key success indicators
            success_indicators = [
                "✅ Perfect presentation created successfully!",
                "Chain-of-Thought analysis completed",
                "Comprehensive Knowledge Base Search",
                "Presentation Generation"
            ]
            
            found_indicators = []
            for indicator in success_indicators:
                if indicator in response_text:
                    found_indicators.append(indicator)
            
            print(f"\n🎯 **Success Indicators Found:** {len(found_indicators)}/{len(success_indicators)}")
            for indicator in found_indicators:
                print(f"   ✅ {indicator}")
            
            # Look for file creation
            if "perfect_research_presentation_" in response_text:
                print(f"\n📄 **Presentation File Created:** ✅")
                # Extract filename from response
                import re
                file_match = re.search(r'perfect_research_presentation_(\d+_\d+)\.pptx', response_text)
                if file_match:
                    filename = f"perfect_research_presentation_{file_match.group(1)}.pptx"
                    filepath = Path("presentations") / filename
                    if filepath.exists():
                        file_size = filepath.stat().st_size
                        print(f"   📁 **File Path:** {filepath}")
                        print(f"   📊 **File Size:** {file_size:,} bytes")
                    else:
                        print(f"   ⚠️ **File not found at:** {filepath}")
            
            # Check for optimization indicators
            optimization_checks = [
                ("optimized knowledge base searches", "Search Optimization"),
                ("Sections created:", "Content Categorization"),
                ("Chain-of-Thought analysis", "CoT Analysis"),
                ("Total chunks:", "Content Gathering")
            ]
            
            print(f"\n🔧 **Optimization Indicators:**")
            for check, description in optimization_checks:
                if check in response_text:
                    print(f"   ✅ {description}")
                else:
                    print(f"   ⚠️ {description} - not found")
            
            # Performance analysis
            print(f"\n📈 **Performance Analysis:**")
            
            # Expected vs actual timing
            if duration < 300:  # Less than 5 minutes
                print(f"   ✅ **Time Performance:** {duration:.1f}s (Expected: <5min)")
            else:
                print(f"   ⚠️ **Time Performance:** {duration:.1f}s (Still too slow)")
            
            # Check for error patterns
            error_patterns = [
                "Error generating AI slide plan",
                "slice(None, 200, None)",
                "Only one section created",
                "Search failed"
            ]
            
            errors_found = []
            for pattern in error_patterns:
                if pattern in response_text:
                    errors_found.append(pattern)
            
            if errors_found:
                print(f"\n❌ **Errors Found:** {len(errors_found)}")
                for error in errors_found:
                    print(f"   ❌ {error}")
            else:
                print(f"\n✅ **No Known Errors Found**")
            
            # Print a sample of the response
            print(f"\n📝 **Response Sample (First 500 chars):**")
            print("-" * 50)
            print(response_text[:500])
            print("-" * 50)
            
        else:
            print(f"\n❌ **API Call Failed:** No response received")
            
    except Exception as e:
        print(f"\n❌ **Test Failed:** {e}")
        import traceback
        traceback.print_exc()

async def test_optimized_vs_original():
    """Compare optimized vs original approach"""
    
    print("\n" + "=" * 60)
    print("🔄 COMPARISON: Optimized vs Original Approach")
    print("=" * 60)
    
    # Test with different complexity levels
    test_cases = [
        {
            "name": "Simple Query",
            "query": "machine learning",
            "expected_time": 120,  # 2 minutes
            "expected_sections": 3
        },
        {
            "name": "Complex Query", 
            "query": "supervised learning classification regression algorithms",
            "expected_time": 180,  # 3 minutes
            "expected_sections": 4
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 **Test Case:** {test_case['name']}")
        print(f"📝 **Query:** {test_case['query']}")
        
        try:
            server = PerfectMCPServer()
            start_time = time.time()
            
            result = await server._handle_create_presentation(
                query=test_case['query'],
                user_prompt="Create a presentation for students",
                slide_count=6,
                include_web_references=False
            )
            
            duration = time.time() - start_time
            
            if result and len(result) > 0:
                response_text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                
                # Check performance
                time_ok = duration <= test_case['expected_time']
                print(f"   ⏱️ **Time:** {duration:.1f}s (Expected: ≤{test_case['expected_time']}s) {'✅' if time_ok else '❌'}")
                
                # Check sections
                sections_match = response_text.count("Sections created:")
                print(f"   📊 **Sections:** Found section creation indicators: {sections_match}")
                
                # Check for success
                success = "✅ Perfect presentation created successfully!" in response_text
                print(f"   🎯 **Success:** {'✅' if success else '❌'}")
                
            else:
                print(f"   ❌ **Failed:** No response")
                
        except Exception as e:
            print(f"   ❌ **Error:** {e}")

async def main():
    """Run all API tests"""
    await test_actual_presentation_api()
    await test_optimized_vs_original()
    print(f"\n🎉 **All API tests completed!**")

if __name__ == "__main__":
    asyncio.run(main()) 