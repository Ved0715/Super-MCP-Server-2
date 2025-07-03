#!/usr/bin/env python3
"""
🚀 Perfect Research PPT System - Setup & Validation
Validates advanced configuration and launches the complete system
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required packages are installed"""
    # Package name to import name mapping
    package_import_map = {
        'python-pptx': 'pptx',
        'google-search-results': 'serpapi', 
        'python-dotenv': 'dotenv',
        'scikit-learn': 'sklearn',
        'beautifulsoup4': 'bs4',
        'llama-parse': 'llama_parse'
    }
    
    # Core packages (package_name: import_name)
    core_packages = [
        'streamlit', 'python-pptx', 'pypdf', 'google-search-results',
        'requests', 'mcp', 'pydantic', 'python-dotenv', 'openai'
    ]
    
    # Advanced packages for perfect system
    advanced_packages = [
        'pinecone', 'numpy', 'scikit-learn', 'nltk', 'textstat',
        'matplotlib', 'plotly', 'beautifulsoup4', 'lxml', 'pandas'
    ]
    
    # Optional packages
    optional_packages = [
        'llama-parse', 'spacy', 'wordcloud', 'seaborn'
    ]
    
    def check_package(package_name):
        """Check if a package can be imported"""
        import_name = package_import_map.get(package_name, package_name.replace('-', '_'))
        try:
            __import__(import_name)
            return True
        except ImportError:
            return False
    
    missing_core = [pkg for pkg in core_packages if not check_package(pkg)]
    missing_advanced = [pkg for pkg in advanced_packages if not check_package(pkg)]
    missing_optional = [pkg for pkg in optional_packages if not check_package(pkg)]
    
    return missing_core, missing_advanced, missing_optional

def check_environment():
    """Check if required environment variables are set"""
    # Core required variables
    required_vars = ['OPENAI_API_KEY', 'SERPAPI_KEY', 'PINECONE_API_KEY']
    
    # Optional but recommended
    optional_vars = ['LLAMA_PARSE_API_KEY', 'UNSPLASH_ACCESS_KEY']
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    return missing_required, missing_optional

def validate_config():
    """Validate advanced configuration"""
    try:
        from config import AdvancedConfig
        config = AdvancedConfig()
        issues = config.validate_config()
        
        critical_issues = [issue for issue in issues if issue.startswith("ERROR")]
        warnings = [issue for issue in issues if issue.startswith("WARNING")]
        
        return critical_issues, warnings
        
    except Exception as e:
        return [f"ERROR: Advanced configuration validation failed: {e}"], []

def setup_directories():
    """Create necessary directories for perfect system"""
    directories = [
        'presentations',  # PPT output
        'logs',          # System logs
        'cache',         # Caching
        'temp',          # Temporary files
        'exports'        # Export outputs
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_streamlit():
    """Run the Streamlit application"""
    try:
        logger.info("Starting Streamlit application...")
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        return True
    return True

def run_mcp_server():
    """Run the basic MCP server"""
    try:
        logger.info("Starting basic MCP server...")
        subprocess.run([sys.executable, 'mcp_server.py'], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start MCP server: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("MCP server stopped by user")
        return True
    return True

def run_perfect_mcp_server():
    """Run the perfect MCP server with all advanced features"""
    try:
        logger.info("Starting Perfect MCP server with advanced features...")
        subprocess.run([sys.executable, 'perfect_mcp_server.py'], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Perfect MCP server: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Perfect MCP server stopped by user")
        return True
    return True

def main():
    """Main entry point for Perfect Research PPT System"""
    print("🎯 Perfect Research PPT System - Setup & Validation")
    print("=" * 60)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    missing_core, missing_advanced, missing_optional = check_dependencies()
    
    if missing_core:
        print("❌ Missing core packages:")
        for package in missing_core:
            print(f"   - {package}")
        print("\nInstall core packages with:")
        print(f"   pip install {' '.join(missing_core)}")
        return False
    
    if missing_advanced:
        print("❌ Missing advanced packages:")
        for package in missing_advanced:
            print(f"   - {package}")
        print("\nInstall advanced packages with:")
        print(f"   pip install {' '.join(missing_advanced)}")
        return False
    
    print("✅ All required dependencies installed")
    
    if missing_optional:
        print("⚠️ Optional packages missing (reduced functionality):")
        for package in missing_optional:
            print(f"   - {package}")
        print("Install optional packages for full features:")
        print(f"   pip install {' '.join(missing_optional)}")
    
    # Check environment variables
    print("\n🔑 Checking environment variables...")
    missing_required, missing_optional_vars = check_environment()
    
    if missing_required:
        print("❌ Missing required environment variables:")
        for var in missing_required:
            print(f"   - {var}")
        print("\nCreate a .env file with:")
        print("   cp .env.template .env")
        print("   # Edit .env and add your API keys")
        return False
    
    print("✅ Required environment variables configured")
    
    if missing_optional_vars:
        print("⚠️ Optional environment variables missing:")
        for var in missing_optional_vars:
            print(f"   - {var}")
        print("Add these to .env for enhanced functionality")
    
    # Validate configuration
    print("\n⚙️ Validating advanced configuration...")
    critical_issues, warnings = validate_config()
    
    if critical_issues:
        print("❌ Critical configuration issues:")
        for issue in critical_issues:
            print(f"   - {issue}")
        return False
    
    if warnings:
        print("⚠️ Configuration warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    
    print("✅ Advanced configuration validated")
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    print("✅ Directories created")
    
    # System overview
    print("\n🚀 Perfect Research PPT System Ready!")
    print("\n🧠 ADVANCED FEATURES AVAILABLE:")
    print("   • LlamaParse PDF Processing + pypdf fallback")
    print("   • Pinecone Vector Storage & Semantic Search")
    print("   • Research Intelligence Analysis Engine")
    print("   • AI-Powered Perfect PPT Generation")
    print("   • SerpAPI Enhanced Search Integration")
    
    # Choose what to run
    print("\n🎯 Choose launch option:")
    print("1. Perfect MCP Server (🔥 RECOMMENDED - All features)")
    print("2. Streamlit Web Interface")
    print("3. Basic MCP Server (fallback)")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\n🚀 Launching Perfect MCP Server with all advanced features...")
                return run_perfect_mcp_server()
            elif choice == "2":
                print("\n🌐 Launching Streamlit Web Interface...")
                return run_streamlit()
            elif choice == "3":
                print("\n📡 Launching Basic MCP Server...")
                return run_mcp_server()
            elif choice == "4":
                print("Goodbye! 👋")
                return True
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 