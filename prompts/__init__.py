"""
🚀 Prompt Management System
Centralized prompt management for the Perfect MCP Server
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    """Centralized prompt management system"""
    
    def __init__(self):
        """Initialize the prompt manager"""
        self.prompts_dir = Path(__file__).parent
        self._prompt_cache = {}
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """Load all prompts from files"""
        try:
            # Load workflow prompts
            workflow_file = self.prompts_dir / "workflows" / "research_workflows.yaml"
            if workflow_file.exists():
                with open(workflow_file, 'r', encoding='utf-8') as f:
                    self._prompt_cache['workflows'] = yaml.safe_load(f)
            
            # Load AI enhancement prompts  
            ai_file = self.prompts_dir / "ai_enhancement" / "enhancement_prompts.yaml"
            if ai_file.exists():
                with open(ai_file, 'r', encoding='utf-8') as f:
                    self._prompt_cache['ai_enhancement'] = yaml.safe_load(f)
            
            # Load processing prompts
            processing_file = self.prompts_dir / "processing" / "document_processing.yaml"
            if processing_file.exists():
                with open(processing_file, 'r', encoding='utf-8') as f:
                    self._prompt_cache['processing'] = yaml.safe_load(f)
            
            # Load knowledge base prompts
            kb_file = self.prompts_dir / "knowledge_base" / "retrieval_prompts.yaml"
            if kb_file.exists():
                with open(kb_file, 'r', encoding='utf-8') as f:
                    self._prompt_cache['knowledge_base'] = yaml.safe_load(f)
            
            # Load presentation prompts
            presentation_file = self.prompts_dir / "presentation" / "generation_prompts.yaml"
            if presentation_file.exists():
                with open(presentation_file, 'r', encoding='utf-8') as f:
                    self._prompt_cache['presentation'] = yaml.safe_load(f)
            
            logger.info(f"✅ Loaded prompts from {len(self._prompt_cache)} categories")
            
        except Exception as e:
            logger.error(f"❌ Error loading prompts: {e}")
            self._prompt_cache = {}
    
    def get_prompt(self, category: str, prompt_name: str, **kwargs) -> str:
        """
        Get a prompt with optional variable substitution
        
        Args:
            category: Prompt category (workflows, ai_enhancement, etc.)
            prompt_name: Name of the specific prompt
            **kwargs: Variables to substitute in the prompt
            
        Returns:
            Formatted prompt string
        """
        try:
            if category not in self._prompt_cache:
                logger.warning(f"⚠️ Prompt category '{category}' not found")
                return f"Prompt category '{category}' not found"
            
            if prompt_name not in self._prompt_cache[category]:
                logger.warning(f"⚠️ Prompt '{prompt_name}' not found in category '{category}'")
                return f"Prompt '{prompt_name}' not found"
            
            prompt_template = self._prompt_cache[category][prompt_name]
            
            # Handle both string and dict formats
            if isinstance(prompt_template, dict):
                if 'template' in prompt_template:
                    prompt_text = prompt_template['template']
                elif 'content' in prompt_template:
                    prompt_text = prompt_template['content']
                else:
                    prompt_text = str(prompt_template)
            else:
                prompt_text = str(prompt_template)
            
            # Substitute variables
            if kwargs:
                try:
                    return prompt_text.format(**kwargs)
                except KeyError as e:
                    logger.warning(f"⚠️ Missing variable {e} in prompt '{prompt_name}'")
                    return prompt_text
            
            return prompt_text
            
        except Exception as e:
            logger.error(f"❌ Error getting prompt: {e}")
            return f"Error loading prompt: {str(e)}"
    
    def get_prompt_metadata(self, category: str, prompt_name: str) -> Dict[str, Any]:
        """Get metadata for a specific prompt"""
        try:
            if category in self._prompt_cache and prompt_name in self._prompt_cache[category]:
                prompt_data = self._prompt_cache[category][prompt_name]
                if isinstance(prompt_data, dict):
                    return {
                        'description': prompt_data.get('description', ''),
                        'variables': prompt_data.get('variables', []),
                        'category': category,
                        'name': prompt_name
                    }
            return {}
        except Exception as e:
            logger.error(f"❌ Error getting prompt metadata: {e}")
            return {}
    
    def list_prompts(self, category: Optional[str] = None) -> Dict[str, Any]:
        """List all available prompts"""
        if category:
            return {category: list(self._prompt_cache.get(category, {}).keys())}
        
        return {cat: list(prompts.keys()) for cat, prompts in self._prompt_cache.items()}
    
    def reload_prompts(self):
        """Reload all prompts from files"""
        self._prompt_cache.clear()
        self._load_all_prompts()

# Global prompt manager instance
prompt_manager = PromptManager()

# Convenience functions
def get_prompt(category: str, prompt_name: str, **kwargs) -> str:
    """Get a prompt with variable substitution"""
    return prompt_manager.get_prompt(category, prompt_name, **kwargs)

def get_workflow_prompt(prompt_name: str, **kwargs) -> str:
    """Get a workflow prompt"""
    return get_prompt('workflows', prompt_name, **kwargs)

def get_ai_enhancement_prompt(prompt_name: str, **kwargs) -> str:
    """Get an AI enhancement prompt"""
    return get_prompt('ai_enhancement', prompt_name, **kwargs)

def get_processing_prompt(prompt_name: str, **kwargs) -> str:
    """Get a processing prompt"""
    return get_prompt('processing', prompt_name, **kwargs)

def get_knowledge_base_prompt(prompt_name: str, **kwargs) -> str:
    """Get a knowledge base prompt"""
    return get_prompt('knowledge_base', prompt_name, **kwargs)

def get_presentation_prompt(prompt_name: str, **kwargs) -> str:
    """Get a presentation prompt"""
    return get_prompt('presentation', prompt_name, **kwargs) 