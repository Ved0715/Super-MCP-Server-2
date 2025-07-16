# 🚀 Prompt Management System

Centralized prompt management for the Perfect MCP Server. This system externalizes all prompts into structured YAML files for better maintainability, collaboration, and version control.

## 📁 Directory Structure

```
prompts/
├── __init__.py                     # Prompt management system
├── README.md                       # This documentation
├── workflows/                      # MCP workflow prompts
│   └── research_workflows.yaml    # Research analysis workflows
├── ai_enhancement/                 # AI enhancement prompts
│   └── enhancement_prompts.yaml   # Research enhancement prompts
├── processing/                     # Document processing prompts
│   └── document_processing.yaml   # PDF and content processing
├── knowledge_base/                 # Knowledge base retrieval
│   └── retrieval_prompts.yaml     # Search and retrieval prompts
└── presentation/                   # Presentation generation
    └── generation_prompts.yaml    # Presentation creation prompts
```

## 🎯 Benefits

### **MCP Architecture Alignment**
- ✅ Treats prompts as first-class resources (MCP principle)
- ✅ Separates content from logic
- ✅ Enables dynamic prompt loading

### **Development Benefits**
- ✅ **Maintainability**: Edit prompts without touching code
- ✅ **Collaboration**: Non-developers can edit prompts
- ✅ **Version Control**: Track prompt evolution in git
- ✅ **Testing**: Easy A/B testing of different prompts
- ✅ **Environment Support**: Different prompts per environment

## 🚀 Usage

### **Basic Usage**
```python
from prompts import get_prompt

# Get a workflow prompt
workflow_text = get_prompt('workflows', 'research_analysis_workflow', 
                          paper_id='paper_123', 
                          analysis_focus='methodology')

# Get an AI enhancement prompt
enhancement_text = get_prompt('ai_enhancement', 'research_insights_enhancement',
                             title='Paper Title',
                             authors='Author Names',
                             research_analysis='Analysis Data')
```

### **Convenience Functions**
```python
from prompts import (
    get_workflow_prompt,
    get_ai_enhancement_prompt,
    get_processing_prompt,
    get_knowledge_base_prompt,
    get_presentation_prompt
)

# Use category-specific functions
workflow = get_workflow_prompt('research_analysis_workflow', paper_id='123')
enhancement = get_ai_enhancement_prompt('quality_assessment_enhancement', paper_data='data')
```

### **Direct Manager Access**
```python
from prompts import prompt_manager

# List all available prompts
all_prompts = prompt_manager.list_prompts()

# Get prompt metadata
metadata = prompt_manager.get_prompt_metadata('workflows', 'research_analysis_workflow')

# Reload prompts (useful in development)
prompt_manager.reload_prompts()
```

## 📝 Prompt Categories

### **1. Workflows (`workflows/`)**
MCP workflow prompts for complete research processes:
- `research_analysis_workflow` - Complete paper analysis
- `presentation_creation_workflow` - Presentation creation
- `literature_review_workflow` - Literature review process
- `research_insights_workflow` - Insight generation

### **2. AI Enhancement (`ai_enhancement/`)**
Prompts for AI-powered research enhancement:
- `research_insights_enhancement` - Generate insights
- `quality_assessment_enhancement` - Quality assessment
- `general_enhancement` - General enhancement
- `ai_sampling_system_prompt` - System prompt for sampling

### **3. Processing (`processing/`)**
Document processing and extraction prompts:
- `research_paper_llamaparse_instructions` - Research paper processing
- `knowledge_base_llamaparse_instructions` - Knowledge base processing
- `universal_processor_system_prompt` - Universal processor

### **4. Knowledge Base (`knowledge_base/`)**
Knowledge base search and retrieval prompts:
- `query_analysis_prompt` - Query intent analysis
- `base_knowledge_base_prompt` - Base system prompt
- `definition_query_prompt` - Definition queries
- `explanation_query_prompt` - Explanation queries
- `comparison_query_prompt` - Comparison queries
- `implementation_query_prompt` - Implementation queries
- `mathematical_query_prompt` - Mathematical queries
- `general_query_prompt` - General queries

### **5. Presentation (`presentation/`)**
Presentation generation and analysis prompts:
- `chain_of_thought_presentation_analysis` - CoT analysis
- `presentation_system_prompt` - System prompt
- `academic_presentation_prompt` - Academic style
- `business_presentation_prompt` - Business style

## 🔧 YAML Format

### **Standard Format**
```yaml
prompt_name:
  description: "Brief description of the prompt"
  variables: ["var1", "var2"]  # Optional: list of variables
  template: |
    Your prompt template here.
    Use {var1} and {var2} for substitution.
```

### **Simple Format**
```yaml
simple_prompt: |
  Direct prompt text without metadata.
```

## 🔄 Integration with Existing Code

### **Before (Inline Prompts)**
```python
def _setup_prompts(self):
    prompt_text = f"""
    # Research Paper Analysis Workflow
    Please analyze the research paper with ID: {paper_id}
    ...
    """
```

### **After (Externalized Prompts)**
```python
def _setup_prompts(self):
    from prompts import get_workflow_prompt
    
    prompt_text = get_workflow_prompt('research_analysis_workflow',
                                     paper_id=paper_id,
                                     analysis_focus=analysis_focus)
```

## 🛠️ Development Workflow

### **Adding New Prompts**
1. Choose appropriate category or create new one
2. Add prompt to relevant YAML file
3. Use descriptive names and include metadata
4. Test with variables if needed

### **Updating Existing Prompts**
1. Edit YAML file directly
2. Test changes
3. Commit changes for version tracking

### **Testing Prompts**
```python
from prompts import prompt_manager

# Test prompt loading
prompt = prompt_manager.get_prompt('workflows', 'test_prompt', var1='value1')
print(prompt)

# Test with missing variables
prompt = prompt_manager.get_prompt('workflows', 'test_prompt')  # Should handle gracefully
```

## 🚨 Error Handling

The system handles various error scenarios gracefully:
- **Missing categories**: Returns error message, logs warning
- **Missing prompts**: Returns error message, logs warning  
- **Missing variables**: Returns unsubstituted prompt, logs warning
- **File not found**: Continues with empty cache, logs error

## 🔄 Dynamic Reloading

In development, you can reload prompts without restarting:
```python
from prompts import prompt_manager
prompt_manager.reload_prompts()
```

## 📊 Best Practices

1. **Use descriptive names**: `research_analysis_workflow` not `prompt1`
2. **Include metadata**: Always add description and variables list
3. **Variable naming**: Use clear variable names like `{paper_id}` not `{p}`
4. **Documentation**: Document complex prompts in comments
5. **Testing**: Test prompts with various inputs
6. **Version control**: Commit prompt changes with descriptive messages

## 🔍 Monitoring

The system provides logging for:
- Prompt loading success/failure
- Missing prompts/categories
- Variable substitution issues
- File I/O errors

Check logs for prompt-related issues:
```
✅ Loaded prompts from 5 categories
⚠️ Prompt 'missing_prompt' not found in category 'workflows'
❌ Error loading prompts: FileNotFoundError
``` 