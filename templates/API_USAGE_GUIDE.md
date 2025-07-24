# AI PowerPoint Generator API Usage Guide

## Overview
This API generates professional PowerPoint presentations from user content using AI formatting. The system takes raw content and a topic, then creates a formatted presentation using a predefined template.

## Function Details

### Import
```python
from main import generate_presentation_api
```

### Function Signature
```python
def generate_presentation_api(content, topic):
    """
    Generate PowerPoint presentation from content and topic
    
    Args:
        content (str): The content to format for presentation
        topic (str): Main topic/title for presentation
    
    Returns:
        str: Path to generated PowerPoint file (e.g., "presentations/Topic_presentation.pptx")
    """
```

## Basic Usage

### Simple Example
```python
from main import generate_presentation_api

# Your content and topic
user_content = """
Machine learning is a powerful technology that enables computers 
to learn from data. It involves algorithms that can identify patterns 
and make predictions without explicit programming. The field encompasses 
supervised learning, unsupervised learning, and reinforcement learning.
"""

user_topic = "Machine Learning Overview"

# Generate presentation
try:
    output_file = generate_presentation_api(user_content, user_topic)
    print(f"✅ Presentation created: {output_file}")
    # Returns: "presentations/Machine_Learning_Overview_presentation.pptx"
except Exception as e:
    print(f"❌ Error: {e}")
```

## Backend Integration

### Service Function
```python
# backend_service.py
from main import generate_presentation_api
import os

def create_presentation(content, topic):
    """
    Service function to create PowerPoint presentation
    """
    try:
        # Generate presentation
        output_path = generate_presentation_api(content, topic)
        
        # Check if file was created
        if os.path.exists(output_path):
            return {
                "success": True,
                "file_path": output_path,
                "message": "Presentation created successfully"
            }
        else:
            return {
                "success": False,
                "error": "File was not created"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Usage example
if __name__ == "__main__":
    content = "Your content here..."
    topic = "Your Topic"
    
    result = create_presentation(content, topic)
    
    if result["success"]:
        print(f"Success: {result['file_path']}")
    else:
        print(f"Error: {result['error']}")
```

## Web Framework Integration

### Flask Integration
```python
from flask import Flask, request, jsonify, send_file
from main import generate_presentation_api
import os

app = Flask(__name__)

@app.route('/generate-presentation', methods=['POST'])
def generate_presentation():
    data = request.json
    content = data.get('content')
    topic = data.get('topic')
    
    if not content or not topic:
        return jsonify({
            "success": False,
            "error": "Content and topic are required"
        }), 400
    
    try:
        output_path = generate_presentation_api(content, topic)
        return jsonify({
            "success": True,
            "file_path": output_path,
            "message": "Presentation generated successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/download/<filename>')
def download_file(filename):
    file_path = f"presentations/{filename}"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
```

### FastAPI Integration
```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from main import generate_presentation_api
import os

app = FastAPI()

class PresentationRequest(BaseModel):
    content: str
    topic: str

@app.post("/generate-presentation")
async def generate_presentation(request: PresentationRequest):
    try:
        output_path = generate_presentation_api(request.content, request.topic)
        return {
            "success": True,
            "file_path": output_path,
            "message": "Presentation generated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"presentations/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Requirements

### Dependencies
```bash
pip install openai python-pptx python-dotenv flask fastapi uvicorn
```

### Environment Setup
Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Project Structure
```
your_project/
├── main.py                    # Contains generate_presentation_api()
├── ai_content_fetcher.py      # AI processing logic
├── prompt_builder.py          # Prompt creation
├── extract_metadata.py        # Template analysis
├── replace_content.py         # PowerPoint modification
├── .env                       # OpenAI API key
├── templates/
│   └── test_slide.pptx       # Template file (required)
└── presentations/             # Output folder (auto-created)
```

## Function Behavior

### Input Processing
- **Content**: Raw text content that will be formatted and distributed across slides
- **Topic**: Main presentation topic used for titles and filename generation

### Output
- **Return Type**: String
- **Format**: `"presentations/{clean_topic}_presentation.pptx"`
- **Example**: `"presentations/Machine_Learning_presentation.pptx"`

### Topic Processing
- Spaces converted to underscores
- Special characters removed
- Example: "Machine Learning" → "Machine_Learning"

### File Creation
- File is created and saved before function returns
- Path is relative to project directory
- File is ready for immediate use/download

## Error Handling

### Common Errors
1. **Template not found**: Ensure `templates/test_slide.pptx` exists
2. **OpenAI API key missing**: Check `.env` file
3. **Permission denied**: Close PowerPoint if file is open
4. **Invalid content**: Ensure content is not empty

### Error Handling Example
```python
try:
    output_path = generate_presentation_api(content, topic)
    print(f"Success: {output_path}")
except FileNotFoundError as e:
    print(f"Template error: {e}")
except ValueError as e:
    print(f"Input error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Notes

### Processing Time
- Typical processing: 30-60 seconds
- Depends on content length and complexity
- Uses batch processing for efficiency

### Resource Usage
- Requires OpenAI API calls
- Processes ~116 text fields per presentation
- Creates professional, formatted output

## Testing

### Quick Test
```python
from main import generate_presentation_api

# Test function
test_content = "AI and machine learning are transforming industries through automation and intelligent decision-making."
test_topic = "AI Test"

result = generate_presentation_api(test_content, test_topic)
print(f"Generated: {result}")
```

### CLI Testing
```bash
python main.py  # Interactive CLI version for testing
```

## Production Deployment

### Considerations
1. **API Rate Limits**: Monitor OpenAI usage
2. **File Storage**: Clean up old presentations periodically
3. **Error Logging**: Implement comprehensive logging
4. **Validation**: Validate input content length and format
5. **Security**: Sanitize user inputs

### Scaling
- Function is stateless and thread-safe
- Can be deployed in containerized environments
- Supports concurrent requests (limited by OpenAI rate limits)

## Support

### File Locations
- **Main API**: `main.py` - `generate_presentation_api()`
- **Template**: `templates/test_slide.pptx`
- **Output**: `presentations/` folder
- **Config**: `.env` file

### Integration Ready
This API is production-ready and can be integrated into any Python web application or service. 