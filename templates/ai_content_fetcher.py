import openai
import os
from dotenv import load_dotenv
import json
import time
import re
from openai import OpenAI

def validate_content_quality(ai_output, original_contexts, user_topic=""):
    """Validate that AI output doesn't reuse template text"""
    for i, item in enumerate(ai_output):
        text = item.get("text", "")
        original = original_contexts[i] if i < len(original_contexts) else ""
        
        # Check for template text reuse (with topic awareness)
        if has_template_overlap(text, original, user_topic):
            raise ValueError(f"AI reused template text in field {i}: '{text}'")
        
        # Check for appropriate length and content
        if not text or not text.strip():
            raise ValueError(f"AI returned empty text for field {i}")
            
        # Check for visual fit (rough estimate)
        if len(text.split()) > 15 and item.get("field_type") == "title":
            raise ValueError(f"Title text too long for field {i}: '{text}'")

def has_template_overlap(new_text, original_text, user_topic=""):
    """Check if new text reuses significant portions of original template text"""
    if len(original_text) < 10:
        return False
    
    # Split into words and check for overlap
    new_words = set(re.findall(r'\w+', new_text.lower()))
    original_words = set(re.findall(r'\w+', original_text.lower()))
    
    if len(original_words) == 0:
        return False
        
    # Create whitelist of allowed topic-related terms
    topic_words = set(re.findall(r'\w+', user_topic.lower())) if user_topic else set()
    common_tech_terms = {
        'java', 'oops', 'oop', 'object', 'oriented', 'programming', 'class', 'method',
        'inheritance', 'polymorphism', 'encapsulation', 'abstraction', 'interface',
        'benefits', 'principles', 'concepts', 'design', 'patterns', 'code', 'reusability',
        'flexibility', 'modularity', 'hierarchy', 'structure', 'implementation'
    }
    
    # Remove topic-related and technical terms from overlap calculation
    allowed_terms = topic_words.union(common_tech_terms)
    
    # Filter out allowed terms from both sets
    filtered_new = new_words - allowed_terms
    filtered_original = original_words - allowed_terms
    
    if len(filtered_original) == 0:
        return False
        
    # Use higher threshold and filtered words for overlap calculation
    overlap_ratio = len(filtered_new.intersection(filtered_original)) / len(filtered_original)
    return overlap_ratio > 0.6  # Increased from 30% to 60%

def extract_json_list(content):
    start = content.find('[')
    end = content.rfind(']')
    if start != -1 and end != -1 and end > start:
        json_str = content[start:end+1]
        try:
            data = json.loads(json_str)
            for obj in data:
                text = obj.get("text")
                if not text or not text.strip():
                    raise ValueError(f"AI returned blank/whitespace for field: {obj}")
            return data
        except Exception as e:
            print("JSON parsing error:", e)
            print("JSON string:\n", repr(json_str))
            raise
    print("AI output (repr):", repr(content))
    raise ValueError("No valid JSON list found.")

def repair_json(json_string):
    """Attempt to repair common JSON syntax errors"""
    try:
        # Fix common issues
        repaired = json_string
        
        # Fix missing quotes around keys
        repaired = re.sub(r'(\w+):', r'"\1":', repaired)
        
        # Fix trailing commas
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
        
        # Fix unescaped quotes in text
        repaired = re.sub(r'(?<!\\)"(?=.*")', r'\\"', repaired)
        
        return repaired
    except:
        return json_string

def extract_json_from_response(response_text):
    """Extract JSON content from AI response, handling markdown code blocks"""
    try:
        # Remove markdown code blocks if present
        cleaned = response_text.strip()
        
        # Handle markdown code blocks
        if cleaned.startswith('```'):
            lines = cleaned.split('\n')
            # Remove first line if it's ```json or ```
            if lines[0].strip().startswith('```'):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned = '\n'.join(lines)
        
        # Method 1: Direct JSON array extraction
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        
        if start != -1 and end != -1 and end > start:
            json_str = cleaned[start:end+1]
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return json_str
            except:
                pass
        
        # Method 2: Extract JSON objects and create array
        import re
        # More comprehensive regex for JSON objects
        json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
        json_objects = re.findall(json_pattern, cleaned)
        
        if json_objects:
            valid_objects = []
            for obj_str in json_objects:
                try:
                    parsed_obj = json.loads(obj_str)
                    if isinstance(parsed_obj, dict) and 'text' in parsed_obj:
                        valid_objects.append(obj_str)
                except:
                    continue
            
            if valid_objects:
                return '[' + ','.join(valid_objects) + ']'
        
        # Method 3: Look for structured content and rebuild JSON
        if '"text"' in cleaned and ('"source"' in cleaned or '"shape"' in cleaned):
            lines = cleaned.split('\n')
            potential_json_lines = []
            
            for line in lines:
                line = line.strip()
                if (line.startswith('{') or line.startswith('"') or 
                    line.startswith('[') or line.endswith('}') or 
                    line.endswith(',') or line.endswith(']')):
                    potential_json_lines.append(line)
            
            if potential_json_lines:
                reconstructed = '\n'.join(potential_json_lines)
                try:
                    # Clean up common issues
                    reconstructed = reconstructed.replace("'", '"')  # Single to double quotes
                    reconstructed = re.sub(r',(\s*[}\]])', r'\1', reconstructed)  # Remove trailing commas
                    
                    parsed = json.loads(reconstructed)
                    if isinstance(parsed, list):
                        return reconstructed
                except:
                    pass
        
        return None
    except Exception as e:
        print(f"   🔍 JSON extraction error: {e}")
        return None

def create_fallback_formatted_content(original_contexts, user_content, topic_title):
    """Create fallback content using user's provided content"""
    fallback_results = []
    
    if not user_content or not user_content.strip():
        # If no user content, create generic content
        for i, context_info in enumerate(original_contexts):
            char_count = context_info.get("char_count", 100)
            if char_count < 80:
                text = f"{topic_title} Overview"
            else:
                text = f"{topic_title} provides comprehensive insights and practical applications for modern requirements in this field."
            
            result_item = {
                "source": context_info.get("source", "slide"),
                "source_index": context_info.get("source_index", 0),
                "slide": context_info.get("slide", 0),
                "shape": context_info.get("shape", i),
                "text": text[:char_count] if len(text) > char_count else text
            }
            fallback_results.append(result_item)
        return fallback_results
    
    # Parse user content into meaningful chunks
    import re
    
    # Split into paragraphs first
    paragraphs = [p.strip() for p in user_content.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in user_content.split('\n') if p.strip()]
    
    # Split paragraphs into sentences
    all_sentences = []
    for para in paragraphs:
        sentences = re.split(r'[.!?]+', para)
        clean_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
        all_sentences.extend(clean_sentences)
    
    # Create content pools
    header_phrases = []
    body_content = []
    
    # Extract potential headers (first few words of sentences)
    for sentence in all_sentences[:10]:  # Use first 10 sentences for headers
        words = sentence.split()
        if len(words) >= 3:
            header_phrases.append(" ".join(words[:5]))  # First 5 words
    
    # Use all sentences for body content
    body_content = all_sentences
    
    sentence_index = 0
    header_index = 0
    
    for i, context_info in enumerate(original_contexts):
        char_count = context_info.get("char_count", 100)
        
        # Determine if this should be a header or body based on char count
        if char_count < 80:
            # This is likely a header - use header phrases
            if header_index < len(header_phrases):
                text = header_phrases[header_index]
                header_index += 1
            else:
                text = f"{topic_title} Key Concept {i+1}"
        else:
            # This is body text - use sentences
            text = ""
            target_length = char_count * 0.8
            
            sentences_used = 0
            while len(text) < target_length and sentence_index < len(body_content) and sentences_used < 3:
                if text:
                    text += " " + body_content[sentence_index]
                else:
                    text = body_content[sentence_index]
                sentence_index += 1
                sentences_used += 1
            
            if not text:
                text = f"{topic_title} provides comprehensive solutions and practical applications for modern requirements."
        
        # Ensure text fits character count but don't make it too short
        if len(text) > char_count:
            text = text[:char_count-3] + "..."
        elif len(text) < char_count // 3 and char_count > 50:
            # If text is too short for larger fields, add more content
            text += f" This aspect of {topic_title} is essential for understanding the broader concepts."
            if len(text) > char_count:
                text = text[:char_count-3] + "..."
        
        result_item = {
            "source": context_info.get("source", "slide"),
            "source_index": context_info.get("source_index", 0),
            "slide": context_info.get("slide", 0),
            "shape": context_info.get("shape", i),
            "text": text
        }
        fallback_results.append(result_item)
    
    return fallback_results

def create_enhanced_fallback_content(original_contexts, user_content, topic_title):
    """Create high-quality fallback content with proper header-body relationships"""
    fallback_results = []
    
    if not user_content or not user_content.strip():
        # If no user content, create comprehensive generic content
        for i, context_info in enumerate(original_contexts):
            char_count = context_info.get("char_count", 100)
            if char_count < 80:
                text = f"{topic_title} Overview"
            else:
                text = f"{topic_title} provides comprehensive insights and practical applications for modern requirements. This technology offers significant benefits including improved efficiency, enhanced performance, and streamlined processes that drive innovation in today's competitive landscape."
            
            result_item = {
                "source": context_info.get("source", "slide"),
                "source_index": context_info.get("source_index", 0),
                "slide": context_info.get("slide", 0),
                "shape": context_info.get("shape", i),
                "text": text[:char_count] if len(text) > char_count else text
            }
            fallback_results.append(result_item)
        return fallback_results
    
    # Parse user content comprehensively
    import re
    from collections import defaultdict
    
    # Split into paragraphs and sentences
    paragraphs = [p.strip() for p in user_content.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in user_content.split('\n') if p.strip()]
    
    all_sentences = []
    for para in paragraphs:
        sentences = re.split(r'[.!?]+', para)
        clean_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        all_sentences.extend(clean_sentences)
    
    # Group contexts by slide to maintain spatial relationships
    slides_content = defaultdict(list)
    other_content = []
    
    for i, context_info in enumerate(original_contexts):
        slide_num = context_info.get("slide", -1)
        if slide_num >= 0:
            slides_content[slide_num].append((i, context_info))
        else:
            other_content.append((i, context_info))
    
    sentence_index = 0
    
    # Process each slide to maintain header-body relationships
    for slide_num in sorted(slides_content.keys()):
        slide_items = slides_content[slide_num]
        
        # Sort by shape order (spatial positioning)
        slide_items.sort(key=lambda x: x[1].get("shape", 0))
        
        # Identify headers and bodies
        headers = []
        bodies = []
        
        for idx, context_info in slide_items:
            char_count = context_info.get("char_count", 100)
            if char_count < 80:
                headers.append((idx, context_info))
            else:
                bodies.append((idx, context_info))
        
        # Create header-body pairs
        if headers and bodies:
            for i, (header_idx, header_info) in enumerate(headers):
                # Create header content
                if sentence_index < len(all_sentences):
                    # Extract meaningful header from sentence
                    sentence = all_sentences[sentence_index]
                    words = sentence.split()
                    if len(words) >= 4:
                        header_text = " ".join(words[:6])  # More words for better headers
                    else:
                        header_text = f"{topic_title} - Key Concept {i+1}"
                else:
                    header_text = f"{topic_title} - Key Concept {i+1}"
                
                # Ensure header fits
                header_char_count = header_info.get("char_count", 50)
                if len(header_text) > header_char_count:
                    header_text = header_text[:header_char_count-3] + "..."
                
                fallback_results.append({
                    "source": header_info.get("source", "slide"),
                    "source_index": header_info.get("source_index", 0),
                    "slide": header_info.get("slide", 0),
                    "shape": header_info.get("shape", header_idx),
                    "text": header_text
                })
                
                # Find related bodies for this header
                header_shape = header_info.get("shape", 0)
                if i + 1 < len(headers):
                    next_header_shape = headers[i + 1][1].get("shape", 999)
                    related_bodies = [(idx, info) for idx, info in bodies 
                                    if header_shape < info.get("shape", 0) < next_header_shape]
                else:
                    related_bodies = [(idx, info) for idx, info in bodies 
                                    if info.get("shape", 0) > header_shape]
                
                if not related_bodies and bodies:
                    # Distribute remaining bodies evenly
                    bodies_per_header = len(bodies) // len(headers)
                    start_idx = i * bodies_per_header
                    end_idx = start_idx + bodies_per_header if i < len(headers) - 1 else len(bodies)
                    related_bodies = bodies[start_idx:end_idx]
                
                # Create body content for related bodies
                for body_idx, body_info in related_bodies:
                    body_char_count = body_info.get("char_count", 200)
                    target_length = int(body_char_count * 0.9)  # Use 90% of capacity
                    
                    body_text = ""
                    sentences_used = 0
                    
                    while (len(body_text) < target_length and 
                           sentence_index < len(all_sentences) and 
                           sentences_used < 5):  # Up to 5 sentences
                        
                        if body_text:
                            body_text += " " + all_sentences[sentence_index]
                        else:
                            body_text = all_sentences[sentence_index]
                        
                        sentence_index += 1
                        sentences_used += 1
                    
                    # If still too short, enhance with topic-related content
                    if len(body_text) < target_length * 0.6 and body_char_count > 100:
                        enhancement = f" This aspect of {topic_title} demonstrates significant importance in understanding the broader concepts and practical applications within the field."
                        body_text += enhancement
                    
                    # Ensure it fits
                    if len(body_text) > body_char_count:
                        body_text = body_text[:body_char_count-3] + "..."
                    
                    if not body_text:
                        body_text = f"{topic_title} encompasses comprehensive methodologies and practical approaches that provide essential insights for effective implementation and optimal results."
                        if len(body_text) > body_char_count:
                            body_text = body_text[:body_char_count-3] + "..."
                    
                    fallback_results.append({
                        "source": body_info.get("source", "slide"),
                        "source_index": body_info.get("source_index", 0),
                        "slide": body_info.get("slide", 0),
                        "shape": body_info.get("shape", body_idx),
                        "text": body_text
                    })
        
        else:
            # Handle slides with only headers or only bodies
            for idx, context_info in slide_items:
                char_count = context_info.get("char_count", 100)
                
                if char_count < 80:
                    # Header
                    if sentence_index < len(all_sentences):
                        words = all_sentences[sentence_index].split()[:6]
                        text = " ".join(words)
                        sentence_index += 1
                    else:
                        text = f"{topic_title} Overview"
                else:
                    # Body
                    text = ""
                    target_length = int(char_count * 0.9)
                    sentences_used = 0
                    
                    while (len(text) < target_length and 
                           sentence_index < len(all_sentences) and 
                           sentences_used < 4):
                        
                        if text:
                            text += " " + all_sentences[sentence_index]
                        else:
                            text = all_sentences[sentence_index]
                        
                        sentence_index += 1
                        sentences_used += 1
                    
                    if not text:
                        text = f"{topic_title} provides comprehensive solutions and practical applications for modern requirements in this specialized field."
                
                if len(text) > char_count:
                    text = text[:char_count-3] + "..."
                
                fallback_results.append({
                    "source": context_info.get("source", "slide"),
                    "source_index": context_info.get("source_index", 0),
                    "slide": context_info.get("slide", 0),
                    "shape": context_info.get("shape", idx),
                    "text": text
                })
    
    # Handle other content (master, layout)
    for idx, context_info in other_content:
        char_count = context_info.get("char_count", 100)
        
        if sentence_index < len(all_sentences) and char_count > 50:
            text = all_sentences[sentence_index]
            sentence_index += 1
        else:
            text = f"{topic_title} - Professional Presentation"
        
        if len(text) > char_count:
            text = text[:char_count-3] + "..."
        
        fallback_results.append({
            "source": context_info.get("source", "slide"),
            "source_index": context_info.get("source_index", 0),
            "slide": context_info.get("slide", 0),
            "shape": context_info.get("shape", idx),
            "text": text
        })
    
    # Sort results by original order
    fallback_results.sort(key=lambda x: (x.get("slide", -1), x.get("shape", 0)))
    
    return fallback_results

def get_ai_content(prompt, original_contexts=None, api_key=None, model="gpt-4o", user_topic="", use_enhanced=True, user_content=""):
    load_dotenv()
    
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=api_key)
    
    # Extract context strings if original_contexts contains metadata objects
    if original_contexts and isinstance(original_contexts[0], dict):
        context_strings = [item.get("context", "") for item in original_contexts]
        metadata_objects = original_contexts
    else:
        context_strings = original_contexts or []
        metadata_objects = original_contexts
    
    try:
        # Use the provided prompt directly
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that formats content for PowerPoint presentations. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.3
        )
        
        raw_response = response.choices[0].message.content.strip()
        print(f"   📝 AI response received ({len(raw_response)} chars)")
        
        # Extract and parse JSON from response
        json_content = extract_json_from_response(raw_response)
        
        if json_content:
            output = json.loads(json_content)
            
            # Validate and fill empty fields
            if isinstance(output, list) and len(output) > 0:
                print(f"   ✅ Parsed {len(output)} items from AI response")
                
                # Fill empty fields with fallback content
                for i, item in enumerate(output):
                    if not item.get("text", "").strip():
                        # Use user content for fallback
                        if user_content and metadata_objects:
                            # Use user content for fallback
                            fallback_output = create_enhanced_fallback_content(metadata_objects, user_content, user_topic)
                            print(f"   ✅ Generated {len(fallback_output)} fallback items from user content")
                            return fallback_output
                        else:
                            # Simple fallback if no user content
                            item["text"] = f"{user_topic} Key Point {i+1}"
                            print(f"   📝 Generated simple fallback for field {i}")
                
                # Validate content quality if original contexts provided
                if context_strings:
                    validate_content_quality(output, context_strings, user_topic)
                    
                return output
            else:
                print("   ❌ No valid items in AI response")
                if user_content and metadata_objects:
                    fallback_output = create_enhanced_fallback_content(metadata_objects, user_content, user_topic)
                    print(f"   ✅ Generated {len(fallback_output)} fallback items from user content")
                    return fallback_output
        else:
            print("   ❌ No JSON content found in response")
            if user_content and metadata_objects:
                fallback_output = create_enhanced_fallback_content(metadata_objects, user_content, user_topic)
                print(f"   ✅ Generated {len(fallback_output)} fallback items from user content")
                return fallback_output
                
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parsing failed: {e}")
        if user_content and metadata_objects:
            fallback_output = create_enhanced_fallback_content(metadata_objects, user_content, user_topic)
            print(f"   ✅ Generated {len(fallback_output)} fallback items from user content")
            return fallback_output
    except Exception as e:
        print(f"   ❌ AI request failed: {e}")
        if user_content and metadata_objects:
            fallback_output = create_enhanced_fallback_content(metadata_objects, user_content, user_topic)
            print(f"   ✅ Generated {len(fallback_output)} fallback items from user content")
            return fallback_output
    
    raise ValueError("AI did not return valid content and no fallback available.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ai_content_fetcher.py <prompt.txt>")
    else:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            prompt = f.read()
        print(json.dumps(get_ai_content(prompt), indent=2))
