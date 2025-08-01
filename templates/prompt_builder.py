import json
from collections import defaultdict
import openai
import os
from dotenv import load_dotenv
from .content_format_analyzer import content_analyzer

def decompose_user_prompt(user_prompt, num_slides, user_topic=""):
    """
    Break down user prompt into slide-specific questions using AI
    """
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found")
    
    client = openai.OpenAI(api_key=api_key)
    
    decomposition_prompt = f"""
    You are an expert presentation planner. Break down this presentation prompt into {num_slides} specific questions.

    PRESENTATION TOPIC: {user_topic}
    USER PROMPT: {user_prompt}
    NUMBER OF SLIDES: {num_slides}

    TASK: Create {num_slides} specific questions that will guide content for each slide.
    
    REQUIREMENTS:
    - Each question should be specific and focused
    - Questions should create a logical flow
    - Questions should cover different aspects of the topic
    - Questions should be answerable from typical content about {user_topic}
    - Make questions engaging and presentation-friendly
    - VARY THE QUESTION TYPES to create different content formats:
      * Use "What are..." for lists and bullet points
      * Use "How do...compare" or "What are the differences..." for comparison/boxes
      * Use "What data shows..." or "What statistics..." for tables/data
      * Use "How has...evolved" or "What is the process..." for timeline/process
      * Use "How does...work" or "Why is..." for paragraph/explanatory

    RESPONSE FORMAT: Return ONLY a JSON array of strings, no explanations.
    Example: ["What is the main concept?", "How do different approaches compare?", "What data shows the effectiveness?", "How has the field evolved?", "Why is this important?"]

    Generate {num_slides} questions with varied formats:
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": decomposition_prompt}],
            max_tokens=1000,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        raw_response = response.choices[0].message.content
        questions = json.loads(raw_response)
        
        # Handle different response formats
        if isinstance(questions, dict):
            # If response is {"questions": [...]}
            if "questions" in questions:
                questions = questions["questions"]
            else:
                # If response is {"1": "question1", "2": "question2", ...}
                questions = list(questions.values())
        elif isinstance(questions, list):
            questions = questions
        else:
            raise ValueError("Unexpected response format")
        
        # Ensure we have the right number of questions
        if len(questions) != num_slides:
            # Pad or truncate to match num_slides
            if len(questions) < num_slides:
                for i in range(len(questions), num_slides):
                    questions.append(f"Additional aspect of {user_topic}")
            else:
                questions = questions[:num_slides]
        
        return questions
        
    except Exception as e:
        print(f"Error decomposing prompt: {e}")
        # Fallback: generate generic questions
        fallback_questions = []
        for i in range(num_slides):
            if i == 0:
                fallback_questions.append(f"What is {user_topic}?")
            elif i == num_slides - 1:
                fallback_questions.append(f"What are the key takeaways about {user_topic}?")
            else:
                fallback_questions.append(f"What is aspect {i+1} of {user_topic}?")
        return fallback_questions

def extract_relevant_content(user_content, question, user_topic=""):
    """
    Search through user content to find relevant answers for each question
    """
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found")
    
    client = openai.OpenAI(api_key=api_key)
    
    # Truncate content if too long
    max_content_length = 3000
    truncated_content = user_content[:max_content_length]
    if len(user_content) > max_content_length:
        truncated_content += "..."
    
    search_prompt = f"""
    QUESTION: {question}
    
    USER CONTENT:
    {truncated_content}
    
    TASK: Find the most relevant information from the user content that answers this question.
    
    REQUIREMENTS:
    - Extract and format the answer using ONLY the provided content
    - Focus on information that directly answers the question
    - Include supporting details and examples from the content
    - Maintain the original meaning and context
    - If the content doesn't fully answer the question, use what's available
    - Do not add any external knowledge or information
    
    RESPONSE: Provide a comprehensive answer using only the user content.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": search_prompt}],
            max_tokens=1500,
            temperature=0.2
        )
        
        relevant_content = response.choices[0].message.content.strip()
        return relevant_content
        
    except Exception as e:
        print(f"Error extracting relevant content: {e}")
        # Fallback: return original content
        return user_content

def build_question_based_prompt(user_topic, slide_question, relevant_content, slide_metadata, slide_num):
    """
    Build prompt for specific slide based on question and relevant content
    """
    if not relevant_content:
        raise ValueError("No relevant content provided for slide")
    
    # Analyze content format with flow and template assignment
    from .extract_metadata import analyze_template_inventory
    template_path = "templates/test_slide.pptx"
    template_inventory = analyze_template_inventory(template_path)
    
    format_analysis = content_analyzer.analyze_content_format_with_flow(
        slide_question, relevant_content, slide_metadata, slide_num + 1, 
        slide_num + 1, user_topic, template_inventory
    )
    
    # Truncate content if too long
    max_content_length = 2000
    truncated_content = relevant_content[:max_content_length]
    if len(relevant_content) > max_content_length:
        truncated_content += "..."
    
    # Group metadata by slide first
    slides_content = defaultdict(list)
    other_content = []
    
    for item in slide_metadata:
        slide_num_meta = item.get('slide', -1)
        if slide_num_meta >= 0:
            slides_content[slide_num_meta].append(item)
        else:
            other_content.append(item)
    
    prompt = f"""You are an expert content formatter for PowerPoint about '{user_topic}'.
SLIDE {slide_num + 1} QUESTION: {slide_question}

TASK: Format the relevant content to answer the slide question while fitting PowerPoint fields.

RELEVANT CONTENT FOR THIS SLIDE:
{truncated_content}

CONTENT FORMAT ANALYSIS:
📊 Detected Format: {format_analysis['content_format'].replace('_', ' ').title()}
🎨 Recommended Slide Type: {format_analysis['recommended_slide_type'].replace('_', ' ').title()}
🎯 Confidence Score: {format_analysis['confidence_score']:.2f}
📋 Template Match: Slide {format_analysis['template_match']['slide_number'] + 1} ({format_analysis['template_match']['template_type'].replace('_', ' ').title()})
🎭 Slide Role: {format_analysis['slide_role'].replace('_', ' ').title()}
📈 Flow Adjustment: {format_analysis['flow_adjustment']:.2f}

{format_analysis['formatting_instructions']}

CONTENT FORMATTING APPROACH:
- Answer the specific question for this slide
- Use ONLY the provided relevant content
- Format content according to the detected format above
- Maintain header-body relationships
- Create engaging titles and comprehensive explanations
- If content is insufficient, use what's available without adding external knowledge

CRITICAL: Headers and body text must be thematically connected within each slide.
CRITICAL: Generate content for ALL fields provided - do not skip any fields.
CRITICAL: Use ONLY the relevant content provided - no external knowledge or generation.
CRITICAL: Focus on answering the slide question specifically.
CRITICAL: Follow the format-specific instructions above.

RESPONSE FORMAT: JSON array only. No markdown, no explanations, no additional text.

EXAMPLE:
[
  {{"source": "slide", "source_index": 0, "slide": {slide_num}, "shape": 0, "text": "Answer to: {slide_question}"}},
  {{"source": "slide", "source_index": 0, "slide": {slide_num}, "shape": 1, "text": "Detailed explanation using only the provided relevant content to answer the specific question."}}
]

"""
    
    field_specs = []
    relationship_instructions = []
    
    # Process each slide to establish header-body relationships for formatting
    for slide_num_meta in sorted(slides_content.keys()):
        slide_items = slides_content[slide_num_meta]
        
        # Sort by spatial position
        sorted_items = sorted(slide_items, key=lambda x: (x.get("positioning", {}).get("top", 0), x.get("positioning", {}).get("left", 0)))
        
        # Identify headers and bodies in this slide
        slide_headers = []
        slide_bodies = []
        
        for item in sorted_items:
            role = determine_field_role(item.get("context", ""))
            item["role"] = role
            
            if role in ["title", "subtitle"] or item["char_count"] < 80:
                slide_headers.append(item)
            else:
                slide_bodies.append(item)
        
        # Create header-body pairs for content formatting
        if slide_headers and slide_bodies:
            prompt += f"SLIDE {slide_num_meta + 1} - Question-Based Content:\n"
            prompt += f"  Focus: Answer the question: {slide_question}\n"
            
            # Assign content sections to header-body pairs
            for i, header in enumerate(slide_headers):
                section_focus = f"Answer to: {slide_question}"
                
                # Find body text that should pair with this header
                related_bodies = []
                header_shape = header.get("shape", 0)
                
                if i + 1 < len(slide_headers):
                    next_header_shape = slide_headers[i + 1].get("shape", 999)
                    related_bodies = [b for b in slide_bodies if header_shape < b.get("shape", 0) < next_header_shape]
                else:
                    related_bodies = [b for b in slide_bodies if b.get("shape", 0) > header_shape]
                
                if not related_bodies and slide_bodies:
                    bodies_per_header = len(slide_bodies) // len(slide_headers)
                    start_idx = i * bodies_per_header
                    end_idx = start_idx + bodies_per_header if i < len(slide_headers) - 1 else len(slide_bodies)
                    related_bodies = slide_bodies[start_idx:end_idx]
                
                # Add relationship instruction for formatting
                relationship_instructions.append(f"Header-Body Pair: '{section_focus}' header with {len(related_bodies)} related body text(s)")
                
                # Add header to field specs
                header_spec = {
                    "source": "slide",  # Default for new format
                    "source_index": 0,  # Default for new format
                    "slide": header.get("slide", -1),
                    "shape": header["shape"],
                    "role": "header",
                    "char_count": header["char_count"],
                    "content_focus": section_focus,
                    "relationship": "header"
                }
                field_specs.append(header_spec)
                
                # Add related bodies to field specs
                for body in related_bodies:
                    body_spec = {
                        "source": "slide",  # Default for new format
                        "source_index": 0,  # Default for new format
                        "slide": body.get("slide", -1),
                        "shape": body["shape"],
                        "role": "body",
                        "char_count": body["char_count"],
                        "content_focus": section_focus,
                        "relationship": f"explains_header_{header['shape']}"
                    }
                    field_specs.append(body_spec)
                
                prompt += f"  - Header: '{section_focus}' → Body: Answer the question using relevant content\n"
            
            prompt += "\n"
        
        else:
            # Handle slides with only headers or only bodies
            for item in sorted_items:
                role = determine_field_role(item.get("context", ""))
                field_spec = {
                    "source": "slide",  # Default for new format
                    "source_index": 0,  # Default for new format
                    "slide": item.get("slide", -1),
                    "shape": item["shape"],
                    "role": role,
                    "char_count": item["char_count"],
                    "content_focus": f"Answer to: {slide_question}",
                    "relationship": "standalone"
                }
                field_specs.append(field_spec)
    
    # Add other content (master, layout)
    for item in other_content:
        role = determine_field_role(item.get("context", ""))
        field_spec = {
            "source": "template",  # Default for new format
            "source_index": 0,  # Default for new format
            "slide": item.get("slide", -1),
            "shape": item["shape"],
            "role": role,
            "char_count": item["char_count"],
            "content_focus": f"Answer to: {slide_question}",
            "relationship": "template"
        }
        field_specs.append(field_spec)
    
    # Add detailed instructions for content formatting
    prompt += f"""CONTENT FORMATTING RULES:
1. Use ONLY the relevant content provided - no external knowledge
2. Each header should directly relate to answering the slide question
3. Body text should provide comprehensive answers using the relevant content
4. Match approximate character counts but prioritize answering the question
5. Maintain exact field order in response
6. Create thematic consistency between paired headers and bodies
7. Focus on answering the specific question: {slide_question}
8. If relevant content is insufficient, use what's available without adding external information

HEADER-BODY RELATIONSHIPS:
{chr(10).join(relationship_instructions)}

FIELDS TO FILL:
{json.dumps(field_specs, indent=2)}

RESPONSE: JSON array only. Fill "text" field for each item using ONLY the relevant content. No other text or formatting.
"""
    
    return prompt

def generate_slide_content(question, relevant_content, slide_metadata, user_topic, slide_num):
    """
    Generate content for each slide based on question and relevant content
    """
    # Build the question-based prompt
    prompt = build_question_based_prompt(user_topic, question, relevant_content, slide_metadata, slide_num)
    
    # Get AI content using existing infrastructure
    from .ai_content_fetcher import get_ai_content
    slide_content = get_ai_content(
        prompt=prompt,
        original_contexts=slide_metadata,
        user_topic=user_topic,
        user_content=relevant_content
    )
    
    return slide_content

def generate_presentation_with_prompt(user_content, user_prompt, topic, num_slides):
    """
    Generate presentation using user prompt decomposition and question-based content generation
    """
    from tqdm import tqdm
    import time
    
    print(f"🎯 Decomposing user prompt into {num_slides} questions...")
    
    # Step 1: Decompose user prompt into questions
    with tqdm(total=1, desc="🤖 Generating questions", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        slide_questions = decompose_user_prompt(user_prompt, num_slides, topic)
        pbar.update(1)
    
    print(f"✅ Generated {len(slide_questions)} questions:")
    for i, question in enumerate(slide_questions):
        print(f"   Slide {i+1}: {question}")
    
    # Step 2: Extract template metadata
    from .extract_metadata import extract_all_metadata
    template_path = "templates/test_slide.pptx"
    
    with tqdm(total=1, desc="📋 Analyzing template", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        all_metadata = extract_all_metadata(template_path)
        pbar.update(1)
    
    # Step 3: Group metadata by slides
    slides_metadata = defaultdict(list)
    other_metadata = []
    
    # Handle the new format where all_metadata is a list of slide metadata lists
    for slide_num, slide_metadata in enumerate(all_metadata):
        if slide_num < num_slides:
            slides_metadata[slide_num] = slide_metadata
        else:
            other_metadata.extend(slide_metadata)
    
    all_results = []
    
    # Step 4: For each slide/question with progress bar
    with tqdm(total=num_slides, desc="🎯 Generating slide content", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} slides") as pbar:
        for slide_num in range(num_slides):
            question = slide_questions[slide_num]
            pbar.set_description(f"🎯 Processing Slide {slide_num + 1}")
            
            # Extract relevant content for this question
            relevant_content = extract_relevant_content(user_content, question, topic)
            
            # Get slide metadata for this specific slide
            slide_metadata = slides_metadata.get(slide_num, [])
            
            if not slide_metadata:
                print(f"   ⚠️ No metadata found for slide {slide_num + 1}, skipping...")
                pbar.update(1)
                continue
            
            # Analyze content format with flow and template assignment
            from .extract_metadata import analyze_template_inventory
            template_path = "templates/test_slide.pptx"
            template_inventory = analyze_template_inventory(template_path)
            
            format_analysis = content_analyzer.analyze_content_format_with_flow(
                question, relevant_content, slide_metadata, slide_num + 1, 
                num_slides, topic, template_inventory
            )
            
            print(f"\n🎯 SLIDE {slide_num + 1} CONTENT ANALYSIS:")
            print(f"   📝 Question: {question}")
            print(f"   📊 Content Format: {format_analysis['content_format'].replace('_', ' ').title()}")
            print(f"   🎨 Recommended Slide Type: {format_analysis['recommended_slide_type'].replace('_', ' ').title()}")
            print(f"   🎯 Confidence: {format_analysis['confidence_score']:.2f}")
            print(f"   📋 Template Match: Slide {format_analysis['template_match']['slide_number'] + 1} ({format_analysis['template_match']['template_type'].replace('_', ' ').title()})")
            print(f"   🎭 Slide Role: {format_analysis['slide_role'].replace('_', ' ').title()}")
            print(f"   📈 Flow Adjustment: {format_analysis['flow_adjustment']:.2f}")
            
            # Generate content for this slide
            slide_content = generate_slide_content(question, relevant_content, slide_metadata, topic, slide_num)
            
            if slide_content:
                all_results.extend(slide_content)
                print(f"   ✅ Generated {len(slide_content)} content items for slide {slide_num + 1}")
            else:
                print(f"   ❌ Failed to generate content for slide {slide_num + 1}")
            
            pbar.update(1)
            time.sleep(0.1)  # Small delay to show progress
    
    # Step 5: Handle other metadata (master, layout, etc.)
    if other_metadata:
        print(f"\n📋 Processing template metadata...")
        with tqdm(total=1, desc="📋 Processing template content", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            # Use simple prompt for template content
            template_prompt = build_simple_prompt(topic, other_metadata, user_content)
            from .ai_content_fetcher import get_ai_content
            template_content = get_ai_content(
                prompt=template_prompt,
                original_contexts=other_metadata,
                user_topic=topic,
                user_content=user_content
            )
            if template_content:
                all_results.extend(template_content)
                print(f"   ✅ Generated {len(template_content)} template content items")
            pbar.update(1)
    
    print(f"\n🎉 Question-based presentation generation completed!")
    print(f"📊 Total content items generated: {len(all_results)}")
    
    return all_results

def analyze_slide_structure(slide_fields):
    """Automatically detect slide structure and content relationships"""
    if not slide_fields:
        return {"type": "unknown", "structure": []}
    
    # Categorize fields by type and position
    headers = []
    body_texts = []
    
    for field in slide_fields:
        role = determine_field_role(field["context"])
        field["role"] = role
        
        if role in ["title", "subtitle"] or field["char_count"] < 80:
            headers.append(field)
        else:
            body_texts.append(field)
    
    # Detect slide patterns
    structure_type = detect_slide_pattern(headers, body_texts)
    
    # Create content groups (headers with their related body text)
    content_groups = create_content_groups(headers, body_texts)
    
    return {
        "type": structure_type,
        "headers": headers,
        "body_texts": body_texts,
        "content_groups": content_groups,
        "total_fields": len(slide_fields)
    }

def detect_slide_pattern(headers, body_texts):
    """Detect the type of slide based on structure"""
    header_count = len(headers)
    body_count = len(body_texts)
    
    if header_count >= 2 and body_count >= 2:
        return "multi_section"  # Multiple sections with headers and content
    elif header_count == 1 and body_count >= 2:
        return "single_topic_detailed"  # One main topic with multiple details
    elif header_count >= 2 and body_count <= 1:
        return "comparison_overview"  # Multiple topics/features comparison
    elif header_count == 1 and body_count == 1:
        return "simple_slide"  # Simple slide with title and content
    else:
        return "complex_layout"  # Complex or unusual layout

def create_content_groups(headers, body_texts):
    """Group headers with their related body content - preserving original order"""
    content_groups = []
    
    if not headers:
        # No headers, treat all as standalone content
        for body in body_texts:
            content_groups.append({
                "header": None,
                "content": [body],
                "relationship": "standalone"
            })
        return content_groups
    
    # DON'T SORT - preserve original order to maintain field mapping
    # Use original order from metadata extraction
    headers_original_order = headers  # Keep original order
    body_original_order = body_texts  # Keep original order
    
    # Simple pairing strategy: match headers with nearby body text
    for i, header in enumerate(headers_original_order):
        group = {
            "header": header,
            "content": [],
            "relationship": "header_content"
        }
        
        # Find body text that should go with this header
        # Look for body text that comes after this header but before the next
        next_header_shape = headers_original_order[i + 1].get("shape", 999) if i + 1 < len(headers_original_order) else 999
        
        for body in body_original_order:
            body_shape = body.get("shape", 0)
            header_shape = header.get("shape", 0)
            
            # Body text belongs to this header if it's positioned after the header but before the next header
            if header_shape < body_shape < next_header_shape:
                group["content"].append(body)
        
        content_groups.append(group)
    
    # Handle orphaned body text (not matched to any header)
    matched_body_shapes = set()
    for group in content_groups:
        for content in group["content"]:
            matched_body_shapes.add(content.get("shape"))
    
    for body in body_original_order:
        if body.get("shape") not in matched_body_shapes:
            content_groups.append({
                "header": None,
                "content": [body],
                "relationship": "orphaned"
            })
    
    return content_groups

def build_contextual_prompt(user_topic, slide_structure):
    """Build intelligent prompts based on slide structure and content relationships"""
    structure_type = slide_structure["type"]
    content_groups = slide_structure["content_groups"]
    
    # Create context-aware instructions based on slide type
    if structure_type == "multi_section":
        context_instruction = f"Create a multi-section slide about '{user_topic}' with related headers and content. Each section should cover a different aspect of {user_topic}, and the content under each header should directly explain or elaborate on that header's topic."
        
    elif structure_type == "comparison_overview":
        context_instruction = f"Create a comparison or overview slide about '{user_topic}' with multiple key points or features. Each header should represent a different aspect, feature, or component of {user_topic}."
        
    elif structure_type == "single_topic_detailed":
        context_instruction = f"Create a detailed slide about '{user_topic}' with one main header and supporting content. The header should introduce the main concept, and the body content should provide detailed explanations, examples, or elaborations."
        
    else:
        context_instruction = f"Create coherent content about '{user_topic}' where all elements work together to explain the topic comprehensively."
    
    # Build section-specific instructions
    section_instructions = []
    for i, group in enumerate(content_groups):
        if group["header"]:
            header_context = clean_context_hint(group["header"]["context"])
            section_instructions.append(f"Section {i+1}: Create a header about {user_topic} (focusing on the {header_context} aspect)")
            
            for j, content in enumerate(group["content"]):
                section_instructions.append(f"  - Content {i+1}.{j+1}: Write body text that directly explains or elaborates on the header above")
        else:
            for j, content in enumerate(group["content"]):
                section_instructions.append(f"Standalone content: Write about {user_topic} (independent section)")
    
    return context_instruction, section_instructions

def clean_context_hint(context_text):
    """Extract meaningful hints from original context while avoiding template text"""
    if not context_text or len(context_text) < 3:
        return "general"
    
    # Remove common template phrases
    template_phrases = ["slidesgo", "template", "presentation", "here's what", "below is", "visit", "click here", "read more"]
    cleaned = context_text.lower()
    for phrase in template_phrases:
        cleaned = cleaned.replace(phrase, "")
    
    # Extract meaningful words (keep only first few words as hint)
    words = cleaned.split()[:3]
    meaningful_words = [w for w in words if len(w) > 2 and w.isalpha()]
    
    if meaningful_words:
        return " ".join(meaningful_words)
    else:
        return "general"

def build_simple_prompt(user_topic, metadata, user_content=""):
    """Simple content formatting prompt - ONLY uses user content"""
    if not user_content:
        raise ValueError("No user content provided for formatting")
    
    # Truncate content if too long to stay within token limits
    max_content_length = 1500
    truncated_content = user_content[:max_content_length]
    if len(user_content) > max_content_length:
        truncated_content += "..."
    
    prompt = (
        f"You are an expert content formatter for PowerPoint presentations about '{user_topic}'. "
        "TASK: Reformat the PROVIDED CONTENT below to fit PowerPoint slide fields.\n\n"
        f"USER'S CONTENT:\n{truncated_content}\n\n"
        "FORMATTING RULES:\n"
        "1. Use ONLY information from the user's content above - DO NOT add any external knowledge\n"
        "2. If user content is insufficient for a field, use relevant parts of user content or leave field minimal\n"
        "3. Headers: Create engaging titles using ONLY user content (4-8 words minimum)\n"
        "4. Body text: Write explanations using ONLY user content (3-5 complete sentences minimum)\n"
        "5. Use available character capacity but prioritize user content accuracy over length\n"
        "6. Create informative content using ONLY the provided user content\n"
        "7. Never generate single words or extremely short phrases\n"
        "8. Maintain exact field order in your response\n"
        "9. DO NOT add any concepts, examples, or information not present in user content\n\n"
        "CRITICAL: Respond ONLY with valid JSON array. No markdown, no explanations.\n"
        "CRITICAL: Generate content for ALL fields provided - do not skip any.\n"
        "CRITICAL: Use ONLY user content - no external knowledge or generation.\n\n"
        "EXAMPLE RESPONSE FORMAT:\n"
        '[\n'
        '  {"source": "slide", "source_index": 0, "slide": 0, "shape": 0, "text": "Machine Learning Fundamentals and Applications"},\n'
        '  {"source": "slide", "source_index": 0, "slide": 0, "shape": 1, "text": "Machine learning algorithms enable computers to learn patterns from data without explicit programming. These systems improve performance through experience and can make predictions on new, unseen data. The field encompasses supervised, unsupervised, and reinforcement learning paradigms."}\n'
        ']\n\n'
        "FIELDS TO FORMAT:\n"
    )
    
    field_specs = []
    for item in metadata:
        field_spec = {
            "source": "template",  # Default for new format
            "source_index": 0,  # Default for new format
            "slide": item.get("slide", -1),
            "shape": item["shape"],
            "text": "",
            "char_count": item.get("char_count", len(item.get("text", "")))
        }
        field_specs.append(field_spec)
    
    prompt += json.dumps(field_specs, indent=2)
    prompt += "\n\nRespond with JSON array only. Fill 'text' field for each item using ONLY user content. NO OTHER TEXT."
    
    return prompt

def build_enhanced_prompt(user_topic, metadata, user_content=""):
    """Enhanced content formatting with header-body relationships - ONLY uses user content"""
    if not user_content:
        raise ValueError("No user content provided for formatting")
    
    # Truncate content for enhanced mode
    max_content_length = 2000  # Increased from 1200
    truncated_content = user_content[:max_content_length]
    if len(user_content) > max_content_length:
        truncated_content += "..."
    
    # Group metadata by slide first
    slides_content = defaultdict(list)
    other_content = []
    
    for item in metadata:
        slide_num = item.get('slide', -1)
        if slide_num >= 0:
            slides_content[slide_num].append(item)
        else:
            other_content.append(item)
    
    prompt = f"""You are an expert content formatter for PowerPoint about '{user_topic}'.
TASK: Format the provided user content to fit PowerPoint slide fields while maintaining header-body relationships.

USER'S CONTENT:
{truncated_content}

CONTENT FORMATTING APPROACH:
- Use ONLY the provided user content as source material
- DO NOT add any external knowledge, concepts, or examples
- Format user content to fit slide fields while preserving meaning
- Maintain thematic consistency between headers and related body text
- If user content is insufficient, use relevant parts or keep content minimal

CRITICAL: Headers and body text must be thematically connected within each slide.
CRITICAL: Generate content for ALL fields provided - do not skip any fields.
CRITICAL: Use ONLY user content - no external knowledge or generation.
CRITICAL: DO NOT add progressive depth, foundational concepts, or expert insights.

CONTENT DISTRIBUTION: Distribute user content across slides based on field requirements.
- Use user content to fill all available fields
- Maintain logical flow and relationships
- Do not add concepts or information not present in user content

RESPONSE FORMAT: JSON array only. No markdown, no explanations, no additional text.

EXAMPLE:
[
  {{"source": "slide", "source_index": 0, "slide": 0, "shape": 0, "text": "Introduction to {user_topic} Systems"}},
  {{"source": "slide", "source_index": 0, "slide": 0, "shape": 1, "text": "Explanation based on user content only. This includes the key points and concepts mentioned in the provided material. The content focuses on the specific information shared by the user."}}
]

"""
    
    field_specs = []
    relationship_instructions = []
    
    # Process each slide to establish header-body relationships for formatting
    for slide_num in sorted(slides_content.keys()):
        slide_items = slides_content[slide_num]
        
        # Sort by spatial position
        sorted_items = sorted(slide_items, key=lambda x: (x.get("positioning", {}).get("top", 0), x.get("positioning", {}).get("left", 0)))
        
        # Identify headers and bodies in this slide
        slide_headers = []
        slide_bodies = []
        
        for item in sorted_items:
            role = determine_field_role(item.get("context", ""))
            item["role"] = role
            
            if role in ["title", "subtitle"] or item["char_count"] < 80:
                slide_headers.append(item)
            else:
                slide_bodies.append(item)
        
        # Create header-body pairs for content formatting
        if slide_headers and slide_bodies:
            prompt += f"SLIDE {slide_num + 1} - Content Distribution:\n"
            prompt += f"  Focus: Format user content to fit slide fields while maintaining relationships\n"
            
            # Assign content sections to header-body pairs
            for i, header in enumerate(slide_headers):
                section_focus = f"Section {i+1} of {user_topic}"
                
                # Find body text that should pair with this header
                related_bodies = []
                header_shape = header.get("shape", 0)
                
                if i + 1 < len(slide_headers):
                    next_header_shape = slide_headers[i + 1].get("shape", 999)
                    related_bodies = [b for b in slide_bodies if header_shape < b.get("shape", 0) < next_header_shape]
                else:
                    related_bodies = [b for b in slide_bodies if b.get("shape", 0) > header_shape]
                
                if not related_bodies and slide_bodies:
                    bodies_per_header = len(slide_bodies) // len(slide_headers)
                    start_idx = i * bodies_per_header
                    end_idx = start_idx + bodies_per_header if i < len(slide_headers) - 1 else len(slide_bodies)
                    related_bodies = slide_bodies[start_idx:end_idx]
                
                # Add relationship instruction for formatting
                relationship_instructions.append(f"Header-Body Pair: '{section_focus}' header with {len(related_bodies)} related body text(s)")
                
                # Add header to field specs
                header_spec = {
                    "source": header["source"],
                    "source_index": header["source_index"],
                    "slide": header.get("slide", -1),
                    "shape": header["shape"],
                    "role": "header",
                    "char_count": header["char_count"],
                    "content_focus": section_focus,
                    "relationship": "header"
                }
                field_specs.append(header_spec)
                
                # Add related bodies to field specs
                for body in related_bodies:
                    body_spec = {
                        "source": body["source"],
                        "source_index": body["source_index"],
                        "slide": body.get("slide", -1),
                        "shape": body["shape"],
                        "role": "body",
                        "char_count": body["char_count"],
                        "content_focus": section_focus,
                        "relationship": f"explains_header_{header['shape']}"
                    }
                    field_specs.append(body_spec)
                
                prompt += f"  - Header: '{section_focus}' → Body: Format user content to explain same topic\n"
            
            prompt += "\n"
        
        else:
            # Handle slides with only headers or only bodies
            for item in sorted_items:
                role = determine_field_role(item.get("context", ""))
                field_spec = {
                    "source": item["source"],
                    "source_index": item["source_index"],
                    "slide": item.get("slide", -1),
                    "shape": item["shape"],
                    "role": role,
                    "char_count": item["char_count"],
                    "content_focus": user_topic,
                    "relationship": "standalone"
                }
                field_specs.append(field_spec)
    
    # Add other content (master, layout)
    for item in other_content:
        role = determine_field_role(item.get("context", ""))
        field_spec = {
            "source": item["source"],
            "source_index": item["source_index"],
            "slide": item.get("slide", -1),
            "shape": item["shape"],
            "role": role,
            "char_count": item["char_count"],
            "content_focus": user_topic,
            "relationship": "template"
        }
        field_specs.append(field_spec)
    
    # Add detailed instructions for content formatting
    prompt += f"""CONTENT FORMATTING RULES:
1. Use ONLY user content as source material - no external knowledge
2. Each header should be engaging and descriptive using user content only
3. Body text should be comprehensive explanations using user content only
4. Match approximate character counts but prioritize user content accuracy
5. Maintain exact field order in response
6. Create thematic consistency between paired headers and bodies
7. DO NOT add progressive depth, foundational concepts, or expert insights
8. If user content is insufficient, use relevant parts or keep minimal

HEADER-BODY RELATIONSHIPS:
{chr(10).join(relationship_instructions)}

FIELDS TO FILL:
{json.dumps(field_specs, indent=2)}

RESPONSE: JSON array only. Fill "text" field for each item using ONLY user content. No other text or formatting.
"""
    
    return prompt

# def generate_slide_subtopics(user_topic, num_headers, slide_number=0, total_slides=1):
#     """Generate specific subtopics for headers based on topic and progressive depth"""
#     topic_lower = user_topic.lower()
#     
#     # Calculate depth level based on slide position
#     depth_level = calculate_depth_level(slide_number, total_slides)
#     
#     # Get progressive subtopics based on topic and depth
#     if "machine learning" in topic_lower or "ml" in topic_lower:
#         subtopics = get_ml_progressive_subtopics(depth_level, num_headers)
#     elif "data structure" in topic_lower or "dsa" in topic_lower:
#         subtopics = get_dsa_progressive_subtopics(depth_level, num_headers)
#     elif "java" in topic_lower and "oop" in topic_lower:
#         subtopics = get_java_oop_progressive_subtopics(depth_level, num_headers)
#     elif "web development" in topic_lower:
#         subtopics = get_web_dev_progressive_subtopics(depth_level, num_headers)
#     elif "artificial intelligence" in topic_lower or "ai" in topic_lower:
#         subtopics = get_ai_progressive_subtopics(depth_level, num_headers)
#     else:
#         # Generic progressive subtopics
#         subtopics = get_generic_progressive_subtopics(user_topic, depth_level, num_headers)
#     
#     # Return the required number of subtopics
#     if num_headers <= len(subtopics):
#         return subtopics[:num_headers]
#     else:
#         # Extend with depth-appropriate topics if needed
#         extended = subtopics.copy()
#         for i in range(len(subtopics), num_headers):
#             extended.append(f"{user_topic} {get_depth_qualifier(depth_level)} {i+1}")
#         return extended

# def calculate_depth_level(slide_number, total_slides):
#     """Calculate depth level (1-5) based on slide position"""
#     if total_slides <= 1:
#         return 3  # Medium depth for single slide
#     
#     # Map slide position to depth level
#     progress = slide_number / (total_slides - 1)  # 0.0 to 1.0
#     
#     if progress <= 0.2:
#         return 1  # Introduction/Overview
#     elif progress <= 0.4:
#         return 2  # Basic concepts
#     elif progress <= 0.6:
#         return 3  # Detailed explanations
#     elif progress <= 0.8:
#         return 4  # Implementation/Advanced
#     else:
#         return 5  # Expert/Future trends

# def get_depth_qualifier(depth_level):
#     """Get qualifier word based on depth level"""
#     qualifiers = {
#         1: "Overview",
#         2: "Fundamentals", 
#         3: "Implementation",
#         4: "Advanced",
#         5: "Expert"
#     }
#     return qualifiers.get(depth_level, "Concepts")

# def get_content_distribution_instruction(depth_level):
#     """Get a specific instruction for content distribution based on depth level"""
#     instructions = {
#         1: "Introduce the main topic briefly and provide a general overview of the topic.",
#         2: "Dive into the core concepts, definitions, and fundamental principles.",
#         3: "Provide detailed explanations, examples, and practical applications of the concepts.",
#         4: "Discuss implementation strategies, best practices, and advanced topics.",
#         5: "Conclude with future trends, emerging technologies, and expert insights."
#     }
#     return instructions.get(depth_level, "Provide a comprehensive and detailed explanation of the topic.")

# def get_ml_progressive_subtopics(depth_level, num_headers):
#     """Get Machine Learning subtopics based on depth level"""
#     depth_topics = {
#         1: ["Machine Learning Introduction", "What is ML", "ML Applications", "Why Use ML"],
#         2: ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "ML Types"],
#         3: ["Linear Regression", "Decision Trees", "Neural Networks", "Support Vector Machines"],
#         4: ["Deep Learning Architecture", "Convolutional Networks", "Recurrent Networks", "Transformer Models"],
#         5: ["AutoML Systems", "Federated Learning", "Quantum ML", "ML Ethics"]
#     }
#     return depth_topics.get(depth_level, depth_topics[3])

# def get_dsa_progressive_subtopics(depth_level, num_headers):
#     """Get Data Structures subtopics based on depth level"""
#     depth_topics = {
#         1: ["Data Structures Introduction", "Why Data Structures", "Memory Management", "Algorithm Basics"],
#         2: ["Arrays", "Linked Lists", "Stacks", "Queues"],
#         3: ["Binary Trees", "Hash Tables", "Heaps", "Graphs"],
#         4: ["AVL Trees", "B-Trees", "Trie Structures", "Advanced Graphs"],
#         5: ["Persistent Data Structures", "Lock-Free Structures", "Cache-Oblivious Algorithms", "Parallel Data Structures"]
#     }
#     return depth_topics.get(depth_level, depth_topics[3])

# def get_java_oop_progressive_subtopics(depth_level, num_headers):
#     """Get Java OOP subtopics based on depth level"""
#     depth_topics = {
#         1: ["Object-Oriented Programming", "Why OOP", "OOP vs Procedural", "Java Introduction"],
#         2: ["Classes and Objects", "Methods and Variables", "Constructors", "Access Modifiers"],
#         3: ["Encapsulation", "Inheritance", "Polymorphism", "Abstraction"],
#         4: ["Interface Design", "Abstract Classes", "Method Overriding", "Design Patterns"],
#         5: ["Reflection API", "Annotations", "Generics Advanced", "Concurrency Patterns"]
#     }
#     return depth_topics.get(depth_level, depth_topics[3])

# def get_web_dev_progressive_subtopics(depth_level, num_headers):
#     """Get Web Development subtopics based on depth level"""
#     depth_topics = {
#         1: ["Web Development Introduction", "Frontend vs Backend", "Web Technologies", "Development Process"],
#         2: ["HTML Structure", "CSS Styling", "JavaScript Basics", "Responsive Design"],
#         3: ["React Components", "Node.js Backend", "Database Integration", "API Development"],
#         4: ["State Management", "Authentication", "Performance Optimization", "Testing Strategies"],
#         5: ["Microservices Architecture", "Serverless Computing", "Progressive Web Apps", "Web3 Integration"]
#     }
#     return depth_topics.get(depth_level, depth_topics[3])

# def get_ai_progressive_subtopics(depth_level, num_headers):
#     """Get Artificial Intelligence subtopics based on depth level"""
#     depth_topics = {
#         1: ["Artificial Intelligence Introduction", "AI History", "AI Applications", "AI vs Human Intelligence"],
#         2: ["Machine Learning", "Natural Language Processing", "Computer Vision", "Robotics"],
#         3: ["Neural Networks", "Deep Learning", "Reinforcement Learning", "Expert Systems"],
#         4: ["Transformer Architecture", "Generative AI", "AI Safety", "Ethical AI"],
#         5: ["Artificial General Intelligence", "Quantum AI", "AI Consciousness", "Future of AI"]
#     }
#     return depth_topics.get(depth_level, depth_topics[3])

# def get_generic_progressive_subtopics(user_topic, depth_level, num_headers):
#     """Get generic progressive subtopics for any topic"""
#     depth_qualifiers = {
#         1: ["Introduction", "Overview", "Basics", "Why Important"],
#         2: ["Fundamentals", "Core Concepts", "Key Principles", "Main Components"],
#         3: ["Implementation", "Detailed Analysis", "Practical Applications", "Case Studies"],
#         4: ["Advanced Techniques", "Best Practices", "Optimization", "Professional Use"],
#         5: ["Future Trends", "Emerging Technologies", "Research Directions", "Expert Insights"]
#     }
#     
#     qualifiers = depth_qualifiers.get(depth_level, depth_qualifiers[3])
#     return [f"{user_topic} {qualifier}" for qualifier in qualifiers]

def build_prompt(user_topic, metadata, use_enhanced=True, user_content=""):
    """Hybrid prompt builder with automatic fallback"""
    try:
        if use_enhanced:
            return build_enhanced_prompt(user_topic, metadata, user_content)
        else:
            return build_simple_prompt(user_topic, metadata, user_content)
    except Exception as e:
        print(f"Enhanced prompt failed, using simple mode: {e}")
        return build_simple_prompt(user_topic, metadata, user_content)

def determine_field_role(context_text):
    """Determine if field is title, subtitle, body, etc. based on context"""
    if len(context_text) < 30:
        return "title"
    elif len(context_text) < 100:
        return "subtitle"
    else:
        return "body"

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 3:
        print("Usage: python prompt_builder.py <topic> <metadata.json>")
    else:
        with open(sys.argv[2], "r") as f:
            mdata = json.load(f)
        print(build_prompt(sys.argv[1], mdata))
