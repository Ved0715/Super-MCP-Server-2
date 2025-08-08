import json
from collections import defaultdict
import openai
import os
from dotenv import load_dotenv
from .content_format_analyzer import content_analyzer

def generate_slide_title(question, content, format_type, topic=""):
    """
    Generate professional slide title using AI based on question, content, and format type
    
    Args:
        question (str): The slide question
        content (str): The slide content
        format_type (str): The detected format (timeline, boxes, table, etc.)
        topic (str): Overall presentation topic
    
    Returns:
        str: Professional slide title
    """
    try:
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            # Fallback if no API key
            return extract_title_fallback(question, topic)
        
        client = openai.OpenAI(api_key=api_key)
        
        # Truncate content if too long
        truncated_content = content[:500] if len(content) > 500 else content
        
        title_prompt = f"""
        Create a professional, concise slide title based on the following information:
        
        QUESTION: {question}
        CONTENT PREVIEW: {truncated_content}
        FORMAT TYPE: {format_type}
        PRESENTATION TOPIC: {topic}
        
        REQUIREMENTS:
        - Create a clear, professional slide title (3-8 words)
        - Focus on the main topic/concept, not the question format
        - Make it presentation-ready and engaging
        - Avoid question words like "What", "How", "Why"
        - Match the content format style:
          * Timeline/Process: Use "Process", "Steps", "Journey", "Evolution"
          * Comparison/Boxes: Use "Comparison", "Analysis", "Overview"  
          * Table/Data: Use "Analysis", "Metrics", "Data", "Statistics"
          * Bullet Points: Use "Key Points", "Overview", "Highlights"
          * Paragraph: Use descriptive topic names
        
        EXAMPLES:
        - Question: "What is the process of PGP key generation?" → Title: "PGP Key Generation Process"
        - Question: "How do algorithms compare?" → Title: "Algorithm Comparison Analysis"
        - Question: "What are the benefits?" → Title: "Key Benefits Overview"
        
        RESPONSE: Return ONLY the title, no explanations or quotes.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": title_prompt}],
            max_tokens=50,
            temperature=0.3
        )
        
        generated_title = response.choices[0].message.content.strip()
        
        # Clean up the title (remove quotes if AI added them)
        generated_title = generated_title.strip('"\'')
        
        # Validate title length (fallback if too long/short)
        if len(generated_title) < 3 or len(generated_title) > 80:
            return extract_title_fallback(question, topic)
        
        return generated_title
        
    except Exception as e:
        print(f"   ⚠️ Title generation failed: {e}, using fallback")
        return extract_title_fallback(question, topic)

def extract_title_fallback(question, topic=""):
    """
    Fallback title extraction using rule-based approach
    """
    import re
    
    # Remove common question starters
    question_clean = re.sub(r'^(what|how|why|when|where|which|who)\s+(is|are|do|does|can|will|would|should)\s+', '', question.lower())
    question_clean = re.sub(r'^(what|how|why|when|where|which|who)\s+', '', question_clean)
    
    # Clean up and capitalize
    title = question_clean.strip()
    if title:
        # Capitalize first letter of each word
        title = ' '.join(word.capitalize() for word in title.split())
        # Limit length
        if len(title) > 50:
            title = title[:47] + "..."
        return title
    
    # Ultimate fallback
    return f"{topic} Overview" if topic else "Slide Overview"

def generate_slide_subheading(question, content, title, content_type, topic):
    """Generate descriptive subheading using AI with enhanced detail requirements"""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        return get_fallback_subheading(content_type, topic)
    
    client = openai.OpenAI(api_key=api_key)
    
    subheading_prompt = f"""
    Create a detailed, descriptive subheading that tells users exactly what this slide content covers.
    
    REQUIREMENTS:
    - 6-15 words describing what the content covers in detail
    - Should be highly specific so users understand exactly what to expect
    - Include key concepts, methods, or data types mentioned in content
    - Must be different from the title: "{title}"
    - Format: {content_type}
    - Topic: {topic}
    
    Question: {question}
    Content preview: {content[:400]}...
    
    Generate only the subheading text, nothing else.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at creating detailed, descriptive slide subheadings."},
                {"role": "user", "content": subheading_prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        
        subheading = response.choices[0].message.content.strip()
        
        # Validate length and word count
        if len(subheading) > 120:
            subheading = subheading[:117] + "..."
        
        word_count = len(subheading.split())
        if word_count < 6 or word_count > 15:
            return get_fallback_subheading(content_type, topic)
            
        return subheading
        
    except Exception as e:
        print(f"   Warning: AI subheading generation failed ({e}), using fallback")
        return get_fallback_subheading(content_type, topic)

def get_fallback_subheading(content_type, topic):
    """Enhanced fallback subheadings with more detailed descriptions"""
    fallbacks = {
        "timeline": "Key developments and milestones in chronological sequence",
        "table": "Structured data comparison with detailed analysis and metrics", 
        "boxes": "Comparative analysis of key concepts with detailed breakdowns",
        "bullet_points": "Essential points and actionable insights with clear explanations",
        "paragraph": "Comprehensive overview with detailed explanations and context"
    }
    
    base_subheading = fallbacks.get(content_type, "Detailed analysis and comprehensive insights")
    
    if topic:
        return f"{topic}: {base_subheading}"
    else:
        return base_subheading

def extract_subheading_fallback(content, format_type):
    """Fallback subheading based on content analysis"""
    content_lower = content.lower()
    
    # Analyze content patterns for meaningful subheading
    if "vs" in content_lower or "comparison" in content_lower:
        return "Detailed comparison and analysis"
    elif "steps" in content_lower or "process" in content_lower:
        return "Step-by-step process guide"
    elif "benefits" in content_lower or "advantages" in content_lower:
        return "Key benefits and value analysis"
    elif "security" in content_lower:
        return "Security features and implementation"
    elif "performance" in content_lower or "%" in content:
        return "Performance metrics and data"
    elif format_type == "timeline":
        return "Sequential stages and key milestones"
    elif format_type == "table":
        return "Data comparison and metrics"
    elif format_type == "boxes":
        return "Component breakdown and features"
    elif format_type == "bullet_points":
        return "Key highlights and important points"
    else:
        return "Comprehensive overview and details"

def generate_presentation_subtitle(topic, user_prompt):
    """
    Generate professional presentation subtitle using AI based on topic and user prompt
    
    Args:
        topic (str): Main presentation topic
        user_prompt (str): User's presentation instructions/prompt
    
    Returns:
        str: Professional subtitle
    """
    try:
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            # Fallback if no API key
            return f"A Comprehensive Overview"
        
        client = openai.OpenAI(api_key=api_key)
        
        subtitle_prompt = f"""
        Create a professional, engaging subtitle for a presentation based on:
        
        MAIN TOPIC: {topic}
        USER INSTRUCTIONS: {user_prompt}
        
        REQUIREMENTS:
        - Create a clear, professional subtitle (4-8 words)
        - Should complement the main topic, not repeat it
        - Make it engaging and informative
        - Focus on the presentation's purpose/scope
        - Examples:
          * Topic: "Machine Learning", Prompt: "explain basics to advanced" → "From Fundamentals to Advanced Applications"
          * Topic: "Marketing Strategy", Prompt: "cover digital trends" → "Digital Trends and Modern Approaches"
          * Topic: "Data Science", Prompt: "practical applications" → "Real-World Applications and Case Studies"
        
        RESPONSE: Return ONLY the subtitle, no explanations or quotes.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": subtitle_prompt}],
            max_tokens=30,
            temperature=0.3
        )
        
        generated_subtitle = response.choices[0].message.content.strip()
        
        # Clean up the subtitle (remove quotes if AI added them)
        generated_subtitle = generated_subtitle.strip('"\'')
        
        # Validate subtitle length (fallback if too long/short)
        if len(generated_subtitle) < 5 or len(generated_subtitle) > 100:
            return f"A Comprehensive Overview"
        
        return generated_subtitle
        
    except Exception as e:
        print(f"   ⚠️ Subtitle generation failed: {e}, using fallback")
        return f"A Comprehensive Overview"

def decompose_user_prompt(user_prompt, num_slides, user_topic=""):
    """
    Break down user prompt into slide-specific questions using AI
    Note: num_slides should be the total requested slides. Function will generate (num_slides-2) content questions
    to account for title slide and thank you slide.
    """
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found")
    
    client = openai.OpenAI(api_key=api_key)
    
    # Calculate content slides (total - title slide - thank you slide)
    content_slides = max(1, num_slides - 2)  # Ensure at least 1 content slide
    
    decomposition_prompt = f"""
    You are an expert presentation planner. Break down this presentation prompt into {content_slides} specific questions.
    
    NOTE: This is for content slides only. A title slide and thank you slide will be added separately.

    PRESENTATION TOPIC: {user_topic}
    USER PROMPT: {user_prompt}
    TOTAL SLIDES REQUESTED: {num_slides}
    CONTENT SLIDES TO GENERATE: {content_slides}

    TASK: Create {content_slides} specific questions that will guide content for each content slide.
    
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

    Generate {content_slides} questions with varied formats:
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
        
        # Ensure we have the right number of content questions
        if len(questions) != content_slides:
            # Pad or truncate to match content_slides
            if len(questions) < content_slides:
                for i in range(len(questions), content_slides):
                    questions.append(f"Additional aspect of {user_topic}")
            else:
                questions = questions[:content_slides]
        
        return questions
        
    except Exception as e:
        print(f"Error decomposing prompt: {e}")
        # Fallback: generate generic content questions (no title/thank you)
        fallback_questions = []
        for i in range(content_slides):
            if i == 0:
                fallback_questions.append(f"What is {user_topic}?")
            elif i == content_slides - 1:
                fallback_questions.append(f"What are the key takeaways about {user_topic}?")
            else:
                fallback_questions.append(f"What is aspect {i+1} of {user_topic}?")
        return fallback_questions

def decompose_user_prompt_content_aware(user_prompt, num_slides, user_topic="", user_content=""):
    """
    Content-aware question decomposition: Analyze content capabilities first, then generate questions
    that align with both user prompt intent AND available content to prevent 'content not available' errors
    """
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found")
    
    client = openai.OpenAI(api_key=api_key)
    
    # Calculate content slides (total - title slide - thank you slide)
    content_slides = max(1, num_slides - 2)
    
    # Truncate content if too long for analysis
    max_content_length = 4000
    truncated_content = user_content[:max_content_length]
    if len(user_content) > max_content_length:
        truncated_content += "..."
    
    # Step 1: Analyze content capabilities
    content_analysis_prompt = f"""
    Analyze this content to understand what information is available and what questions can be answered.

    CONTENT TO ANALYZE:
    {truncated_content}

    TASK: Identify the main themes, topics, and types of information present in this content.
    
    REQUIREMENTS:
    - List the key topics and themes covered
    - Identify what types of information are available (explanations, data, processes, comparisons, features, algorithms, etc.)
    - Note specific details, examples, or data points present
    - Identify comparative content (X vs Y, different types, features, algorithms, methods)
    - Determine what aspects of {user_topic} are well-covered vs missing
    
    RESPONSE FORMAT: Return a JSON object with:
    {{
        "main_themes": ["theme1", "theme2", ...],
        "available_information_types": ["explanations", "data", "processes", "examples", "comparisons", "features", ...],
        "specific_topics_covered": ["topic1", "topic2", ...],
        "comparative_content": ["X vs Y", "different algorithms", "feature comparisons", ...],
        "content_strengths": ["what the content does well"],
        "content_gaps": ["what might be missing"]
    }}
    """
    
    try:
        content_analysis_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content_analysis_prompt}],
            max_tokens=1000,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        content_capabilities = json.loads(content_analysis_response.choices[0].message.content)
        
        # Step 2: Generate questions that align with both user prompt and content capabilities
        question_generation_prompt = f"""
        Generate {content_slides} presentation questions that align with BOTH the user's intent AND the available content.

        USER PROMPT: {user_prompt}
        TOPIC: {user_topic}
        CONTENT CAPABILITIES: {json.dumps(content_capabilities, indent=2)}
        
        TASK: Create {content_slides} specific questions that:
        1. Follow the user's presentation intent from their prompt
        2. Can be answered using the available content (based on content analysis above)
        3. Create a logical presentation flow
        4. Use varied question types for different slide formats
        5. Generate table questions when comparative_content or structured information is available
        
        QUESTION TYPE VARIETY (use varied mix):
        - "What are..." for lists and bullet points
        - EXACT TABLE PATTERNS: "Compare [A] vs [B]" or "Statistics on [topic]" or "Data about [topic]" for tables
        - HISTORICAL TIMELINE PATTERNS: "History of [topic]" or "How has [topic] evolved over time" or "Development of [topic] since its inception"
        - PROCESS TIMELINE PATTERNS: "What is the process of..." or "Steps to..." or "How to..." for process flows
        - "How does...work" or "Why is..." for paragraph/explanatory
        - "How do...compare" or "What are the differences..." for comparison/boxes
        
        IMPORTANT: Use EXACT wording for table questions:
        ✅ "Compare PGP vs GPG security methods"  
        ✅ "Statistics on PGP adoption rates"
        ✅ "Data about encryption performance"
        ❌ "How do PGP and GPG compare?" (this triggers boxes, not tables)
        
        CRITICAL: Only ask questions that the content can actually answer. If the content lacks certain information, adjust the question or skip that aspect.
        
        RESPONSE FORMAT: Return a JSON object with a "questions" array.
        Example: {{"questions": ["What is the main concept covered?", "History of PGP encryption development", "Compare PGP vs GPG encryption methods", "Statistics on PGP usage worldwide"]}}
        
        Generate {content_slides} answerable questions:
        """
        
        questions_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question_generation_prompt}],
            max_tokens=1000,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        raw_response = questions_response.choices[0].message.content
        print("   🔍 AI Response:", raw_response[:200] + "...")
        
        questions_data = json.loads(raw_response)
        
        # Handle different response formats with better debugging
        if isinstance(questions_data, dict):
            if "questions" in questions_data:
                questions = questions_data["questions"]
                print(f"   ✅ Found questions array with {len(questions)} questions")
            else:
                # Fallback: try to extract values or keys that look like questions
                all_values = list(questions_data.values())
                if all_values and isinstance(all_values[0], list):
                    questions = all_values[0]  # Take first list found
                elif all_values and isinstance(all_values[0], str):
                    questions = all_values  # All values are strings
                else:
                    print("   ⚠️  Unexpected dict format:", list(questions_data.keys()))
                    questions = list(questions_data.values())
                print(f"   🔄 Extracted {len(questions)} questions from dict values")
        elif isinstance(questions_data, list):
            questions = questions_data
            print(f"   ✅ Found direct list with {len(questions)} questions")
        else:
            print("   ❌ Unexpected response type:", type(questions_data))
            raise ValueError(f"Unexpected response format: {type(questions_data)}")
        
        # Ensure we have the right number of questions
        if len(questions) != content_slides:
            if len(questions) < content_slides:
                for i in range(len(questions), content_slides):
                    questions.append(f"What additional insights about {user_topic} are available?")
            else:
                questions = questions[:content_slides]
        
        print(f"   ✅ Content-aware generation: {len(questions)} questions aligned with available content")
        return questions
        
    except Exception as e:
        print(f"   ❌ Content-aware generation failed: {e}")
        print(f"   🔄 Falling back to standard question generation...")
        # Fallback to original method
        return decompose_user_prompt(user_prompt, num_slides, user_topic)

def regenerate_content_for_uniqueness_format_aware(question, user_content, topic, previous_slides_summaries, slide_num, format_type):
    """
    Format-aware content regeneration that preserves the structure needed for specific slide formats
    """
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return extract_relevant_content(user_content, question, topic)  # fallback
    
    client = openai.OpenAI(api_key=api_key)
    
    # Extract previous content summaries (handle both old string format and new dict format)
    previous_summaries_text = ""
    for i, slide_info in enumerate(previous_slides_summaries):
        if isinstance(slide_info, dict):
            summary = slide_info.get('content_summary', '')
            slide_format = slide_info.get('format', 'unknown')
            previous_summaries_text += f"Slide {i+1} ({slide_format}): {summary}\n"
        else:
            # Backward compatibility with old string format
            previous_summaries_text += f"Slide {i+1}: {slide_info}\n"
    
    # Format-specific regeneration prompts
    format_specific_instructions = {
        'table': """
        CRITICAL: This question requires TABULAR/COMPARISON data. You must preserve comparison structure.
        - Maintain A vs B comparison format if present
        - Keep numerical data, statistics, or measurable differences
        - Ensure data can be organized into rows and columns
        - Focus on different comparison criteria while preserving table structure
        Example: If previous slide compared "cost and features", find data about "performance and security"
        """,
        'timeline': """
        CRITICAL: This question requires SEQUENTIAL/CHRONOLOGICAL data. You must preserve timeline structure.
        - Maintain chronological order or step-by-step progression
        - Keep time markers, sequence indicators, or process stages
        - Ensure content can be organized into timeline format
        - Focus on different time periods or process phases while preserving timeline flow
        """,
        'boxes': """
        CRITICAL: This question requires CATEGORICAL/COMPARISON data. You must preserve box structure.
        - Maintain distinct categories or comparison items
        - Keep structured groupings of related information
        - Ensure content can be organized into separate boxes/categories
        - Focus on different categories while preserving comparison structure
        """,
        'bullet_points': """
        CRITICAL: This question requires LIST/ENUMERATION data. You must preserve list structure.
        - Maintain list format with distinct items
        - Keep enumerable points or features
        - Ensure content can be organized into bullet points
        - Focus on different aspects while preserving list structure
        """,
        'paragraph': """
        This question requires explanatory content. Focus on different explanatory aspects.
        - Provide detailed explanations from different angles
        - Focus on aspects not covered in previous slides
        - Maintain comprehensive explanatory format
        """
    }
    
    format_instructions = format_specific_instructions.get(format_type, format_specific_instructions['paragraph'])
    
    unique_content_prompt = f"""
    Extract content that answers the question while avoiding repetition of previous slides.
    
    QUESTION: {question}
    TOPIC: {topic}
    SLIDE NUMBER: {slide_num}
    FORMAT TYPE: {format_type.upper()}
    
    PREVIOUS SLIDES ALREADY COVERED:
    {previous_summaries_text}
    
    USER CONTENT:
    {user_content[:3000]}...
    
    FORMAT-SPECIFIC REQUIREMENTS:
    {format_instructions}
    
    TASK: Find information that answers the question while:
    1. Preserving the format structure required by the question type
    2. Avoiding repetition of topics already covered in previous slides
    3. Finding complementary aspects that enhance rather than duplicate previous content
    4. Ensuring the content can still properly answer the original question
    
    CRITICAL: The regenerated content MUST still be suitable for {format_type} format. 
    Do not sacrifice format requirements for uniqueness.
    
    RESPONSE: Provide content that maintains {format_type} structure while offering unique insights.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": unique_content_prompt}],
            max_tokens=1000,
            temperature=0.4  # Slightly higher for more creative alternatives
        )
        
        regenerated_content = response.choices[0].message.content.strip()
        print(f"   ✅ Format-preserving regeneration completed for {format_type} format")
        return regenerated_content
        
    except Exception as e:
        print(f"   ❌ Format-aware regeneration failed: {e}, using original content")
        return extract_relevant_content(user_content, question, topic)

def analyze_slide_content_similarity_format_aware(new_slide_content, previous_slides_summaries, topic, format_type, current_question):
    """
    Enhanced format-aware similarity analysis that understands different slide formats
    and focuses on content uniqueness while preserving format structure
    """
    if not previous_slides_summaries:
        return {"is_similar": False, "similarity_score": 0.0, "recommendation": "first_slide"}
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return {"is_similar": False, "similarity_score": 0.5, "recommendation": "api_unavailable"}
    
    client = openai.OpenAI(api_key=api_key)
    
    # Build context-rich comparison data
    previous_summaries_text = ""
    for i, slide_info in enumerate(previous_slides_summaries):
        if isinstance(slide_info, dict):
            summary = slide_info.get('content_summary', '')
            slide_format = slide_info.get('format', 'unknown')
            question_type = slide_info.get('question_type', 'general')
            previous_summaries_text += f"Slide {i+1} ({slide_format}, {question_type}): {summary}\n"
        else:
            # Backward compatibility with old string format
            previous_summaries_text += f"Slide {i+1}: {slide_info}\n"
    
    # Format-specific similarity analysis
    format_specific_guidance = {
        'table': """
        SPECIAL INSTRUCTIONS FOR TABLE SIMILARITY:
        - Tables comparing different aspects (cost vs performance vs security) are UNIQUE
        - Tables with the same subjects but different comparison criteria are UNIQUE  
        - Focus on the comparison criteria and data points, not the subjects being compared
        - Example: "PGP vs GPG cost" is different from "PGP vs GPG performance"
        - Structural similarity (having columns/rows) does not mean content similarity
        """,
        'timeline': """
        SPECIAL INSTRUCTIONS FOR TIMELINE SIMILARITY:
        - Timelines covering different time periods are UNIQUE
        - Timelines with different focus areas (technical vs historical vs adoption) are UNIQUE
        - Focus on the time periods and events covered, not the general topic
        - Sequential structure similarity does not mean content similarity
        """,
        'boxes': """
        SPECIAL INSTRUCTIONS FOR BOXES SIMILARITY:
        - Boxes comparing different categories or aspects are UNIQUE
        - Focus on the specific categories and features being compared
        - Similar structure does not mean content similarity
        """,
        'bullet_points': """
        SPECIAL INSTRUCTIONS FOR BULLET POINTS SIMILARITY:
        - Lists covering different aspects of the same topic are UNIQUE
        - Focus on the specific points and details covered
        """,
        'paragraph': """
        SPECIAL INSTRUCTIONS FOR PARAGRAPH SIMILARITY:
        - Focus on the explanatory content and specific details
        - Different explanatory angles on the same topic are UNIQUE
        """
    }
    
    format_guidance = format_specific_guidance.get(format_type, format_specific_guidance['paragraph'])
    
    similarity_prompt = f"""
    Analyze if this new slide content is too similar to previous slides, considering format context.

    CURRENT SLIDE:
    Format: {format_type.upper()}
    Question: {current_question}
    Content: {new_slide_content[:800]}...
    
    TOPIC: {topic}
    
    PREVIOUS SLIDES SUMMARY:
    {previous_summaries_text}
    
    {format_guidance}
    
    ANALYSIS TASK:
    Determine if this new slide offers substantially different information or if it duplicates previous content.
    
    KEY CONSIDERATIONS:
    - Same format + same comparison criteria = SIMILAR
    - Same format + different comparison criteria = UNIQUE  
    - Different format + same topic = UNIQUE
    - Focus on INFORMATION VALUE, not structural patterns
    - Consider the QUESTION being answered - different questions = likely unique
    
    RESPONSE FORMAT: Return a JSON object:
    {{
        "similarity_score": 0.0-1.0,
        "is_similar": true/false,
        "main_overlap_reasons": ["reason1", "reason2"],
        "recommendation": "unique|similar|very_similar",
        "format_analysis": "how format affects similarity assessment"
    }}
    
    Consider similar if > 80% information overlap for structured formats, > 70% for others.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": similarity_prompt}],
            max_tokens=400,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "is_similar": result.get("is_similar", False),
            "similarity_score": result.get("similarity_score", 0.0),
            "recommendation": result.get("recommendation", "unknown"),
            "main_overlap_reasons": result.get("main_overlap_reasons", []),
            "format_analysis": result.get("format_analysis", "")
        }
        
    except Exception as e:
        print(f"   ❌ Format-aware similarity analysis failed: {e}")
        # Fallback to conservative approach
        return {"is_similar": False, "similarity_score": 0.4, "recommendation": "analysis_failed"}

def extract_display_content_from_slide(slide_content):
    """
    Extract the actual text content that will be displayed in the slide
    from the generated slide_content structure
    """
    display_texts = []
    
    for item in slide_content:
        if isinstance(item, dict) and 'text' in item:
            display_texts.append(item['text'])
        elif isinstance(item, str):
            display_texts.append(item)
    
    return " | ".join(display_texts)  # Combine all display text

def analyze_slide_content_similarity(new_slide_content, previous_slides_summaries, topic):
    """
    Format-aware similarity analysis for slide content
    Returns similarity score and recommendation
    """
    if not previous_slides_summaries:
        return {"is_similar": False, "similarity_score": 0.0, "recommendation": "first_slide"}
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return {"is_similar": False, "similarity_score": 0.5, "recommendation": "api_unavailable"}
    
    client = openai.OpenAI(api_key=api_key)
    
    # Handle both old string format and new dict format for backward compatibility
    previous_summaries_text = ""
    for i, slide_info in enumerate(previous_slides_summaries):
        if isinstance(slide_info, dict):
            summary = slide_info.get('content_summary', '')
            slide_format = slide_info.get('format', 'unknown')
            question_type = slide_info.get('question_type', 'general')
            previous_summaries_text += f"Slide {i+1} ({slide_format}, {question_type}): {summary}\n"
        else:
            # Backward compatibility with old string format
            previous_summaries_text += f"Slide {i+1}: {slide_info}\n"
    
    similarity_prompt = f"""
    Analyze if this new slide content is too similar to previous slides in the presentation.
    Consider format context when determining similarity.

    TOPIC: {topic}
    
    PREVIOUS SLIDES SUMMARY (with format context):
    {previous_summaries_text}
    
    NEW SLIDE CONTENT:
    {new_slide_content[:500]}...
    
    TASK: Determine if the new slide covers substantially different aspects or if it's repetitive.
    
    IMPORTANT CONSIDERATIONS:
    - Different slide formats can naturally cover similar topics with different perspectives
    - Table slides comparing A vs B are different from paragraph slides explaining A or B
    - Timeline slides about topic X are different from bullet slides listing features of X
    - Comparison slides (table/boxes) can have structural similarity while being content-different
    - Focus on CONTENT overlap, not structural/format similarities
    
    RESPONSE FORMAT: Return a JSON object:
    {{
        "similarity_score": 0.0-1.0,
        "is_similar": true/false,
        "main_topic_overlap": ["overlapping_concept1", "overlapping_concept2"],
        "recommendation": "unique|similar|very_similar",
        "format_consideration": "explanation of how format affects similarity assessment"
    }}
    
    Consider similar if > 70% CONTENT topic overlap (ignoring format structural similarities).
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": similarity_prompt}],
            max_tokens=300,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"   ❌ Similarity analysis failed: {e}")
        return {"is_similar": False, "similarity_score": 0.5, "recommendation": "analysis_failed"}

def regenerate_content_for_uniqueness(question, user_content, topic, previous_slides_summaries, slide_num):
    """
    Regenerate content with explicit instructions to avoid similarity to previous slides
    """
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return extract_relevant_content(user_content, question, topic)  # fallback
    
    client = openai.OpenAI(api_key=api_key)
    
    previous_summaries_text = "\n".join([f"Slide {i+1}: {summary}" for i, summary in enumerate(previous_slides_summaries)])
    
    unique_content_prompt = f"""
    Extract content that focuses on DIFFERENT aspects than previous slides.

    QUESTION: {question}
    TOPIC: {topic}
    SLIDE NUMBER: {slide_num}
    
    PREVIOUS SLIDES ALREADY COVERED:
    {previous_summaries_text}
    
    USER CONTENT:
    {user_content[:3000]}...
    
    TASK: Find information that answers the question while avoiding topics already covered in previous slides.
    
    REQUIREMENTS:
    - Focus on UNIQUE aspects not yet covered
    - Avoid repeating concepts from previous slides
    - Find different angles, examples, or details
    - If similar concepts must be mentioned, provide different perspectives
    - Ensure content is substantially different from previous slides
    
    RESPONSE: Provide unique content that complements rather than repeats previous slides.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": unique_content_prompt}],
            max_tokens=1500,
            temperature=0.3
        )
        
        unique_content = response.choices[0].message.content.strip()
        print(f"   ✅ Regenerated unique content for slide {slide_num}")
        return unique_content
        
    except Exception as e:
        print(f"   ❌ Unique content generation failed: {e}")
        return extract_relevant_content(user_content, question, topic)  # fallback

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
    
    # Step 1: Content-aware question decomposition
    with tqdm(total=1, desc="🤖 Analyzing content + generating questions", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        slide_questions = decompose_user_prompt_content_aware(user_prompt, num_slides, topic, user_content)
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
    previous_slides_display_content = []  # Track ACTUAL slide display content
    
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
                # Extract the actual display text from slide_content
                display_content = extract_display_content_from_slide(slide_content)
                
                # Check similarity with previous slides' display content
                if previous_slides_display_content:
                    similarity_analysis = analyze_slide_content_similarity(display_content, previous_slides_display_content, topic)
                    print(f"   🔍 Display content similarity: {similarity_analysis['recommendation']} (score: {similarity_analysis['similarity_score']:.2f})")
                    
                    if similarity_analysis['is_similar'] and similarity_analysis['similarity_score'] > 0.7:
                        print(f"   🔄 Display content too similar, regenerating...")
                        # Regenerate with uniqueness constraints
                        relevant_content = regenerate_content_for_uniqueness(question, user_content, topic, previous_slides_display_content, slide_num + 1)
                        slide_content = generate_slide_content(question, relevant_content, slide_metadata, topic, slide_num)
                        display_content = extract_display_content_from_slide(slide_content)
                
                # Store this slide's display content for future comparisons
                previous_slides_display_content.append(display_content)
                
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
