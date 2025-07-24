from .extract_metadata import extract_all_metadata, is_thank_you_slide
from .prompt_builder import build_prompt
from .ai_content_fetcher import get_ai_content, create_enhanced_fallback_content
from .replace_content import replace_with_ai_content
from pptx import Presentation
import os
import time

# Global variable to store user content
USER_CONTENT = ""

def get_user_content_and_topic():
    """Get presentation content and topic from user input"""
    print("=== AI PowerPoint Content Formatter ===")
    print("Enter your complete content to be formatted for the presentation:")
    print("(Paste your content, then press Enter twice to finish)")
    print("-" * 50)
    
    content_lines = []
    empty_line_count = 0
    
    while True:
        try:
            line = input()
            if line.strip() == "":
                empty_line_count += 1
                if empty_line_count >= 2:
                    break
                content_lines.append(line)
            else:
                empty_line_count = 0
                content_lines.append(line)
        except EOFError:
            break
    
    # Remove trailing empty lines
    while content_lines and content_lines[-1].strip() == "":
        content_lines.pop()
    
    user_content = "\n".join(content_lines).strip()
    
    if not user_content:
        print("❌ No content provided. Please try again.")
        return get_user_content_and_topic()
    
    # Get topic title for context
    topic_title = input("\nWhat's the main topic/title for this presentation? ").strip()
    if not topic_title:
        topic_title = "Presentation"
    
    return topic_title, user_content


def get_template_path():
    """Get the single template path"""
    template_path = "templates/templates/test_slide.pptx"
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path


def generate_output_filename(topic):
    """Generate output filename based on topic"""
    # Clean topic for filename (remove special characters)
    clean_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
    clean_topic = clean_topic.replace(' ', '_')
    return f"presentations/{clean_topic}_presentation.pptx"


def batch_metadata_by_token_estimate(metadata, max_tokens_per_batch=4000):
    """Batch metadata by estimated token count"""
    batches = []
    current_batch = []
    current_tokens = 0
    
    for field in metadata:
        approx_tokens = len(str(field)) // 4 + field["char_count"] // 4 + 20  # Extra buffer
        if current_tokens + approx_tokens > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = [field]
            current_tokens = approx_tokens
        else:
            current_batch.append(field)
            current_tokens += approx_tokens
    
    if current_batch:
        batches.append(current_batch)
    return batches


def generate_presentation_api(content, topic):
    """
    API function to generate PowerPoint presentation
    
    Args:
        content (str): The content to format for presentation
        topic (str): Main topic/title for presentation
    
    Returns:
        str: Output filename of generated presentation
    """
    try:
        # Get template and output paths
        template_path = get_template_path()
        output_path = generate_output_filename(topic)
        
        # Extract metadata from template
        print(f"🚀 Starting AI-powered content formatting...")
        all_metadata = extract_all_metadata(template_path)
        
        # Filter out thank you slides
        metadata = []
        skipped_slides = []
        
        # Load presentation to check slides
        prs = Presentation(template_path)
        
        for item in all_metadata:
            slide_idx = item.get('slide', 0)
            if slide_idx < len(prs.slides) and is_thank_you_slide(prs.slides[slide_idx]):
                skipped_slides.append(slide_idx + 1)  # 1-indexed for display
            else:
                metadata.append(item)
        
        print(f"✅ Extracted {len(metadata)} text fields to replace")
        if skipped_slides:
            print(f"🔒 Preserved slides: {', '.join(map(str, skipped_slides))} (thank you/closing slides)")
        
        # Process in batches to avoid overwhelming the AI
        batch_size = 20  # Reduced from 40 for better AI completion
        batches = [metadata[i:i + batch_size] for i in range(0, len(metadata), batch_size)]
        all_ai_results = []
        total_batches = len(batches)
        
        print(f"📦 Processing {total_batches} batches...")
        
        for batch_index, batch in enumerate(batches):
            print(f"🔄 Processing batch {batch_index + 1} of {total_batches} ({len(batch)} fields)...")
            
            success = False
            for attempt in range(3):  # Increased attempts
                try:
                    # Use enhanced mode for first attempt, simple mode for retries
                    use_enhanced = attempt == 0
                    prompt = build_prompt(topic, batch, use_enhanced=use_enhanced, user_content=content)
                    ai_results = get_ai_content(prompt, batch, user_topic=topic, use_enhanced=use_enhanced, user_content=content)
                    
                    # Validate that we got results for all fields in the batch
                    if len(ai_results) < len(batch):
                        print(f"   ❌ CRITICAL: AI failed to generate all items. Expected {len(batch)}, got {len(ai_results)}")
                        print(f"   🔄 This should not happen with the new retry logic!")
                        # The enhanced AI retry logic in get_ai_content should handle missing fields
                    
                    all_ai_results.extend(ai_results)
                    success = True
                    print(f"   ✅ Batch {batch_index + 1} completed successfully ({len(ai_results)} fields processed)")
                    break
                except Exception as e:
                    print(f"   ❌ Batch {batch_index + 1} failed (attempt {attempt+1}): {e}")
                    if attempt < 2:
                        print(f"   🔄 Retrying with simpler prompt...")
                        time.sleep(2)  # Brief pause between retries
            
            if not success:
                print(f"   ⚠️ Batch {batch_index+1} permanently skipped. See logs.")
        
        print(f"\n📊 Generated content for {len(all_ai_results)} fields")
        
        # Replace content in PowerPoint
        print("📝 Replacing content in PowerPoint template...")
        replace_with_ai_content(template_path, all_ai_results, output_path)
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error generating presentation: {e}")
        raise


def main():
    """CLI version for interactive use"""
    print("=== AI PowerPoint Content Formatter ===")
    
    # Get user inputs
    user_topic, user_content = get_user_content_and_topic()
    template_path = get_template_path()
    output_path = generate_output_filename(user_topic)
    
    # Display processing summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"📝 Topic: {user_topic}")
    print(f"📋 Template: test_slide.pptx")
    print(f"💾 Output: {output_path}")
    print(f"📄 Content: {len(user_content)} characters provided")
    print(f"🔄 Mode: Content Formatting")
    print("="*50)
    
    # Confirm processing
    proceed = input("\nProceed with processing? (y/n): ").lower().strip()
    if proceed != 'y':
        print("❌ Processing cancelled.")
        return
    
    try:
        # Use the API function for processing
        output_file = generate_presentation_api(user_content, user_topic)
        
        print(f"\n🎉 FORMATTING COMPLETED! 🎉")
        print(f"📁 Output saved as: {output_file}")
        print(f"📊 Topic: {user_topic}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check the logs above for details.")


if __name__ == "__main__":
    main()
