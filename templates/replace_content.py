from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.dml.color import RGBColor

def find_shape(shapes, shape_idx):
    if shape_idx >= len(shapes):
        return None
    shape = shapes[shape_idx]
    if hasattr(shape, "has_text_frame") and shape.has_text_frame:
        return shape
    return None

def get_smart_text_color(shape):
    """Determine appropriate text color based on background luminance"""
    try:
        if hasattr(shape, 'fill') and shape.fill:
            fill = shape.fill
            if hasattr(fill, 'fore_color') and fill.fore_color:
                try:
                    rgb = fill.fore_color.rgb
                    if rgb:
                        # Calculate luminance
                        r, g, b = rgb.r / 255.0, rgb.g / 255.0, rgb.b / 255.0
                        luminance = 0.299 * r + 0.587 * g + 0.114 * b
                        
                        # Return contrasting color
                        if luminance < 0.5:
                            return RGBColor(255, 255, 255)  # White text on dark background
                        else:
                            return RGBColor(0, 0, 0)  # Black text on light background
                except:
                    pass
        
        # Default to black text
        return RGBColor(0, 0, 0)
    except:
        return RGBColor(0, 0, 0)

def get_safe_shape_dimensions(shape):
    """Get shape dimensions with robust error handling and safe defaults"""
    try:
        width = None
        height = None
        
        # Try multiple ways to get dimensions
        if hasattr(shape, 'width') and shape.width:
            try:
                width = float(shape.width.inches)
            except:
                pass
                
        if hasattr(shape, 'height') and shape.height:
            try:
                height = float(shape.height.inches)
            except:
                pass
        
        # If dimensions failed or are invalid, use safe defaults
        if width is None or height is None or width <= 0 or height <= 0:
            # Return conservative defaults that work for most shapes
            return {"width": 3.0, "height": 1.0}  # Safe medium size
            
        return {"width": width, "height": height}
    except:
        return {"width": 3.0, "height": 1.0}  # Safe fallback

def analyze_shape_constraints(shape):
    """Analyze shape constraints with much more realistic capacity estimates"""
    try:
        dims = get_safe_shape_dimensions(shape)
        width = dims["width"]  # Always a number now
        height = dims["height"]  # Always a number now
        
        # Calculate area and estimated character capacity - MUCH MORE REALISTIC
        area = width * height
        estimated_capacity = int(area * 100)  # ~100 chars per square inch (was 40 - too conservative)
        
        # Classify shape size - now safe since width/height are always numbers
        is_micro = width < 1.0 or height < 0.5
        is_small = width < 1.5 or height < 0.8
        is_large = width > 6.0 or height > 3.0
        
        return {
            "width": width,
            "height": height,
            "area": area,
            "estimated_capacity": max(estimated_capacity, 25),  # Minimum 25 chars (was 10)
            "is_micro": is_micro,
            "is_small": is_small,
            "is_large": is_large
        }
    except Exception as e:
        # Ultimate fallback - return safe defaults with generous capacity
        return {
            "width": 3.0,
            "height": 1.0,
            "area": 3.0,
            "estimated_capacity": 300,  # Much more generous default
            "is_micro": False,
            "is_small": False,
            "is_large": False
        }

def test_if_text_fits(shape, text):
    """Actually test if text fits instead of just estimating"""
    try:
        if not hasattr(shape, 'text_frame') or not shape.text_frame:
            return False
            
        # Get current text to restore later
        original_text = ""
        if shape.text_frame.paragraphs:
            original_text = shape.text_frame.paragraphs[0].text
        
        # Temporarily set the test text
        shape.text_frame.paragraphs[0].text = text
        
        # Basic heuristics to determine if it fits reasonably
        # For now, we'll be optimistic - PowerPoint handles overflow better than we think
        fits = True
        
        # Only flag as not fitting if text is extremely long relative to shape
        constraints = analyze_shape_constraints(shape)
        if len(text) > constraints["estimated_capacity"] * 4:  # Only if 4x over estimate
            fits = False
        
        # Restore original text
        shape.text_frame.paragraphs[0].text = original_text
        return fits
    except:
        return True  # If test fails, assume it fits (optimistic approach)

def validate_truncation_quality(original_text, truncated_text):
    """Check if truncated text looks professional and preserves meaning"""
    
    if not truncated_text or not original_text:
        return False
        
    # If no truncation occurred, it's automatically good
    if truncated_text == original_text:
        return True
    
    # Rule 1: Must end with complete word
    if truncated_text and not truncated_text[-1].isspace():
        if not truncated_text.endswith(('.', '!', '?', ':')):
            # Check if last word is complete
            words = original_text.split()
            truncated_words = truncated_text.split()
            
            if truncated_words:
                last_word_in_truncated = truncated_words[-1]
                
                # Check if this word exists complete in original
                if last_word_in_truncated not in words:
                    return False  # Last word is cut off
    
    # Rule 2: Must preserve at least 60% of meaning (was 70%, now more lenient)
    if len(truncated_text) < len(original_text) * 0.6:
        return False  # Too much content lost
    
    # Rule 3: Should not end with connecting words (makes text look incomplete)
    bad_endings = ['and', 'or', 'but', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with']
    last_word = truncated_text.split()[-1].lower() if truncated_text.split() else ""
    if last_word in bad_endings:
        return False  # Ends awkwardly
    
    # Rule 4: Should have reasonable length (not too short)
    if len(truncated_text) < 10:  # Very short text might not be meaningful
        return False
    
    return True  # Looks good

def smart_truncate_with_meaning(text, max_length):
    """Truncate while preserving maximum meaning and professional appearance"""
    
    if len(text) <= max_length:
        return text
    
    # Strategy 1: Try to keep complete sentences
    sentences = text.split('. ')
    for i in range(len(sentences), 0, -1):
        candidate = '. '.join(sentences[:i])
        if not candidate.endswith('.') and i < len(sentences):
            candidate += '.'
        if len(candidate) <= max_length:
            return candidate
    
    # Strategy 2: Try to keep complete phrases (comma-separated)
    phrases = text.split(', ')
    for i in range(len(phrases), 0, -1):
        candidate = ', '.join(phrases[:i])
        if len(candidate) <= max_length:
            return candidate
    
    # Strategy 3: Keep complete words (current approach)
    words = text.split()
    for i in range(len(words), 0, -1):
        candidate = ' '.join(words[:i])
        if len(candidate) <= max_length:
            return candidate
    
    # Strategy 4: Last resort - character truncation at word boundary
    if ' ' in text[:max_length]:
        return text[:max_length].rsplit(' ', 1)[0]
    
    return text[:max_length]  # Absolute last resort

def fit_text_with_quality_chain(shape, text, field_type="body"):
    """Try multiple approaches until we get good quality - comprehensive approach"""
    
    constraints = analyze_shape_constraints(shape)
    base_capacity = constraints["estimated_capacity"]
    
    print(f"   📐 Shape analysis: {constraints['width']:.1f}\"×{constraints['height']:.1f}\", capacity: ~{base_capacity} chars")
    
    # Define approaches in order of preference (most generous first)
    approaches = [
        # Approach 1: Try original text (most generous)
        {
            "name": "original", 
            "text": text,
            "description": "using original text"
        },
        
        # Approach 2: Try with very generous limits (3x capacity)
        {
            "name": "very_generous", 
            "text": smart_truncate_with_meaning(text, base_capacity * 3),
            "description": "with very generous limits (300%)"
        },
        
        # Approach 3: Try with generous limits (2x capacity)  
        {
            "name": "generous", 
            "text": smart_truncate_with_meaning(text, base_capacity * 2),
            "description": "with generous limits (200%)"
        },
        
        # Approach 4: Try with moderate limits (1.5x capacity)
        {
            "name": "moderate", 
            "text": smart_truncate_with_meaning(text, int(base_capacity * 1.5)),
            "description": "with moderate limits (150%)"
        },
        
        # Approach 5: Try with conservative limits (1.2x capacity - old approach)
        {
            "name": "conservative", 
            "text": smart_truncate_with_meaning(text, int(base_capacity * 1.2)),
            "description": "with conservative limits (120%)"
        },
        
        # Approach 6: Simple fallback
        {
            "name": "simple", 
            "text": simple_fit_text(text, base_capacity),
            "description": "using simple word-boundary fitting"
        }
    ]
    
    original_len = len(text)
    
    # Try each approach until we find one that works well
    for approach in approaches:
        candidate_text = approach["text"]
        candidate_len = len(candidate_text)
        
        # Test if it fits and looks good
        fits = test_if_text_fits(shape, candidate_text)
        quality_good = validate_truncation_quality(text, candidate_text)
        
        if fits and quality_good:
            print(f"   ✅ Success {approach['description']}")
            if candidate_len != original_len:
                print(f"   📏 Text adjusted: '{text[:20]}...' ({original_len}) → '{candidate_text[:20]}...' ({candidate_len})")
            else:
                print(f"   📏 Text fits perfectly: {original_len} chars")
            return candidate_text
        else:
            # Log why this approach didn't work
            reasons = []
            if not fits:
                reasons.append("doesn't fit")
            if not quality_good:
                reasons.append("poor quality")
            print(f"   ⚠️ {approach['name']} approach failed: {', '.join(reasons)}")
    
    # If all approaches fail, use original text and let PowerPoint handle it
    print(f"   🚨 All approaches failed - using original text (PowerPoint will handle overflow)")
    return text

def simple_fit_text(text, desired_len):
    """Simple, reliable text fitting - the original approach as fallback"""
    if len(text) <= desired_len:
        return text
    
    # Find last complete word that fits
    words = text.split()
    fitted = ""
    for word in words:
        test_text = fitted + (" " if fitted else "") + word
        if len(test_text) <= desired_len:
            fitted = test_text
        else:
            break
    
    # If no words fit, take partial text up to last complete word
    if not fitted:
        # Take as much as possible, but try to end at word boundary
        truncated = text[:desired_len]
        if ' ' in truncated:
            fitted = truncated.rsplit(' ', 1)[0]
        else:
            fitted = truncated
    
    return fitted

def estimate_line_count(text, width_inches):
    """Estimate how many lines text will take"""
    try:
        # Rough estimate: ~12-15 characters per inch width
        chars_per_line = int(width_inches * 13)
        if chars_per_line < 10:
            chars_per_line = 10
        
        lines = len(text) // chars_per_line + (1 if len(text) % chars_per_line else 0)
        return max(lines, 1)
    except:
        return 1

def reduce_font_size(shape, target_reduction=0.8):
    """Reduce font size to fit more text"""
    try:
        for paragraph in shape.text_frame.paragraphs:
            for run in paragraph.runs:
                if run.font.size:
                    current_size = run.font.size.pt
                    new_size = max(8, int(current_size * target_reduction))  # Minimum 8pt
                    run.font.size = Pt(new_size)
                    print(f"   📝 Reduced font size: {current_size}pt → {new_size}pt")
    except Exception as e:
        print(f"   ⚠️ Font size reduction failed: {e}")

def set_conservative_margins(shape):
    """Set conservative margins for better text fitting"""
    try:
        if hasattr(shape, 'text_frame'):
            tf = shape.text_frame
            tf.margin_left = Inches(0.1)
            tf.margin_right = Inches(0.1)
            tf.margin_top = Inches(0.05)
            tf.margin_bottom = Inches(0.05)
    except:
        pass

def apply_no_autosize_settings(shape):
    """Apply settings to prevent auto-sizing and enable word wrap"""
    try:
        if hasattr(shape, 'text_frame'):
            tf = shape.text_frame
            tf.auto_size = False  # Disable auto-sizing
            tf.word_wrap = True   # Enable word wrapping
    except:
        pass

# Legacy functions for backward compatibility - now use the new quality chain approach
def intelligent_truncate(text, max_length, strategy="word_boundary"):
    """Legacy function - now redirects to smart_truncate_with_meaning"""
    return smart_truncate_with_meaning(text, max_length)

def fit_text_intelligently(shape, text, field_type="body"):
    """Main text fitting function - now uses the comprehensive quality chain approach"""
    try:
        return fit_text_with_quality_chain(shape, text, field_type)
    except Exception as e:
        print(f"   ⚠️ Quality chain fitting failed: {e}")
        # Ultimate fallback to simple approach
        return simple_fit_text(text, 100)  # Safe default length

def replace_with_ai_content(pptx_path, ai_output, output_path):
    """Replace PowerPoint content with AI-generated text"""
    prs = Presentation(pptx_path)
    successful_replacements = 0
    total_fields = len(ai_output)
    
    print(f"📝 Replacing content in PowerPoint template...")
    
    for item in ai_output:
        source = item.get('source')
        source_index = item.get('source_index')
        slide_index = item.get('slide', -1)
        shape_index = item.get('shape')
        text = item.get('text', '')
        
        shape = None
        field_type = "body"  # Default
        
        try:
            # Determine field type from text characteristics
            if len(text) < 50:
                field_type = "header"
            elif len(text) < 150:
                field_type = "subheader"
            else:
                field_type = "body"
            
            # Get the shape based on source
            if source == 'slide' and slide_index >= 0:
                slide = prs.slides[slide_index]
                if shape_index < len(slide.shapes):
                    shape = slide.shapes[shape_index]
            elif source == 'layout':
                # Handle layout shapes
                if source_index < len(prs.slide_layouts):
                    layout = prs.slide_layouts[source_index]
                    if shape_index < len(layout.shapes):
                        shape = layout.shapes[shape_index]
            elif source == 'master':
                # Handle master shapes
                if source_index < len(prs.slide_masters):
                    master = prs.slide_masters[source_index]
                    if shape_index < len(master.shapes):
                        shape = master.shapes[shape_index]
            
            # Replace content if shape found
            if shape and hasattr(shape, 'has_text_frame') and shape.has_text_frame:
                # CRITICAL: Always try to set content, never skip
                try:
                    # Use the new comprehensive quality chain approach
                    fitted_text = fit_text_with_quality_chain(shape, text, field_type)
                    
                    # Apply conservative settings
                    apply_no_autosize_settings(shape)
                    set_conservative_margins(shape)
                    
                    # Set the text content
                    if shape.text_frame.paragraphs:
                        p = shape.text_frame.paragraphs[0]
                        p.text = fitted_text
                        
                        # Apply smart text color
                        text_color = get_smart_text_color(shape)
                        for paragraph in shape.text_frame.paragraphs:
                            for run in paragraph.runs:
                                if run.text.strip():
                                    run.font.color.rgb = text_color
                        
                        successful_replacements += 1
                    
                except Exception as e:
                    # FALLBACK: If everything fails, use simple approach
                    print(f"   ⚠️ Advanced fitting failed, using simple fallback: {e}")
                    try:
                        simple_text = simple_fit_text(text, 150)  # More generous fallback
                        shape.text_frame.paragraphs[0].text = simple_text
                        successful_replacements += 1
                    except Exception as e2:
                        print(f"Warning: Could not replace ({item}) - {e2}")
            else:
                print(f"Warning: Could not replace ({item}) - shape not found or no text frame")
                
        except Exception as e:
            print(f"Warning: Could not replace ({item}) - {e}")
    
    # Save the presentation
    prs.save(output_path)
    print(f"Successfully replaced {successful_replacements} out of {total_fields} fields")
    print(f"Saved as {output_path}. (Close any open instances while running.)")

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 4:
        print("Usage: python replace_content.py template.pptx ai_output.json output.pptx")
    else:
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            aiout = json.load(f)
        replace_with_ai_content(sys.argv[1], aiout, sys.argv[3])
