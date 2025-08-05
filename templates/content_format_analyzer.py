import re
import json
from collections import Counter
from typing import Dict, List, Tuple, Any

class ContentFormatAnalyzer:
    """
    Enhanced analyzer that determines optimal format, slide type, and template assignment
    with intelligent flow planning and topic-based analysis
    """
    
    def __init__(self):
        self.format_patterns = {
            "bullet_points": {
                "keywords": ["first", "second", "third", "finally", "lastly", "next", "then", "also", "additionally", "furthermore", "moreover"],
                "list_patterns": [r'\d+\.', r'•', r'[-*]\s', r'[a-z]\)', r'[A-Z]\.'],
                "min_items": 3,
                "confidence_boost": 0.2
            },
            "boxes": {
                "keywords": ["different", "various", "multiple", "several", "categories", "types", "kinds", "aspects", "components", "elements"],
                "comparison_words": ["vs", "versus", "compared", "difference", "similar", "contrast"],
                "min_concepts": 2,
                "max_concepts": 6,
                "confidence_boost": 0.15
            },
            "table": {
                "keywords": ["data", "statistics", "figures", "numbers", "percentage", "ratio", "comparison", "analysis"],
                "structured_patterns": [r'\d+%', r'\d+\.\d+', r'\$\d+', r'\d+/\d+'],
                "min_structured_items": 3,
                "confidence_boost": 0.25
            },
            "timeline": {
                "keywords": ["timeline", "history", "evolution", "development", "progress", "stages", "phases", "steps", "process", "procedure", "workflow", "method", "approach", "sequence", "generation", "creating", "building"],
                "time_words": ["before", "after", "during", "while", "when", "then", "next", "previous", "earlier", "later"],
                "sequence_words": ["step", "stage", "phase", "level", "iteration", "first", "second", "third", "finally"],
                "confidence_boost": 0.4
            },
            "paragraph": {
                "keywords": ["explain", "describe", "discuss", "analyze", "examine", "explore", "investigate", "understand"],
                "min_sentences": 3,
                "confidence_boost": 0.1
            }
        }
        
        self.slide_type_mapping = {
            "bullet_points": ["content_slide", "list_slide", "overview_slide"],
            "boxes": ["comparison_slide", "grid_slide", "category_slide"],
            "table": ["table_slide", "data_slide", "comparison_slide"],
            "timeline": ["timeline_slide", "process_slide", "flow_slide"],
            "paragraph": ["content_slide", "overview_slide", "detail_slide"]
        }
    
    def analyze_content_format_with_flow(self, question: str, answer_content: str, slide_metadata: List[Dict], 
                                       slide_position: int, total_slides: int, topic: str, 
                                       template_inventory: Dict = None) -> Dict[str, Any]:
        """
        Enhanced content format analysis with flow planning and intelligent template assignment
        """
        if not answer_content or not answer_content.strip():
            return self._get_default_analysis_with_flow(slide_position, total_slides, topic)
        
        # Basic content analysis - pass question for enhanced detection
        content_analysis = self._analyze_content_characteristics(answer_content)
        content_analysis["question"] = question  # Add question for enhanced detection
        question_analysis = self._analyze_question_context(question)
        
        # Calculate format scores
        format_scores = self._calculate_format_scores(content_analysis, question_analysis)
        
        # Determine slide role in presentation flow
        slide_role = self._determine_slide_role(slide_position, total_slides, topic)
        
        # Adjust format scores based on presentation flow
        adjusted_scores = self._adjust_format_scores_for_flow(format_scores, slide_role, topic)
        
        # Get best format
        best_format = max(adjusted_scores.items(), key=lambda x: x[1])[0]
        confidence_score = adjusted_scores[best_format]
        
        # Get recommended slide type
        recommended_slide_type = self._get_recommended_slide_type(best_format, content_analysis)
        
        # Get formatting instructions
        formatting_instructions = self._get_formatting_instructions(best_format, content_analysis)
        
        # Intelligent template assignment
        template_match = self._assign_intelligent_template(
            best_format, slide_role, template_inventory, slide_position, total_slides
        )
        
        return {
            "content_format": best_format,
            "recommended_slide_type": recommended_slide_type,
            "formatting_instructions": formatting_instructions,
            "confidence_score": confidence_score,
            "template_match": template_match,
            "slide_role": slide_role,
            "content_analysis": content_analysis,
            "format_scores": adjusted_scores,
            "flow_adjustment": self._calculate_flow_adjustment(slide_role, topic)
        }
    
    def _determine_slide_role(self, slide_position: int, total_slides: int, topic: str) -> str:
        """Determine the role of this slide in the presentation flow"""
        if slide_position == 1:
            return "introduction"
        elif slide_position == total_slides:
            return "conclusion"
        elif slide_position <= total_slides * 0.3:
            return "early_content"
        elif slide_position >= total_slides * 0.7:
            return "late_content"
        else:
            return "main_content"
    
    def _calculate_flow_adjustment(self, slide_role: str, topic: str) -> float:
        """Calculate flow adjustment based on slide role and topic"""
        base_adjustments = {
            "introduction": 0.2,
            "early_content": 0.1,
            "main_content": 0.0,
            "late_content": 0.05,
            "conclusion": 0.15
        }
        
        topic_lower = topic.lower()
        topic_adjustment = 0.0
        
        # Topic-specific adjustments
        if "introduction" in topic_lower or "basics" in topic_lower:
            topic_adjustment = 0.1
        elif "comparison" in topic_lower or "analysis" in topic_lower:
            topic_adjustment = 0.05
        elif "process" in topic_lower or "steps" in topic_lower:
            topic_adjustment = 0.1
        
        return base_adjustments.get(slide_role, 0.0) + topic_adjustment
    
    def _adjust_format_scores_for_flow(self, format_scores: Dict[str, float], slide_role: str, topic: str) -> Dict[str, float]:
        """Adjust format scores based on slide role and topic"""
        adjusted_scores = format_scores.copy()
        topic_lower = topic.lower()
        
        if slide_role == "introduction":
            # Boost paragraph and overview formats for introduction
            adjusted_scores["paragraph"] *= 1.3
            adjusted_scores["boxes"] *= 1.1
            if "overview" in topic_lower or "introduction" in topic_lower:
                adjusted_scores["paragraph"] *= 1.2
        
        elif slide_role == "conclusion":
            # Boost paragraph and summary formats for conclusion
            adjusted_scores["paragraph"] *= 1.4
            adjusted_scores["bullet_points"] *= 1.2
            if "summary" in topic_lower or "conclusion" in topic_lower:
                adjusted_scores["paragraph"] *= 1.1
        
        elif slide_role in ["early_content", "main_content"]:
            # Boost content-specific formats based on topic
            if "comparison" in topic_lower or "vs" in topic_lower:
                adjusted_scores["boxes"] *= 1.3
            elif "process" in topic_lower or "steps" in topic_lower:
                adjusted_scores["timeline"] *= 1.3
            elif "data" in topic_lower or "analysis" in topic_lower:
                adjusted_scores["table"] *= 1.3
            elif "types" in topic_lower or "kinds" in topic_lower:
                adjusted_scores["bullet_points"] *= 1.2
        
        # FORCE VARIATION: Reduce bullet points dominance
        # If bullet points score is too high (>0.6), boost other formats
        if adjusted_scores.get("bullet_points", 0) > 0.6:
            # Boost the next best alternative format
            other_formats = {k: v for k, v in adjusted_scores.items() if k != "bullet_points"}
            if other_formats:
                best_other = max(other_formats.items(), key=lambda x: x[1])
                adjusted_scores[best_other[0]] = min(best_other[1] * 1.3, 1.0)
                # Slightly reduce bullet points
                adjusted_scores["bullet_points"] = max(adjusted_scores["bullet_points"] * 0.8, 0.0)
        
        return adjusted_scores
    
    def _assign_intelligent_template(self, format_type: str, slide_role: str, 
                                   template_inventory: Dict, slide_position: int, total_slides: int) -> Dict[str, Any]:
        """Intelligently assign template based on format, role, and available templates"""
        if not template_inventory:
            return self._get_fallback_template(slide_position, total_slides)
        
        # Map content formats to preferred template types
        format_to_template_mapping = {
            "bullet_points": ["list_slides", "content_slides"],
            "boxes": ["comparison_slides", "content_slides"],
            "table": ["table_slides", "content_slides"],
            "timeline": ["timeline_slides", "content_slides"],
            "paragraph": ["content_slides", "overview_slides"]
        }
        
        preferred_types = format_to_template_mapping.get(format_type, ["content_slides"])
        
        # Find available templates of preferred types
        available_templates = []
        for template_type in preferred_types:
            if template_type in template_inventory and template_inventory[template_type]:
                available_templates.extend(template_inventory[template_type])
        
        if not available_templates:
            # Fallback to any available template
            for template_list in template_inventory.values():
                if template_list:
                    available_templates.extend(template_list)
                    break
        
        if not available_templates:
            return self._get_fallback_template(slide_position, total_slides)
        
        # Score templates based on suitability for this position and role
        scored_templates = []
        for template in available_templates:
            score = self._calculate_template_suitability_score(
                template, format_type, slide_role, slide_position, total_slides
            )
            scored_templates.append((template, score))
        
        # Return best scoring template
        best_template = max(scored_templates, key=lambda x: x[1])[0]
        
        return {
            "slide_number": best_template["slide_number"],
            "template_type": best_template["template_type"],
            "match_score": max(scored_templates, key=lambda x: x[1])[1],
            "field_count": best_template["field_count"],
            "slide_role": slide_role
        }
    
    def _calculate_template_suitability_score(self, template: Dict, format_type: str, 
                                            slide_role: str, slide_position: int, total_slides: int) -> float:
        """Calculate how suitable a template is for this specific format and position"""
        base_score = 0.5
        
        # Format compatibility
        format_compatibility = {
            "bullet_points": {"list_slides": 0.9, "content_slides": 0.7},
            "boxes": {"comparison_slides": 0.9, "content_slides": 0.6},
            "table": {"table_slides": 0.9, "content_slides": 0.5},
            "timeline": {"timeline_slides": 0.9, "content_slides": 0.6},
            "paragraph": {"content_slides": 0.9, "overview_slides": 0.8}
        }
        
        layout_type = template.get("template_type", "content_slides")
        format_score = format_compatibility.get(format_type, {}).get(layout_type, 0.5)
        base_score += format_score * 0.3
        
        # Position-based scoring
        if slide_role == "introduction" and slide_position == 1:
            if layout_type in ["title_slides", "overview_slides"]:
                base_score += 0.2
        elif slide_role == "conclusion" and slide_position == total_slides:
            if layout_type in ["conclusion_slides", "overview_slides"]:
                base_score += 0.2
        
        # Field count optimization
        field_count = template.get("field_count", 2)
        if format_type == "bullet_points" and field_count >= 3:
            base_score += 0.1
        elif format_type == "boxes" and field_count >= 4:
            base_score += 0.1
        elif format_type == "table" and field_count >= 5:
            base_score += 0.1
        elif format_type == "paragraph" and field_count <= 3:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _get_fallback_template(self, slide_position: int, total_slides: int) -> Dict[str, Any]:
        """Get fallback template when no inventory is available"""
        if slide_position == 1:
            return {"slide_number": 0, "template_type": "title_slides", "match_score": 0.5, "field_count": 2}
        elif slide_position == total_slides:
            return {"slide_number": 18, "template_type": "conclusion_slides", "match_score": 0.5, "field_count": 4}
        else:
            return {"slide_number": 1, "template_type": "content_slides", "match_score": 0.5, "field_count": 3}
    
    def _get_default_analysis_with_flow(self, slide_position: int, total_slides: int, topic: str) -> Dict[str, Any]:
        """Get default analysis when no content is available"""
        slide_role = self._determine_slide_role(slide_position, total_slides, topic)
        template_match = self._get_fallback_template(slide_position, total_slides)
        
        return {
            "content_format": "paragraph",
            "recommended_slide_type": "content_slide",
            "formatting_instructions": "Use standard paragraph format with clear title and body text.",
            "confidence_score": 0.5,
            "template_match": template_match,
            "slide_role": slide_role,
            "content_analysis": {},
            "format_scores": {"paragraph": 0.5},
            "flow_adjustment": self._calculate_flow_adjustment(slide_role, topic)
        }
    
    # Keep existing methods for backward compatibility
    def analyze_content_format(self, question: str, answer_content: str, slide_metadata: List[Dict]) -> Dict[str, Any]:
        """Legacy method - redirects to enhanced version"""
        return self.analyze_content_format_with_flow(
            question, answer_content, slide_metadata, 1, 1, "General Topic"
        )
    
    def _analyze_content_characteristics(self, content: str) -> Dict[str, Any]:
        """Analyze the characteristics of the content"""
        content_lower = content.lower()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = content.split()
        word_count = len(words)
        
        # Count list patterns
        list_patterns = sum(len(re.findall(pattern, content)) for pattern in 
                          self.format_patterns["bullet_points"]["list_patterns"])
        
        # Count structured data patterns
        structured_patterns = sum(len(re.findall(pattern, content)) for pattern in 
                                self.format_patterns["table"]["structured_patterns"])
        
        # Count time/sequence words
        time_words = sum(1 for word in words if word in self.format_patterns["timeline"]["time_words"])
        sequence_words = sum(1 for word in words if word in self.format_patterns["timeline"]["sequence_words"])
        
        # Count comparison words
        comparison_words = sum(1 for word in words if word in self.format_patterns["boxes"]["comparison_words"])
        
        # Enhanced pattern detection
        # Look for bullet point indicators in content
        bullet_indicators = ["first", "second", "third", "finally", "lastly", "next", "then", "also", "additionally", "furthermore", "moreover", "1.", "2.", "3.", "•", "-", "*"]
        bullet_indicators_count = sum(1 for word in words if word.lower() in bullet_indicators)
        
        # Look for comparison indicators
        comparison_indicators = ["however", "but", "while", "whereas", "on the other hand", "in contrast", "similarly", "likewise", "both", "neither", "either", "vs", "versus"]
        comparison_indicators_count = sum(1 for word in words if word.lower() in comparison_indicators)
        
        # Look for data indicators
        data_indicators = ["percent", "percentage", "ratio", "average", "total", "sum", "increase", "decrease", "growth", "rate", "number", "amount", "quantity"]
        data_indicators_count = sum(1 for word in words if word.lower() in data_indicators)
        
        # Analyze sentence structure
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        return {
            "word_count": word_count,
            "sentence_count": len(sentences),
            "avg_sentence_length": avg_sentence_length,
            "list_patterns": list_patterns,
            "structured_patterns": structured_patterns,
            "time_words": time_words,
            "sequence_words": sequence_words,
            "comparison_words": comparison_words,
            "bullet_indicators_count": bullet_indicators_count,
            "comparison_indicators_count": comparison_indicators_count,
            "data_indicators_count": data_indicators_count,
            "content_lower": content_lower,
            "text": content  # Add the original text for pattern matching
        }
    
    def _analyze_question_context(self, question: str) -> Dict[str, Any]:
        """Analyze the question context for format clues"""
        question_lower = question.lower()
        
        # Enhanced question type detection
        question_indicators = {
            "bullet_points": [
                "what are", "list", "enumerate", "steps", "ways", "methods", "types", "kinds", 
                "examples", "features", "benefits", "advantages", "disadvantages", "characteristics",
                "components", "elements", "factors", "reasons", "causes", "effects", "applications"
            ],
            "boxes": [
                "compare", "difference", "similarities", "categories", "aspects", "components",
                "versus", "vs", "contrast", "different", "various", "multiple", "several",
                "pros and cons", "advantages and disadvantages", "strengths and weaknesses"
            ],
            "table": [
                "data", "statistics", "figures", "analysis", "results", "comparison",
                "numbers", "percentages", "ratios", "metrics", "performance", "efficiency",
                "cost", "price", "value", "measurements", "scores", "ratings"
            ],
            "timeline": [
                "history", "evolution", "development", "process", "stages", "timeline",
                "phases", "steps", "sequence", "order", "progression", "journey",
                "growth", "advancement", "improvement", "transformation", "change over time"
            ],
            "paragraph": [
                "explain", "describe", "discuss", "analyze", "understand", "how does",
                "why is", "what is", "define", "clarify", "elaborate", "detail",
                "comprehensive", "overview", "introduction", "conclusion", "summary"
            ]
        }
        
        question_scores = {}
        for format_type, indicators in question_indicators.items():
            score = sum(1 for indicator in indicators if indicator in question_lower)
            question_scores[format_type] = score
        
        return question_scores
    
    def _calculate_format_scores(self, content_analysis: Dict, question_analysis: Dict) -> Dict[str, float]:
        """Calculate scores for each format based on content and question analysis"""
        import re
        scores = {}
        
        # Ensure content_analysis has the required keys
        if not isinstance(content_analysis, dict):
            content_analysis = {"text": "", "word_count": 0, "sentence_count": 0}
        
        # Get text content safely
        text_content = content_analysis.get("text", "")
        if not text_content:
            # Fallback to content_lower if text is not available
            text_content = content_analysis.get("content_lower", "")
        
        # ENHANCED TIMELINE DETECTION - Use the enhanced timeline analysis system
        timeline_score = 0
        
        # Get the question from content_analysis if available
        question_text = content_analysis.get("question", "")
        if question_text:
            # Use enhanced timeline question analysis
            try:
                from .replace_content import analyze_timeline_question_intent
                detected_type, template = analyze_timeline_question_intent(question_text)
                if detected_type != "generic":
                    # Strong timeline signal from enhanced analysis
                    timeline_score += 0.6
                    print(f"   >>> Enhanced timeline detection: {detected_type} (score: +0.6)")
            except ImportError:
                pass  # Fall back to basic detection
        
        # Timeline patterns in content
        timeline_patterns = [
            r'\d+\.\s+',  # Numbered steps
            r'first.*?second.*?third',  # Sequential words
            r'step\s+\d+',  # Step patterns
            r'phase\s+\d+',  # Phase patterns
            r'stage\s+\d+',  # Stage patterns
            r'process.*?involves',  # Process descriptions
            r'workflow.*?steps',  # Workflow patterns
            r'evolution.*?since',  # Evolution patterns
            r'history.*?of',  # History patterns
            r'development.*?over',  # Development patterns
        ]
        
        for pattern in timeline_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                timeline_score += 0.2
        
        # ENHANCED BOXES DETECTION - Use enhanced table analysis for better detection
        boxes_score = 0
        
        # Check for components/parts questions that should be boxes, not timeline
        components_keywords = ['components', 'parts', 'elements', 'aspects', 'features', 'main', 'key', 'core']
        # BUT don't treat process/generation questions as components questions
        process_indicators = ['process of', 'procedure for', 'steps to', 'how to', 'generating', 'creating', 'building']
        
        is_components_question = question_text and any(keyword in question_text.lower() for keyword in components_keywords)
        is_process_question = question_text and any(indicator in question_text.lower() for indicator in process_indicators)
        
        if is_components_question and not is_process_question:
            # True components questions should be boxes format, not timeline
            boxes_score += 0.5
            print(f"   >>> Components question detected for boxes format (score: +0.5)")
        elif is_process_question:
            # Process questions should be timeline format
            timeline_score += 0.5
            print(f"   >>> Process question detected for timeline format (score: +0.5)")
        
        comparison_patterns = [
            r'(\w+)\s+vs\s+(\w+)',  # A vs B
            r'(\w+)\s+versus\s+(\w+)',  # A versus B
            r'(\w+)\s+compared\s+to\s+(\w+)',  # A compared to B
            r'differences\s+between',  # Differences between
            r'advantages\s+and\s+disadvantages',  # Pros and cons
            r'benefits\s+and\s+drawbacks',  # Benefits and drawbacks
            r'on\s+one\s+hand.*?on\s+the\s+other\s+hand',  # On one hand...on the other
        ]
        
        for pattern in comparison_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                boxes_score += 0.25
        
        # ENHANCED TABLE DETECTION - Use enhanced table analysis system
        table_score = 0
        
        # Use enhanced table question analysis
        if question_text:
            try:
                from .replace_content import analyze_table_question_intent
                detected_type, template = analyze_table_question_intent(question_text)
                if detected_type != "generic":
                    # Strong table signal from enhanced analysis
                    table_score += 0.6
                    print(f"   >>> Enhanced table detection: {detected_type} (score: +0.6)")
            except ImportError:
                pass  # Fall back to basic detection
        
        table_patterns = [
            r'(\w+):\s*(\d+[%$]?)',  # Metric: Value
            r'(\w+)\s+=\s+(\d+[%$]?)',  # Metric = Value
            r'(\w+)\s+is\s+(\d+[%$]?)',  # Metric is Value
            r'performance.*?metrics',  # Performance metrics
            r'data\s+shows',  # Data shows
            r'statistics.*?indicate',  # Statistics indicate
            r'results.*?show',  # Results show
            r'effectiveness.*?data',  # Effectiveness data
            r'statistics.*?show',  # Statistics show
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                table_score += 0.2
        
        # Enhanced bullet detection
        bullet_score = 0
        bullet_patterns = [
            r'[•\-\*]\s+',  # Bullet points
            r'\d+\.\s+',  # Numbered lists
            r'types\s+of',  # Types of
            r'kinds\s+of',  # Kinds of
            r'categories\s+include',  # Categories include
            r'features\s+include',  # Features include
            r'benefits\s+include',  # Benefits include
            r'steps\s+include',  # Steps include
        ]
        
        for pattern in bullet_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                bullet_score += 0.2
        
        # Original bullet points scoring - More sensitive detection
        if content_analysis.get("list_patterns", 0) >= 1:  # Reduced threshold
            bullet_score += 0.3
        if content_analysis.get("bullet_indicators_count", 0) >= 1:  # New indicator
            bullet_score += 0.2
        if content_analysis.get("word_count", 0) > 30 and content_analysis.get("sentence_count", 0) >= 2:  # Reduced requirements
            bullet_score += 0.1
        # Boost for questions asking for lists/types
        bullet_score += question_analysis.get("bullet_points", 0) * 0.4  # Increased weight
        scores["bullet_points"] = min(bullet_score, 1.0)
        
        # Boxes scoring - More sensitive detection
        if content_analysis.get("comparison_words", 0) >= 1:  # Reduced threshold
            boxes_score += 0.2
        if content_analysis.get("comparison_indicators_count", 0) >= 1:  # New indicator
            boxes_score += 0.3
        if content_analysis.get("sentence_count", 0) >= 2 and content_analysis.get("sentence_count", 0) <= 8:  # Wider range
            boxes_score += 0.1
        # Boost for comparison questions
        boxes_score += question_analysis.get("boxes", 0) * 0.4  # Increased weight
        scores["boxes"] = min(boxes_score, 1.0)
        
        # Table scoring - More sensitive detection
        if content_analysis.get("structured_patterns", 0) >= 1:  # Reduced threshold
            table_score += 0.2
        if content_analysis.get("data_indicators_count", 0) >= 1:  # New indicator
            table_score += 0.3
        if content_analysis.get("word_count", 0) > 20:  # Reduced requirement
            table_score += 0.1
        # Boost for data/analysis questions
        table_score += question_analysis.get("table", 0) * 0.4  # Increased weight
        scores["table"] = min(table_score, 1.0)
        
        # Timeline scoring - More sensitive detection
        if content_analysis.get("time_words", 0) >= 1 or content_analysis.get("sequence_words", 0) >= 1:  # Reduced threshold
            timeline_score += 0.3
        if content_analysis.get("sentence_count", 0) >= 2:  # Reduced requirement
            timeline_score += 0.2
        # Boost for process/timeline questions
        timeline_score += question_analysis.get("timeline", 0) * 0.4  # Increased weight
        scores["timeline"] = min(timeline_score, 1.0)
        
        # Paragraph scoring - Reduced base score to allow other formats to win
        paragraph_score = 0.1  # Reduced base score
        if content_analysis.get("avg_sentence_length", 0) > 15:
            paragraph_score += 0.2
        if content_analysis.get("sentence_count", 0) >= 4:  # Increased requirement
            paragraph_score += 0.2
        paragraph_score += question_analysis.get("paragraph", 0) * 0.2
        scores["paragraph"] = min(paragraph_score, 1.0)
        
        return scores
    
    def _get_recommended_slide_type(self, format_type: str, content_analysis: Dict) -> str:
        """Get the recommended slide type based on format and content analysis"""
        slide_types = self.slide_type_mapping.get(format_type, ["content_slide"])
        
        # Refine based on content characteristics
        if format_type == "bullet_points":
            if content_analysis["list_patterns"] >= 4:
                return "list_slide"
            else:
                return "content_slide"
        
        elif format_type == "boxes":
            if content_analysis["comparison_words"] >= 3:
                return "comparison_slide"
            else:
                return "grid_slide"
        
        elif format_type == "table":
            if content_analysis["structured_patterns"] >= 5:
                return "data_slide"
            else:
                return "table_slide"
        
        elif format_type == "timeline":
            # Check for specific timeline subtype first
            timeline_subtype = content_analysis.get("timeline_subtype")
            if timeline_subtype == "process":
                return "process_slide"
            elif timeline_subtype == "historical":
                return "timeline_slide"
            elif content_analysis["sequence_words"] >= 3:
                return "process_slide"
            else:
                return "timeline_slide"
        
        else:
            return slide_types[0]
    
    def _get_formatting_instructions(self, format_type: str, content_analysis: Dict) -> str:
        """Get specific formatting instructions for the detected format"""
        instructions = {
            "bullet_points": f"""
FORMAT: Create bullet points or numbered lists
- Use {content_analysis['sentence_count']} main bullet points
- Each bullet should be a complete thought
- Use parallel structure across bullets
- Keep bullets concise but informative
- Use action verbs where appropriate
- Ensure logical flow between bullets
""",
            "boxes": f"""
FORMAT: Create content boxes or grid layout
- Organize into {min(content_analysis['sentence_count'], 4)} distinct boxes
- Each box should contain a distinct concept
- Use consistent structure across boxes
- Include clear headers for each box
- Balance content across all boxes
- Use visual separation between concepts
""",
            "table": f"""
FORMAT: Create tabular or structured content
- Organize information in rows and columns
- Use clear headers for each column
- Include {content_analysis['structured_patterns']} data points
- Ensure data is comparable across rows
- Keep table structure clean and readable
- Use consistent formatting for data
""",
            "timeline": f"""
FORMAT: Create timeline or process flow
- Organize {content_analysis['sentence_count']} sequential steps
- Use clear progression indicators
- Include time markers or sequence numbers
- Show logical flow between stages
- Use visual connectors between steps
- Maintain chronological order
""",
            "paragraph": f"""
FORMAT: Create paragraph or text-heavy content
- Use {content_analysis['sentence_count']} well-structured sentences
- Create logical paragraph flow
- Include detailed explanations
- Use transitional phrases
- Maintain consistent tone and style
- Focus on comprehensive coverage
"""
        }
        
        return instructions.get(format_type, instructions["paragraph"]).strip()

# Global instance for easy access
content_analyzer = ContentFormatAnalyzer() 