"""
Universal AI + Rule-Based Question Analysis System
Central intelligence engine for all slide format detection and question understanding
"""

import openai
import os
import json
from dotenv import load_dotenv

class UniversalQuestionAnalyzer:
    """
    Universal question analyzer combining AI semantic understanding with rule-based validation
    Replaces all scattered format-specific analysis functions with unified intelligence
    """
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Smart context-aware rules
        self.smart_rules = {
            "timeline_indicators": {
                "strong": ["process of", "steps to", "how to", "procedure for", "workflow for", "method to", "approach to"],
                "historical": ["history", "evolution", "evolved", "developed over time", "timeline", "chronology", "over the years", "development of", "advancement", "progression over time", "since its inception", "over time"],
                "lifecycle": ["lifecycle", "life cycle", "development cycle", "phases of development", "project phases", "software development", "cycle"],
                "sequential": ["sequence of", "order of", "progression of", "flow of", "chain of", "series of"],
                "stages": ["stages of", "phases of", "levels of"],
                "contextual": {
                    "key": {
                        "timeline_context": ["key generation", "key pair", "private key", "public key", "cryptographic key", "generate key", "creating key"],
                        "component_context": ["key features", "key components", "main key", "key aspects", "key parts", "core key"]
                    }
                }
            },
            "table_indicators": {
                "strong": ["statistics", "data on", "metrics", "performance", "figures", "numbers", "analysis", "results"],
                "comparison": ["vs", "versus", "compared to", "difference between", "comparison of", "compare"],
                "data": ["show data", "display statistics", "present figures", "metrics for", "performance of"]
            },
            "boxes_indicators": {
                "comparison": ["differences", "compare", "contrast", "versus", "vs", "different from"],
                "components": ["components", "parts", "elements", "features", "aspects", "categories"],
                "features": ["features of", "characteristics of", "attributes of", "properties of"]
            },
            "bullets_indicators": {
                "listing": ["what are", "list", "enumerate", "types", "kinds", "benefits", "advantages"],
                "simple": ["main points", "key points", "important aspects", "overview"]
            },
            "paragraph_indicators": {
                "explanation": ["explain", "describe", "discuss", "analyze", "understand", "how does", "what is", "why"],
                "complex": ["elaborate", "detail", "comprehensive", "in-depth"]
            }
        }
        
        # Format confidence thresholds
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
    
    def analyze_question_comprehensive(self, question, content="", context=None):
        """
        Single entry point for ALL question analysis
        Returns format recommendation with confidence and reasoning
        """
        if not question or not question.strip():
            return {
                "format": "paragraph",
                "subtype": "explanation",
                "confidence": 0.5,
                "reasoning": "Empty question - defaulting to paragraph format",
                "ai_analysis": None,
                "rule_analysis": None
            }
        
        print(f"   🔍 Universal Question Analysis: '{question[:50]}...'")
        
        # Step 1: AI Semantic Understanding
        ai_analysis = self.get_ai_semantic_analysis(question, content)
        print(f"   🤖 AI Analysis: {ai_analysis}")
        
        # Step 2: Rule-Based Validation & Enhancement
        rule_analysis = self.get_enhanced_rule_analysis(question)
        print(f"   📋 Rule Analysis: {rule_analysis}")
        
        # Step 3: Content-Question Compatibility Check
        compatibility = self.validate_content_match(ai_analysis, content, question)
        print(f"   🔗 Compatibility: {compatibility}")
        
        # Step 4: Unified Decision with Confidence
        final_decision = self.make_final_decision(ai_analysis, rule_analysis, compatibility)
        print(f"   ✅ Final Decision: {final_decision['format']}-{final_decision['subtype']} (confidence: {final_decision['confidence']:.2f})")
        
        return final_decision
    
    def get_ai_semantic_analysis(self, question, content=""):
        """
        AI-powered semantic analysis of question intent
        """
        content_sample = content[:300] + "..." if len(content) > 300 else content
        
        ai_analysis_prompt = f"""
Analyze this question and content to determine the optimal presentation format:

Question: "{question}"
Content Sample: "{content_sample}"

Classify as one of:
1. TIMELINE-PROCESS: Sequential steps/procedures ("How to...", "Process of...", "Steps to...")
2. TIMELINE-HISTORICAL: Evolution over time ("History of...", "Development of...")
3. TIMELINE-LIFECYCLE: Development phases ("Lifecycle of...", "Development cycle")
4. TABLE-COMPARISON: Data/statistics comparison ("Compare X vs Y", "Performance metrics")
5. TABLE-DATA: Structured data presentation ("Statistics on...", "Data about...")
6. BOXES-COMPARISON: Feature/concept comparison ("Differences between...", "X vs Y")
7. BOXES-COMPONENTS: Parts/elements listing ("Components of...", "Features of...")
8. BULLETS-SIMPLE: Simple list/enumeration ("What are...", "Types of...")
9. PARAGRAPH-EXPLANATION: Complex explanation/analysis ("Explain...", "Describe...")

Consider:
- Question intent and context
- Content structure and complexity
- Most natural presentation format for this information

Respond with ONLY: "FORMAT-SUBTYPE confidence" (e.g., "TIMELINE-PROCESS 0.9")
"""
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": ai_analysis_prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Parse AI response
            parts = ai_response.split()
            if len(parts) >= 2:
                format_subtype = parts[0]
                confidence = float(parts[1])
                
                if "-" in format_subtype:
                    format_type, subtype = format_subtype.lower().split("-", 1)
                else:
                    format_type = format_subtype.lower()
                    subtype = "default"
                
                return {
                    "format": format_type,
                    "subtype": subtype,
                    "confidence": confidence,
                    "raw_response": ai_response
                }
            else:
                raise ValueError(f"Invalid AI response format: {ai_response}")
                
        except Exception as e:
            print(f"   WARNING: AI semantic analysis failed: {e}")
            return {
                "format": "unclear",
                "subtype": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_enhanced_rule_analysis(self, question):
        """
        Enhanced rule-based analysis with context awareness
        """
        question_lower = question.lower()
        rule_scores = {}
        detected_indicators = []
        
        # Timeline analysis
        timeline_score = 0
        
        # Strong timeline indicators
        for indicator in self.smart_rules["timeline_indicators"]["strong"]:
            if indicator in question_lower:
                timeline_score += 0.9
                detected_indicators.append(f"timeline-strong: {indicator}")
        
        # Historical indicators
        for indicator in self.smart_rules["timeline_indicators"]["historical"]:
            if indicator in question_lower:
                timeline_score += 0.8
                detected_indicators.append(f"timeline-historical: {indicator}")
        
        # Context-aware key analysis
        if 'key' in question_lower:
            timeline_context_found = any(term in question_lower for term in self.smart_rules["timeline_indicators"]["contextual"]["key"]["timeline_context"])
            component_context_found = any(term in question_lower for term in self.smart_rules["timeline_indicators"]["contextual"]["key"]["component_context"])
            
            if timeline_context_found:
                timeline_score += 0.7
                detected_indicators.append("timeline-key-context: cryptographic")
            elif component_context_found:
                timeline_score -= 0.5
                detected_indicators.append("component-key-context: listing")
        
        rule_scores["timeline"] = min(timeline_score, 1.0)
        
        # Table analysis
        table_score = 0
        for indicator in self.smart_rules["table_indicators"]["strong"]:
            if indicator in question_lower:
                table_score += 0.8
                detected_indicators.append(f"table-strong: {indicator}")
        
        for indicator in self.smart_rules["table_indicators"]["comparison"]:
            if indicator in question_lower:
                table_score += 0.7
                detected_indicators.append(f"table-comparison: {indicator}")
        
        rule_scores["table"] = min(table_score, 1.0)
        
        # Boxes analysis
        boxes_score = 0
        for indicator in self.smart_rules["boxes_indicators"]["comparison"]:
            if indicator in question_lower:
                boxes_score += 0.8
                detected_indicators.append(f"boxes-comparison: {indicator}")
        
        for indicator in self.smart_rules["boxes_indicators"]["components"]:
            if indicator in question_lower:
                boxes_score += 0.6
                detected_indicators.append(f"boxes-components: {indicator}")
        
        rule_scores["boxes"] = min(boxes_score, 1.0)
        
        # Bullets analysis
        bullets_score = 0
        for indicator in self.smart_rules["bullets_indicators"]["listing"]:
            if indicator in question_lower:
                bullets_score += 0.7
                detected_indicators.append(f"bullets-listing: {indicator}")
        
        rule_scores["bullets"] = min(bullets_score, 1.0)
        
        # Paragraph analysis
        paragraph_score = 0
        for indicator in self.smart_rules["paragraph_indicators"]["explanation"]:
            if indicator in question_lower:
                paragraph_score += 0.6
                detected_indicators.append(f"paragraph-explanation: {indicator}")
        
        rule_scores["paragraph"] = min(paragraph_score, 1.0)
        
        # Determine best rule-based format
        if rule_scores:
            best_format = max(rule_scores.items(), key=lambda x: x[1])
            return {
                "format": best_format[0],
                "confidence": best_format[1],
                "scores": rule_scores,
                "indicators": detected_indicators
            }
        else:
            return {
                "format": "paragraph",
                "confidence": 0.3,
                "scores": rule_scores,
                "indicators": ["default: no strong indicators found"]
            }
    
    def validate_content_match(self, ai_analysis, content, question):
        """
        Validate if AI analysis matches content structure
        """
        if not content or ai_analysis.get("format") == "unclear":
            return {"compatibility": 0.5, "reasoning": "Limited content for validation"}
        
        content_length = len(content.split())
        content_lower = content.lower()
        
        compatibility_score = 0.5  # neutral baseline
        reasoning = []
        
        ai_format = ai_analysis.get("format", "")
        
        # Timeline compatibility checks
        if ai_format == "timeline":
            if any(word in content_lower for word in ["step", "first", "second", "then", "next", "finally", "process"]):
                compatibility_score += 0.3
                reasoning.append("Content shows sequential structure")
            if content_length > 100:  # Timeline needs substantial content
                compatibility_score += 0.2
                reasoning.append("Sufficient content for timeline")
        
        # Table compatibility checks
        elif ai_format == "table":
            if any(word in content_lower for word in ["data", "statistics", "numbers", "percentage", "metrics"]):
                compatibility_score += 0.3
                reasoning.append("Content contains data/statistics")
            if "vs" in content_lower or "versus" in content_lower:
                compatibility_score += 0.2
                reasoning.append("Content shows comparison structure")
        
        # Boxes compatibility checks
        elif ai_format == "boxes":
            if any(word in content_lower for word in ["different", "compare", "features", "types", "categories"]):
                compatibility_score += 0.3
                reasoning.append("Content shows categorical structure")
        
        return {
            "compatibility": min(compatibility_score, 1.0),
            "reasoning": "; ".join(reasoning) if reasoning else "Basic compatibility check"
        }
    
    def make_final_decision(self, ai_analysis, rule_analysis, compatibility):
        """
        Combine AI + rules + compatibility to make final decision
        """
        ai_format = ai_analysis.get("format", "unclear")
        ai_confidence = ai_analysis.get("confidence", 0.0)
        ai_subtype = ai_analysis.get("subtype", "default")
        
        rule_format = rule_analysis.get("format", "paragraph")
        rule_confidence = rule_analysis.get("confidence", 0.0)
        
        compatibility_score = compatibility.get("compatibility", 0.5)
        
        # Decision logic: AI + rules with compatibility weighting
        if ai_format != "unclear" and ai_confidence >= self.confidence_thresholds["high"]:
            # High confidence AI - use AI with rule validation
            if rule_format == ai_format or rule_confidence < 0.5:
                final_format = ai_format
                final_subtype = ai_subtype
                final_confidence = min(ai_confidence * compatibility_score * 1.1, 1.0)
                reasoning = f"High-confidence AI analysis confirmed by compatibility"
            else:
                # AI vs rules conflict - check rule confidence
                if rule_confidence >= self.confidence_thresholds["high"]:
                    final_format = rule_format
                    final_subtype = "default"
                    final_confidence = (rule_confidence + ai_confidence) * 0.5 * compatibility_score
                    reasoning = f"Rules override AI due to strong rule indicators"
                else:
                    final_format = ai_format
                    final_subtype = ai_subtype
                    final_confidence = ai_confidence * compatibility_score * 0.9
                    reasoning = f"AI analysis preferred over weak rules"
        
        elif rule_confidence >= self.confidence_thresholds["medium"]:
            # Use rules when AI is unclear but rules are confident
            final_format = rule_format
            final_subtype = "default"
            final_confidence = rule_confidence * compatibility_score
            reasoning = f"Rule-based analysis due to unclear AI"
        
        else:
            # Low confidence overall - use heuristics
            if "?" in ai_analysis.get("raw_response", ""):
                final_format = "paragraph"
                final_subtype = "explanation"
            elif len(rule_analysis.get("indicators", [])) > 0:
                final_format = rule_format
                final_subtype = "default"
            else:
                final_format = "bullets"
                final_subtype = "simple"
            
            final_confidence = 0.4
            reasoning = "Low confidence - using heuristic fallback"
        
        return {
            "format": final_format,
            "subtype": final_subtype,
            "confidence": final_confidence,
            "reasoning": reasoning,
            "ai_analysis": ai_analysis,
            "rule_analysis": rule_analysis,
            "compatibility": compatibility
        }
    
    def get_format_template_info(self, format_type, subtype="default"):
        """
        Get template and formatting information for detected format
        """
        template_info = {
            "timeline": {
                "process": {"headers": ["Step", "Action", "Result"], "style": "process_flow"},
                "historical": {"headers": ["Period", "Event", "Impact"], "style": "timeline"},
                "lifecycle": {"headers": ["Phase", "Activities", "Deliverables"], "style": "lifecycle"},
                "default": {"headers": ["Step", "Description", "Outcome"], "style": "generic"}
            },
            "table": {
                "comparison": {"headers": ["Aspect", "Option A", "Option B"], "style": "comparison"},
                "data": {"headers": ["Category", "Value", "Description"], "style": "data"},
                "default": {"headers": ["Item", "Details", "Notes"], "style": "generic"}
            },
            "boxes": {
                "comparison": {"layout": "side_by_side", "style": "comparison"},
                "components": {"layout": "grid", "style": "components"},
                "default": {"layout": "grid", "style": "generic"}
            },
            "bullets": {
                "simple": {"style": "bullet_points", "max_items": 8},
                "default": {"style": "bullet_points", "max_items": 6}
            },
            "paragraph": {
                "explanation": {"style": "structured_text", "sections": True},
                "default": {"style": "simple_text", "sections": False}
            }
        }
        
        return template_info.get(format_type, {}).get(subtype, template_info.get(format_type, {}).get("default", {}))

# Global instance for easy import
universal_analyzer = UniversalQuestionAnalyzer()