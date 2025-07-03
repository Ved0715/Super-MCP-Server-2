"""
🧠 Research Intelligence Module
Advanced academic paper analysis and understanding
"""

import re
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import textstat
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ResearchElement:
    """Structure for research paper elements"""
    element_type: str
    content: str
    confidence: float
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class StatisticalResult:
    """Structure for statistical findings"""
    test_type: str
    value: float
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    significance: bool = False
    context: str = ""

@dataclass
class ResearchContribution:
    """Structure for research contributions"""
    contribution_type: str
    description: str
    novelty_score: float
    evidence: List[str]
    impact_assessment: str

class ResearchPaperAnalyzer:
    """Advanced research paper analysis engine"""
    
    def __init__(self, config=None):
        self.config = config
        self._setup_nltk()
        
        # Research paper structure patterns
        self.section_patterns = {
            "abstract": [r"abstract", r"summary"],
            "introduction": [r"introduction", r"1\.\s*introduction", r"background"],
            "literature_review": [r"literature review", r"related work", r"previous work"],
            "methodology": [r"methodology", r"methods", r"materials and methods", r"approach"],
            "results": [r"results", r"findings", r"experiments", r"evaluation"],
            "discussion": [r"discussion", r"analysis", r"interpretation"],
            "conclusion": [r"conclusion", r"conclusions", r"summary", r"future work"],
            "references": [r"references", r"bibliography", r"works cited"]
        }
        
        # Statistical patterns
        self.stat_patterns = {
            "p_value": r"p\s*[<>=]\s*(0\.\d+|\d+\.\d*e?-?\d*)",
            "confidence_interval": r"(\d+)%\s*confidence\s*interval",
            "correlation": r"correlation\s*[r=]\s*([-]?0\.\d+)",
            "mean_std": r"(mean|average|μ)\s*[=:]\s*([\d.]+)\s*[\(±]\s*([\d.]+)",
            "t_test": r"t\s*[=\(]\s*([\d.-]+)",
            "chi_square": r"χ²\s*[=\(]\s*([\d.-]+)",
            "f_statistic": r"F\s*[=\(]\s*([\d.-]+)"
        }
        
        # Methodology indicators
        self.methodology_indicators = {
            "experimental": ["experiment", "trial", "controlled", "randomized", "blind"],
            "observational": ["observational", "survey", "cross-sectional", "longitudinal"],
            "meta_analysis": ["meta-analysis", "systematic review", "pooled analysis"],
            "qualitative": ["qualitative", "interview", "focus group", "thematic analysis"],
            "computational": ["simulation", "modeling", "algorithm", "computational"]
        }

    def _setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)

    async def analyze_research_paper(self, paper_content: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive research paper analysis"""
        try:
            sections = paper_content.get("sections", {})
            metadata = paper_content.get("metadata", {})
            full_text = paper_content.get("content", "")
            
            # Run parallel analysis
            tasks = [
                self._extract_research_elements(sections, full_text),
                self._analyze_methodology(sections, full_text),
                self._extract_statistical_results(full_text),
                self._identify_research_contributions(sections, full_text),
                self._analyze_citations(full_text),
                self._assess_research_quality(sections, full_text),
                self._extract_research_questions(sections, full_text),
                self._analyze_limitations(sections, full_text)
            ]
            
            results = await asyncio.gather(*tasks)
            
            return {
                "research_elements": results[0],
                "methodology_analysis": results[1],
                "statistical_results": results[2],
                "research_contributions": results[3],
                "citation_analysis": results[4],
                "quality_assessment": results[5],
                "research_questions": results[6],
                "limitations": results[7],
                "academic_metrics": self._calculate_academic_metrics(full_text),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in research paper analysis: {e}")
            return {"error": str(e)}

    async def _extract_research_elements(self, sections: Dict[str, str], full_text: str) -> List[ResearchElement]:
        """Extract key research elements from the paper"""
        elements = []
        
        # Extract hypotheses
        hypotheses = self._extract_hypotheses(full_text)
        for hypothesis in hypotheses:
            elements.append(ResearchElement(
                element_type="hypothesis",
                content=hypothesis,
                confidence=0.8,
                metadata={"extraction_method": "pattern_matching"}
            ))
        
        # Extract research objectives
        objectives = self._extract_objectives(sections)
        for objective in objectives:
            elements.append(ResearchElement(
                element_type="objective",
                content=objective,
                confidence=0.7
            ))
        
        # Extract key findings
        findings = self._extract_key_findings(sections.get("results", ""))
        for finding in findings:
            elements.append(ResearchElement(
                element_type="finding",
                content=finding,
                confidence=0.6,
                section="results"
            ))
        
        return elements

    async def _analyze_methodology(self, sections: Dict[str, str], full_text: str) -> Dict[str, Any]:
        """Analyze research methodology"""
        methodology_text = sections.get("methodology", sections.get("methods", ""))
        
        # Classify methodology type
        methodology_type = self._classify_methodology(methodology_text)
        
        # Assess rigor
        rigor_score = self._assess_methodology_rigor(methodology_text)
        
        # Extract sample size
        sample_size = self._extract_sample_size(methodology_text)
        
        # Identify data collection methods
        data_methods = self._identify_data_methods(methodology_text)
        
        # Check for reproducibility indicators
        reproducibility = self._check_reproducibility(methodology_text)
        
        return {
            "methodology_type": methodology_type,
            "rigor_score": rigor_score,
            "sample_size": sample_size,
            "data_collection_methods": data_methods,
            "reproducibility_indicators": reproducibility,
            "methodology_completeness": self._assess_completeness(methodology_text)
        }

    async def _extract_statistical_results(self, full_text: str) -> List[StatisticalResult]:
        """Extract statistical results and significance tests"""
        results = []
        
        # Extract p-values
        p_values = re.findall(self.stat_patterns["p_value"], full_text, re.IGNORECASE)
        for p_val in p_values:
            try:
                value = float(p_val.replace("p", "").replace("<", "").replace("=", "").strip())
                results.append(StatisticalResult(
                    test_type="p_value",
                    value=value,
                    p_value=value,
                    significance=value < 0.05
                ))
            except ValueError:
                continue
        
        # Extract correlation coefficients
        correlations = re.findall(self.stat_patterns["correlation"], full_text, re.IGNORECASE)
        for corr in correlations:
            try:
                value = float(corr)
                results.append(StatisticalResult(
                    test_type="correlation",
                    value=value,
                    significance=abs(value) > 0.3  # Moderate correlation threshold
                ))
            except ValueError:
                continue
        
        # Extract t-test results
        t_tests = re.findall(self.stat_patterns["t_test"], full_text, re.IGNORECASE)
        for t_val in t_tests:
            try:
                value = float(t_val)
                results.append(StatisticalResult(
                    test_type="t_test",
                    value=value,
                    significance=abs(value) > 1.96  # Approximate threshold
                ))
            except ValueError:
                continue
        
        return results

    async def _identify_research_contributions(self, sections: Dict[str, str], full_text: str) -> List[ResearchContribution]:
        """Identify novel research contributions"""
        contributions = []
        
        # Analyze novelty claims
        novelty_patterns = [
            r"novel\s+\w+",
            r"first\s+time",
            r"new\s+approach",
            r"innovative\s+\w+",
            r"unprecedented\s+\w+",
            r"breakthrough\s+\w+"
        ]
        
        for pattern in novelty_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                context = self._extract_context(full_text, match.start(), 200)
                contributions.append(ResearchContribution(
                    contribution_type="methodological",
                    description=context,
                    novelty_score=0.7,
                    evidence=[match.group()],
                    impact_assessment="moderate"
                ))
        
        # Analyze conclusion section for contributions
        conclusion_text = sections.get("conclusion", "")
        if conclusion_text:
            contribution_sentences = self._extract_contribution_sentences(conclusion_text)
            for sentence in contribution_sentences:
                contributions.append(ResearchContribution(
                    contribution_type="theoretical",
                    description=sentence,
                    novelty_score=0.6,
                    evidence=[],
                    impact_assessment="to_be_determined"
                ))
        
        return contributions

    async def _analyze_citations(self, full_text: str) -> Dict[str, Any]:
        """Analyze citation patterns and references"""
        # Extract in-text citations
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
            r'\[\d+[,\s\-\d]*\]',    # [1, 2, 3]
            r'\([^)]*et\s+al[^)]*\)', # (Smith et al., 2023)
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, full_text)
            citations.extend(matches)
        
        # Analyze citation density
        words = len(full_text.split())
        citation_density = len(citations) / words if words > 0 else 0
        
        # Extract recent citations (last 5 years)
        recent_years = [str(year) for year in range(2019, 2024)]
        recent_citations = [c for c in citations if any(year in c for year in recent_years)]
        
        return {
            "total_citations": len(citations),
            "citation_density": citation_density,
            "recent_citations": len(recent_citations),
            "citation_recency_ratio": len(recent_citations) / len(citations) if citations else 0,
            "sample_citations": citations[:10]
        }

    async def _assess_research_quality(self, sections: Dict[str, str], full_text: str) -> Dict[str, Any]:
        """Assess overall research quality"""
        quality_indicators = {
            "has_abstract": bool(sections.get("abstract")),
            "has_methodology": bool(sections.get("methodology") or sections.get("methods")),
            "has_results": bool(sections.get("results")),
            "has_discussion": bool(sections.get("discussion")),
            "has_conclusion": bool(sections.get("conclusion")),
            "has_references": bool(sections.get("references"))
        }
        
        # Calculate completeness score
        completeness_score = sum(quality_indicators.values()) / len(quality_indicators)
        
        # Assess writing quality
        readability_score = textstat.flesch_reading_ease(full_text)
        
        # Check for standard academic structure
        structure_score = self._assess_structure_quality(sections)
        
        return {
            "quality_indicators": quality_indicators,
            "completeness_score": completeness_score,
            "readability_score": readability_score,
            "structure_score": structure_score,
            "overall_quality": (completeness_score + structure_score) / 2
        }

    async def _extract_research_questions(self, sections: Dict[str, str], full_text: str) -> List[str]:
        """Extract research questions from the paper"""
        questions = []
        
        # Pattern-based extraction
        question_patterns = [
            r'research question[s]?[:\-\s]+([^.]*\?)',
            r'we ask[:\-\s]+([^.]*\?)',
            r'investigate[s]?\s+whether[:\-\s]+([^.]*)',
            r'aim[s]?\s+to\s+determine[:\-\s]+([^.]*)'
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            questions.extend(matches)
        
        # Look in introduction section specifically
        intro_text = sections.get("introduction", "")
        question_sentences = [s for s in sent_tokenize(intro_text) if '?' in s]
        questions.extend(question_sentences)
        
        return list(set(questions))  # Remove duplicates

    async def _analyze_limitations(self, sections: Dict[str, str], full_text: str) -> Dict[str, Any]:
        """Identify and analyze research limitations"""
        limitation_patterns = [
            r'limitation[s]?',
            r'constraint[s]?',
            r'shortcoming[s]?',
            r'caveat[s]?',
            r'restriction[s]?'
        ]
        
        limitations = []
        for pattern in limitation_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                context = self._extract_context(full_text, match.start(), 300)
                limitations.append(context)
        
        # Check discussion section for limitations
        discussion_text = sections.get("discussion", "")
        limitation_sentences = []
        if discussion_text:
            sentences = sent_tokenize(discussion_text)
            for sentence in sentences:
                if any(pattern in sentence.lower() for pattern in limitation_patterns):
                    limitation_sentences.append(sentence)
        
        return {
            "identified_limitations": limitations,
            "limitation_sentences": limitation_sentences,
            "limitation_count": len(limitations) + len(limitation_sentences),
            "discusses_limitations": len(limitations) > 0 or len(limitation_sentences) > 0
        }

    def _calculate_academic_metrics(self, full_text: str) -> Dict[str, Any]:
        """Calculate various academic writing metrics"""
        return {
            "word_count": len(full_text.split()),
            "sentence_count": len(sent_tokenize(full_text)),
            "paragraph_count": len([p for p in full_text.split('\n\n') if p.strip()]),
            "avg_sentence_length": len(full_text.split()) / len(sent_tokenize(full_text)),
            "readability_score": textstat.flesch_reading_ease(full_text),
            "grade_level": textstat.flesch_kincaid_grade(full_text),
            "lexical_diversity": len(set(word_tokenize(full_text.lower()))) / len(word_tokenize(full_text.lower()))
        }

    # Helper methods
    def _extract_hypotheses(self, text: str) -> List[str]:
        """Extract hypotheses from text"""
        hypothesis_patterns = [
            r'hypothes[ie]s?\s*[:]\s*([^.]*)',
            r'we hypothesize\s+that\s+([^.]*)',
            r'our hypothesis\s+is\s+that\s+([^.]*)',
            r'predict\s+that\s+([^.]*)'
        ]
        
        hypotheses = []
        for pattern in hypothesis_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            hypotheses.extend(matches)
        
        return hypotheses

    def _extract_objectives(self, sections: Dict[str, str]) -> List[str]:
        """Extract research objectives"""
        objectives = []
        
        # Look in abstract and introduction
        for section_name in ["abstract", "introduction"]:
            section_text = sections.get(section_name, "")
            
            objective_patterns = [
                r'objective[s]?\s*[:]\s*([^.]*)',
                r'aim[s]?\s*[:]\s*([^.]*)',
                r'goal[s]?\s*[:]\s*([^.]*)',
                r'purpose\s*[:]\s*([^.]*)'
            ]
            
            for pattern in objective_patterns:
                matches = re.findall(pattern, section_text, re.IGNORECASE)
                objectives.extend(matches)
        
        return objectives

    def _extract_key_findings(self, results_text: str) -> List[str]:
        """Extract key findings from results section"""
        if not results_text:
            return []
        
        sentences = sent_tokenize(results_text)
        
        # Look for sentences with significance indicators
        significance_indicators = [
            'significant', 'p <', 'p=', 'correlation', 'effect',
            'difference', 'increase', 'decrease', 'higher', 'lower'
        ]
        
        key_findings = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in significance_indicators):
                key_findings.append(sentence.strip())
        
        return key_findings

    def _classify_methodology(self, methodology_text: str) -> str:
        """Classify the type of methodology used"""
        text_lower = methodology_text.lower()
        
        scores = {}
        for method_type, indicators in self.methodology_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            scores[method_type] = score
        
        if not scores or max(scores.values()) == 0:
            return "unclassified"
        
        return max(scores, key=scores.get)

    def _assess_methodology_rigor(self, methodology_text: str) -> float:
        """Assess the rigor of the methodology"""
        rigor_indicators = [
            "controlled", "randomized", "blind", "placebo", "validated",
            "reliability", "validity", "statistical power", "sample size",
            "confidence interval", "effect size"
        ]
        
        text_lower = methodology_text.lower()
        score = sum(1 for indicator in rigor_indicators if indicator in text_lower)
        
        # Normalize to 0-1 scale
        return min(score / len(rigor_indicators), 1.0)

    def _extract_sample_size(self, methodology_text: str) -> Optional[int]:
        """Extract sample size from methodology"""
        patterns = [
            r'n\s*=\s*(\d+)',
            r'sample\s+size\s+of\s+(\d+)',
            r'(\d+)\s+participants',
            r'(\d+)\s+subjects'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, methodology_text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None

    def _identify_data_methods(self, methodology_text: str) -> List[str]:
        """Identify data collection methods"""
        methods = [
            "survey", "questionnaire", "interview", "observation",
            "experiment", "measurement", "recording", "analysis"
        ]
        
        text_lower = methodology_text.lower()
        identified_methods = [method for method in methods if method in text_lower]
        
        return identified_methods

    def _check_reproducibility(self, methodology_text: str) -> Dict[str, bool]:
        """Check for reproducibility indicators"""
        indicators = {
            "code_available": any(term in methodology_text.lower() for term in ["code", "software", "github"]),
            "data_available": any(term in methodology_text.lower() for term in ["data available", "dataset", "repository"]),
            "detailed_procedure": len(methodology_text.split()) > 200,
            "statistical_methods": any(term in methodology_text.lower() for term in ["statistical", "analysis", "test"])
        }
        
        return indicators

    def _assess_completeness(self, methodology_text: str) -> float:
        """Assess completeness of methodology description"""
        required_elements = [
            "participants", "procedure", "materials", "analysis",
            "design", "measures", "variables", "statistical"
        ]
        
        text_lower = methodology_text.lower()
        present_elements = sum(1 for element in required_elements if element in text_lower)
        
        return present_elements / len(required_elements)

    def _extract_context(self, text: str, position: int, window: int = 200) -> str:
        """Extract context around a position in text"""
        start = max(0, position - window // 2)
        end = min(len(text), position + window // 2)
        return text[start:end].strip()

    def _extract_contribution_sentences(self, conclusion_text: str) -> List[str]:
        """Extract sentences that describe contributions"""
        contribution_patterns = [
            "contribute", "contribution", "novel", "first", "advance",
            "improve", "enhance", "demonstrate", "show", "establish"
        ]
        
        sentences = sent_tokenize(conclusion_text)
        contribution_sentences = []
        
        for sentence in sentences:
            if any(pattern in sentence.lower() for pattern in contribution_patterns):
                contribution_sentences.append(sentence.strip())
        
        return contribution_sentences

    def _assess_structure_quality(self, sections: Dict[str, str]) -> float:
        """Assess the quality of paper structure"""
        expected_sections = ["abstract", "introduction", "methodology", "results", "discussion", "conclusion"]
        present_sections = [section for section in expected_sections if section in sections]
        
        structure_score = len(present_sections) / len(expected_sections)
        
        # Bonus for having references
        if "references" in sections:
            structure_score += 0.1
        
        return min(structure_score, 1.0) 