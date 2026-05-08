"""
Intent Classifier — Week 2 query routing.

Classifies user queries into intent categories:
  - list: skill/project/cert listing
  - summary: profile summary or overview
  - comparison: resume vs JD comparison
  - gap_analysis: skill-gap / missing skills
  - improvement: improvement suggestions
  - detail: general document question

Uses rule-based pattern matching for simplicity and explainability.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent patterns (checked in priority order)
# ---------------------------------------------------------------------------

_COMPARISON_PATTERNS = [
    r'(?i)\b(?:compare|comparison|match|matching|fit|suitable|aligned)\b',
    r'(?i)\b(?:resume|cv)\b.*\b(?:job|jd|position|role|description)\b',
    r'(?i)\b(?:job|jd|position|role)\b.*\b(?:resume|cv)\b',
    r'(?i)\bhow\s+(?:well|good|closely)\b.*\b(?:match|fit|align)\b',
    r'(?i)\b(?:does|do)\s+(?:the\s+)?(?:resume|candidate|profile)\s+(?:match|fit|align)\b',
    r'(?i)\b(?:matched|matching)\s+skills?\b',
    r'(?i)\brelevance\b',
]

_GAP_PATTERNS = [
    r'(?i)\b(?:gap|gaps|missing|lacking|absent|shortfall)\b',
    r'(?i)\bwhat\b.*\b(?:missing|lack|need|don.t have|doesn.t have)\b',
    r'(?i)\b(?:skill|skills)\s+(?:gap|gaps)\b',
    r'(?i)\b(?:not\s+(?:have|possess|mention|include|list))\b',
    r'(?i)\b(?:unmet|unmatched)\b',
]

_IMPROVEMENT_PATTERNS = [
    r'(?i)\bimprov\w*\b',
    r'(?i)\b(?:suggest|suggestion|recommend|recommendation)\b',
    r'(?i)\bhow\s+(?:can|could|should|to)\s+(?:the\s+)?(?:resume|candidate|profile)\b.*\b(?:better|enhanc|strengthen)\b',
    r'(?i)\bwhat\s+(?:should|can|could)\b.*\b(?:add|include|learn|develop|upskill)\b',
    r'(?i)\b(?:tips?|advice|feedback)\b.*\b(?:resume|profile)\b',
]

_LIST_PATTERNS = [
    r'(?i)\b(?:what|list|name|show|which)\b.*?\b(?:skills?|tools?|technologies?|projects?|certificat(?:es?|ions?)|languages?|education|experience|achievements?|awards?)\b',
    r'(?i)\b(?:skills?|projects?|certificat(?:es?|ions?)|tools?|technologies?)\s+(?:are|were)\b',
]

_SUMMARY_PATTERNS = [
    r'(?i)\bsummar(?:y|ize|ise)\b',
    r'(?i)\boverview\b',
    r'(?i)\btell\s+me\s+about\b',
    r'(?i)\bdescribe\b',
    r'(?i)\bwho\s+is\b',
    r'(?i)\bwhat\s+(?:is|does)\s+(?:the|this)\s+(?:document|resume|profile|candidate)',
    r'(?i)\bbrief(?:ly)?\b',
    r'(?i)\bin\s+(?:two|2|few|short)\s+(?:lines?|sentences?|words?)',
]

# Section keyword mapping (same as Week 1, centralized here)
SECTION_KEYWORDS = {
    "skills": "SKILLS",
    "skill": "SKILLS",
    "tools": "SKILLS",
    "technologies": "SKILLS",
    "tech stack": "SKILLS",
    "programming": "SKILLS",
    "languages": "LANGUAGES",
    "projects": "PROJECTS",
    "project": "PROJECTS",
    "certificates": "CERTIFICATES",
    "certifications": "CERTIFICATES",
    "certificate": "CERTIFICATES",
    "certification": "CERTIFICATES",
    "experience": "EXPERIENCE",
    "work": "EXPERIENCE",
    "education": "EDUCATION",
    "profile": "PROFILE",
    "summary": "PROFILE",
    "contact": "CONTACT",
    "achievements": "ACHIEVEMENTS",
    "awards": "ACHIEVEMENTS",
    "interests": "INTERESTS",
    "hobbies": "INTERESTS",
}


def classify_intent(query: str) -> str:
    """
    Classify a user query into an intent category.

    Priority order: comparison > gap_analysis > improvement > list > summary > detail

    Args:
        query: The user's question string.

    Returns:
        Intent string: 'comparison', 'gap_analysis', 'improvement',
                        'list', 'summary', or 'detail'.
    """
    if not query or not query.strip():
        return "detail"

    # Check in priority order
    for pattern in _COMPARISON_PATTERNS:
        if re.search(pattern, query):
            logger.info("Intent classified as 'comparison' for: '%s...'", query[:50])
            return "comparison"

    for pattern in _GAP_PATTERNS:
        if re.search(pattern, query):
            logger.info("Intent classified as 'gap_analysis' for: '%s...'", query[:50])
            return "gap_analysis"

    for pattern in _IMPROVEMENT_PATTERNS:
        if re.search(pattern, query):
            logger.info("Intent classified as 'improvement' for: '%s...'", query[:50])
            return "improvement"

    for pattern in _LIST_PATTERNS:
        if re.search(pattern, query):
            logger.info("Intent classified as 'list' for: '%s...'", query[:50])
            return "list"

    for pattern in _SUMMARY_PATTERNS:
        if re.search(pattern, query):
            logger.info("Intent classified as 'summary' for: '%s...'", query[:50])
            return "summary"

    logger.info("Intent classified as 'detail' for: '%s...'", query[:50])
    return "detail"


def detect_target_section(query: str) -> str | None:
    """
    Detect which document section the query is asking about.

    Args:
        query: The user's question string.

    Returns:
        Section label string (e.g., 'SKILLS') or None.
    """
    query_lower = query.lower()
    for keyword, section in SECTION_KEYWORDS.items():
        if keyword in query_lower:
            return section
    return None


def needs_jd(intent: str) -> bool:
    """
    Check if an intent requires a job description to produce a meaningful answer.

    Args:
        intent: The classified intent string.

    Returns:
        True if the intent needs JD context.
    """
    return intent in ("comparison", "gap_analysis", "improvement")
