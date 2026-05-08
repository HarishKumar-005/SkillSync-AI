"""
Skill-Gap Matcher — Compares resume skills against JD requirements.

Approach:
  1. Extract skill keywords from resume chunks and JD text
  2. Exact match: case-insensitive keyword comparison
  3. Semantic match: embedding cosine similarity for related skills
  4. Classify into: matched, partially_matched, missing

Does NOT invent skills or overclaim matches.
"""

import re
import logging
from src.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# Similarity threshold for semantic (partial) matches
SEMANTIC_MATCH_THRESHOLD = 0.65

# Common filler words to strip when extracting skills
_FILLER = {
    "and", "or", "the", "a", "an", "of", "in", "for", "with", "to",
    "on", "at", "by", "is", "are", "etc", "such", "as", "including",
    "experience", "knowledge", "proficiency", "proficient", "familiar",
    "strong", "good", "excellent", "required", "preferred", "must",
    "have", "years", "year", "ability", "understanding",
}


def extract_skills_from_text(text: str) -> list[str]:
    """
    Extract individual skill/technology keywords from a text block.

    Works for both resume skill sections and JD requirement lists.

    Args:
        text: Raw text containing skills or requirements.

    Returns:
        Deduplicated list of skill strings.
    """
    if not text or not text.strip():
        return []

    skills = []
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line or len(line) < 2:
            continue

        # Skip lines that look like section headers
        if len(line) < 40 and line.isupper():
            continue

        # Remove bullet prefixes
        clean_line = re.sub(r'^[\-\•\*\▪\►\➤\→\‣\⁃]\s*', '', line)
        clean_line = re.sub(r'^\d+[\.)\]]\s*', '', clean_line)

        # If line has a colon, take the values after it
        if ':' in clean_line:
            parts = clean_line.split(':', 1)
            values_text = parts[1].strip()
            if values_text:
                clean_line = values_text

        # Split by common delimiters
        items = re.split(r'[,;|/]', clean_line)

        for item in items:
            item = item.strip().rstrip('.')

            # Split by " and " or " & "
            sub_items = re.split(r'\s+(?:and|&)\s+', item, flags=re.IGNORECASE)

            for sub in sub_items:
                sub = sub.strip()

                # Remove parenthetical content like "(3 years)"
                sub = re.sub(r'\(.*?\)', '', sub).strip()

                # Skip if too short, too long, or just filler
                if len(sub) < 2 or len(sub) > 60:
                    continue
                if sub.lower() in _FILLER:
                    continue

                # Skip full sentences (likely descriptions, not skills)
                word_count = len(sub.split())
                if word_count > 5:
                    continue

                skills.append(sub)

    # Deduplicate preserving order
    seen = set()
    unique = []
    for skill in skills:
        key = skill.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(skill)

    return unique


def match_skills(
    resume_skills: list[str],
    jd_skills: list[str],
    use_semantic: bool = True,
) -> dict:
    """
    Compare resume skills against JD requirements.

    Args:
        resume_skills: List of skills extracted from the resume.
        jd_skills: List of skills/requirements extracted from the JD.
        use_semantic: Whether to use embedding similarity for partial matches.

    Returns:
        Dict with keys:
          - matched: list of (jd_skill, resume_skill) exact matches
          - partially_matched: list of (jd_skill, resume_skill, similarity) semantic matches
          - missing: list of jd_skill with no match
          - match_summary: brief text summary
    """
    if not jd_skills:
        return {
            "matched": [],
            "partially_matched": [],
            "missing": [],
            "match_summary": "No job description requirements provided for comparison.",
        }

    if not resume_skills:
        return {
            "matched": [],
            "partially_matched": [],
            "missing": list(jd_skills),
            "match_summary": "No resume skills found for comparison.",
        }

    # Build lowercase lookup for resume skills
    resume_lower = {s.lower().strip(): s for s in resume_skills}

    matched = []
    partially_matched = []
    missing = []
    matched_jd_skills = set()

    # --- Phase 1: Exact matching ---
    for jd_skill in jd_skills:
        jd_lower = jd_skill.lower().strip()

        # Direct match
        if jd_lower in resume_lower:
            matched.append((jd_skill, resume_lower[jd_lower]))
            matched_jd_skills.add(jd_lower)
            continue

        # Substring match (e.g., "React" matches "React.js")
        found = False
        for r_lower, r_original in resume_lower.items():
            if jd_lower in r_lower or r_lower in jd_lower:
                matched.append((jd_skill, r_original))
                matched_jd_skills.add(jd_lower)
                found = True
                break

        if not found and jd_lower not in matched_jd_skills:
            missing.append(jd_skill)

    # --- Phase 2: Semantic matching for remaining missing skills ---
    if use_semantic and missing:
        try:
            model = get_embedding_model()
            missing_embeddings = model.encode(missing, normalize_embeddings=True)
            resume_list = list(resume_lower.values())
            resume_embeddings = model.encode(resume_list, normalize_embeddings=True)

            still_missing = []
            for i, jd_skill in enumerate(missing):
                # Compute cosine similarity with all resume skills
                similarities = (missing_embeddings[i] @ resume_embeddings.T).tolist()
                best_idx = max(range(len(similarities)), key=lambda x: similarities[x])
                best_sim = similarities[best_idx]

                if best_sim >= SEMANTIC_MATCH_THRESHOLD:
                    partially_matched.append((jd_skill, resume_list[best_idx], round(best_sim, 2)))
                else:
                    still_missing.append(jd_skill)

            missing = still_missing

        except Exception as e:
            logger.warning("Semantic matching failed, using exact matches only: %s", e)

    # Build summary
    total = len(jd_skills)
    n_matched = len(matched)
    n_partial = len(partially_matched)
    n_missing = len(missing)
    match_pct = ((n_matched + n_partial * 0.5) / total * 100) if total > 0 else 0

    match_summary = (
        f"**Match Score: {match_pct:.0f}%** — "
        f"{n_matched} exact match{'es' if n_matched != 1 else ''}, "
        f"{n_partial} partial match{'es' if n_partial != 1 else ''}, "
        f"{n_missing} missing out of {total} JD requirements."
    )

    logger.info(
        "Skill matching: %d matched, %d partial, %d missing out of %d JD skills",
        n_matched, n_partial, n_missing, total,
    )

    return {
        "matched": matched,
        "partially_matched": partially_matched,
        "missing": missing,
        "match_summary": match_summary,
    }
