"""
Prompt Templates — Structured answer formatters for Week 2.

Each template enforces document-grounded output and produces
clean, structured responses for different query types.

No external LLM is used — these templates format extractive answers.
"""

import logging

logger = logging.getLogger(__name__)


def format_list_answer(items: list[str], subject: str, total: int | None = None) -> str:
    """
    Format a list of items into a clean bulleted answer.

    Args:
        items: List of item strings.
        subject: What the items are (e.g., 'skills/technologies').
        total: Override total count. Defaults to len(items).

    Returns:
        Formatted answer string.
    """
    if not items:
        return f"No {subject} were found in the document."

    count = total or len(items)
    lines = [f"The document mentions the following **{subject}**:\n"]
    for item in items:
        lines.append(f"- {item}")
    lines.append(f"\n*Total: {count} {subject} found.*")
    return "\n".join(lines)


def format_summary_answer(summary_text: str) -> str:
    """
    Format a summary answer.

    Args:
        summary_text: The summarized content.

    Returns:
        Formatted summary string.
    """
    if not summary_text:
        return "No profile information found in the document to summarize."

    return f"**Profile Summary:**\n\n{summary_text.strip()}"


def format_comparison_answer(match_result: dict) -> str:
    """
    Format a resume-vs-JD comparison answer.

    Args:
        match_result: Result dict from matcher.match_skills() with keys:
            matched, partially_matched, missing, match_summary.

    Returns:
        Formatted comparison answer string.
    """
    lines = ["## 📊 Resume vs Job Description Analysis\n"]

    # Match summary
    lines.append(match_result.get("match_summary", ""))
    lines.append("")

    # Matched skills
    matched = match_result.get("matched", [])
    if matched:
        lines.append(f"### ✅ Matched Skills ({len(matched)})")
        for jd_skill, resume_skill in matched:
            if jd_skill.lower() == resume_skill.lower():
                lines.append(f"- {jd_skill}")
            else:
                lines.append(f"- {jd_skill} ↔ {resume_skill}")
        lines.append("")

    # Partially matched
    partial = match_result.get("partially_matched", [])
    if partial:
        lines.append(f"### 🔶 Partially Matched ({len(partial)})")
        for jd_skill, resume_skill, sim in partial:
            lines.append(f"- {jd_skill} ≈ {resume_skill} ({sim:.0%} similarity)")
        lines.append("")

    # Missing
    missing = match_result.get("missing", [])
    if missing:
        lines.append(f"### ❌ Missing Skills ({len(missing)})")
        for skill in missing:
            lines.append(f"- {skill}")
        lines.append("")

    if not matched and not partial:
        lines.append("\n⚠️ *No strong skill matches found between the resume and job description.*")

    return "\n".join(lines)


def format_gap_analysis_answer(match_result: dict) -> str:
    """
    Format a skill-gap analysis focused answer.

    Args:
        match_result: Result dict from matcher.match_skills().

    Returns:
        Formatted gap analysis string.
    """
    lines = ["## 🔍 Skill Gap Analysis\n"]

    # Summary
    lines.append(match_result.get("match_summary", ""))
    lines.append("")

    # Missing skills (primary focus)
    missing = match_result.get("missing", [])
    if missing:
        lines.append(f"### Skills to Develop ({len(missing)})")
        for skill in missing:
            lines.append(f"- ⬜ {skill}")
        lines.append("")
    else:
        lines.append("✅ **No critical skill gaps identified.** The resume covers all JD requirements.\n")

    # Partial matches (secondary)
    partial = match_result.get("partially_matched", [])
    if partial:
        lines.append(f"### Skills to Strengthen ({len(partial)})")
        for jd_skill, resume_skill, sim in partial:
            lines.append(f"- 🔶 {jd_skill} — related skill found: _{resume_skill}_")
        lines.append("")

    # Strong matches (for context)
    matched = match_result.get("matched", [])
    if matched:
        lines.append(f"### Already Strong ({len(matched)})")
        skills_str = ", ".join(jd for jd, _ in matched[:10])
        if len(matched) > 10:
            skills_str += f", +{len(matched) - 10} more"
        lines.append(f"- ✅ {skills_str}")
        lines.append("")

    return "\n".join(lines)


def format_improvement_answer(match_result: dict) -> str:
    """
    Format actionable improvement suggestions.

    Args:
        match_result: Result dict from matcher.match_skills().

    Returns:
        Formatted improvement suggestions string.
    """
    lines = ["## 💡 Improvement Suggestions\n"]

    missing = match_result.get("missing", [])
    partial = match_result.get("partially_matched", [])

    if not missing and not partial:
        lines.append(
            "The resume already covers the key requirements from the job description. "
            "Consider adding more detail to existing experiences to strengthen the match.\n"
        )
        return "\n".join(lines)

    if missing:
        lines.append("### Priority Skills to Add")
        lines.append("Consider gaining experience or certification in:\n")
        for skill in missing[:8]:
            lines.append(f"- **{skill}** — mentioned in JD but not found in resume")
        if len(missing) > 8:
            lines.append(f"- ...and {len(missing) - 8} more")
        lines.append("")

    if partial:
        lines.append("### Skills to Highlight Better")
        lines.append("These related skills were found but could be more explicitly stated:\n")
        for jd_skill, resume_skill, sim in partial:
            lines.append(f"- Emphasize **{jd_skill}** — you have _{resume_skill}_ which is related")
        lines.append("")

    lines.append("### General Tips")
    lines.append("- Use the exact terminology from the job description where accurate")
    lines.append("- Quantify achievements related to the required skills")
    lines.append("- Ensure the most relevant experience is prominently placed")

    return "\n".join(lines)


def format_fallback_answer(reason: str, query: str = "") -> str:
    """
    Format a graceful fallback response when we can't produce a good answer.

    Args:
        reason: Why the answer couldn't be generated.
        query: Original query for context.

    Returns:
        Formatted fallback string.
    """
    fallback_messages = {
        "no_resume": (
            "📄 **No resume uploaded.** Please upload a resume (PDF or DOCX) "
            "using the sidebar to get started."
        ),
        "no_jd": (
            "📋 **No job description provided.** To compare your resume against a job, "
            "please paste the job description in the sidebar text area."
        ),
        "weak_retrieval": (
            "🔍 I couldn't find strong evidence in the document to answer this question. "
            "The uploaded document may not contain information related to your query. "
            "Try rephrasing or asking about specific sections."
        ),
        "no_match": (
            "⚠️ I found the document but couldn't extract a meaningful answer. "
            "The query might be too vague or the information might not be present in the document."
        ),
        "out_of_scope": (
            "ℹ️ This question appears to be outside the scope of the uploaded document. "
            "I can only answer questions based on the content of the uploaded resume "
            "and job description."
        ),
    }

    return fallback_messages.get(reason, fallback_messages["no_match"])


def format_detail_answer(text: str, section: str = "GENERAL") -> str:
    """
    Format a detail answer from chunk text.

    Args:
        text: The relevant text content.
        section: The document section it came from.

    Returns:
        Formatted detail answer string.
    """
    section_display = section.replace("_", " ").title()
    return f"**From the {section_display} section:**\n\n{text}"
