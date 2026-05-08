# SkillSync AI — Week 2 CONTEXT.md

## 0) Purpose of this file
This document is the source of truth for **Week 2** of SkillSync AI.

Any coding agent, including Antigravity, must read this file first and obey it strictly before modifying code.

Week 2 is about turning the Week 1 document-Q&A foundation into a **resume-to-JD analysis engine** with **retrieval, prompt tuning, citations, skill-gap analysis, and refresh workflow**.

Week 2 is **not** the place for heavy agents, memory systems, or overengineering. The goal is to improve usefulness, accuracy, and explainability while keeping the system stable.

---

## 1) Project identity
**Project name:** SkillSync AI

**Project type:** Resume-to-Job Description Match Copilot

**Domain:** Careers / GenAI

**Focus:** Structured document intelligence for resumes and job descriptions

**Core idea:**
SkillSync AI compares a resume against a job description, identifies matched skills, missing skills, and mismatch areas, then produces a clear, evidence-backed analysis.

---

## 2) Week 1 baseline status
Week 1 should already provide:
- document upload
- parsing and text extraction
- cleaning
- chunking
- knowledge base creation
- basic Q&A
- section-aware answers for common structured document questions

Week 2 builds on top of this and should **not break** the working Week 1 pipeline.

---

## 3) Week 2 goal in plain language
Week 2 should make SkillSync AI understand **both the resume and the job description** and produce a meaningful comparison.

The system should answer questions such as:
- What skills match the job description?
- What skills are missing from the resume?
- Which experience is relevant?
- What parts of the resume support the match?
- How can the resume be improved for this job?

The system must remain grounded in retrieved document evidence.

---

## 4) Official Week 2 scope from the project brief
According to the assignment brief, Week 2 implementation should focus on:
- adding retrieval
- prompt tuning
- citations
- an admin refresh workflow

Week 2 testing should focus on:
- answer relevance
- citation correctness
- edge queries
- fallback handling

This is the target scope. Do not drift into unrelated advanced features.

---

## 5) What Week 2 should achieve
Week 2 should convert the app from:

**“document Q&A”**

into:

**“resume vs JD analysis copilot”**

The output should be:
- more structured
- more relevant
- more explainable
- citation-backed
- robust to edge queries

---

## 6) Week 2 deliverables
The Week 2 build must include the following functional capabilities:

### A. Resume + JD retrieval
The system must be able to work with both:
- uploaded resume content
- job description input or uploaded JD document

### B. Skill-gap analysis
The system must identify:
- matched skills
- partially matched skills
- missing skills
- optional skills
- evidence supporting each result

### C. Prompt tuning
The system must use improved prompts for:
- list questions
- summary questions
- match analysis
- gap analysis
- improvement suggestions

### D. Citations
The system must display evidence from the source chunks so the answer is explainable and trustable.

### E. Admin refresh workflow
The system must support refreshing the knowledge base when a new resume or JD is uploaded.

### F. Edge-case handling
The system must safely respond when:
- the JD is missing
- the resume is missing
- no strong match exists
- retrieval is weak
- query is ambiguous

---

## 7) Primary Week 2 user journeys
The app should support these flows:

### Flow 1: Resume only Q&A
User uploads a resume and asks structured questions.

### Flow 2: Resume + JD comparison
User uploads a resume and enters a JD. The system compares both and returns a match analysis.

### Flow 3: Gap analysis
The system identifies missing or weakly represented skills in the resume compared to the JD.

### Flow 4: Improvement suggestions
The system suggests what the candidate should add or improve.

---

## 8) Required outputs for Week 2
The system should be able to produce:
- matched skills list
- missing skills list
- relevance explanation
- short summary of fit
- evidence chunks
- confidence indicator when meaningful

The output should be concise and clearly structured.

---

## 9) Recommended Week 2 tech approach
Use the existing Week 1 stack and add only what is necessary.

### Core stack
- Streamlit UI
- ChromaDB vector store
- sentence-transformers embeddings
- document parsing utilities
- FastAPI if the app already uses it, otherwise do not force a rewrite

### Logic additions
- query intent detection
- section-aware retrieval
- prompt templates for each answer type
- normalization of extracted items
- citation formatting
- refresh/rebuild logic

### Optional
- LangChain can be used if it already helps the current implementation
- do not add unnecessary dependencies just for complexity

---

## 10) Week 2 retrieval design
The system should not just search by similarity.

It should use a hybrid retrieval strategy:
- section-aware filtering where possible
- semantic retrieval for supportive evidence
- fallback to broader search if section-based search fails

For example:
- resume skill query → Skills section
- project query → Projects section
- JD comparison query → compare resume chunks with JD chunks

The retriever should always prefer the most relevant section first.

---

## 11) Week 2 prompt design goals
Prompt tuning is a core Week 2 requirement.

Prompts must be designed to:
- answer only from retrieved context
- avoid hallucination
- keep output concise
- preserve document-grounded accuracy
- show matching / missing / supporting evidence when asked

Prompts should be different for:
- summary
- item listing
- skill-gap analysis
- explanation generation
- fallback/no-answer cases

---

## 12) Citations and evidence policy
Week 2 must emphasize evidence-backed responses.

### Required citation behavior
- Every important answer should show supporting evidence.
- Evidence should point to the most relevant chunks.
- Citations must not be random or misleading.
- If evidence is weak, the system should say so.

### Evidence quality rules
- Prefer the correct section over a generic chunk.
- Avoid mixing unrelated sections.
- Avoid overloading the answer with too many chunks.

---

## 13) Skill-gap analysis rules
Skill-gap analysis should be practical and explainable.

### The system should detect:
- exact keyword matches
- semantically related matches
- missing exact requirements
- likely missing skills
- optional or nice-to-have skills

### The system should not:
- invent skills not present in the resume
- overstate the match
- claim experience that is not in the document

### Output format suggestion
- Match summary
- Matched skills
- Missing skills
- Suggested improvements
- Evidence

---

## 14) Query intent categories for Week 2
The system should classify the query into one of these broad types:
- skill listing
- project listing
- certificate listing
- profile summary
- resume vs JD comparison
- skill-gap analysis
- improvement suggestions
- general document question

The intent classifier can be rule-based, semantic, or hybrid, but it must remain simple and explainable.

---

## 15) Admin refresh workflow requirement
The app should support a clean reset/rebuild flow.

This is important because Week 2 may involve switching between:
- resume only
- resume + JD
- fresh reanalysis

The refresh workflow should:
- clear old state
- rebuild the relevant knowledge base
- avoid stale retrieval results
- prevent mixed session confusion

---

## 16) Suggested Week 2 file structure
A clean structure may look like this:

```text
skillsync-ai/
├── app.py
├── requirements.txt
├── README.md
├── .env
├── data/
│   ├── uploads/
│   └── cache/
├── src/
│   ├── parser.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── intent.py
│   ├── qa.py
│   ├── matcher.py
│   ├── citations.py
│   └── refresh.py
└── chroma_db/
```

This is only a suggested structure. Use what fits the existing codebase best.

---

## 17) Week 2 implementation priorities
The correct order is:

1. stabilize the Week 1 pipeline
2. add JD ingestion/input support
3. add intent classification
4. improve retrieval routing
5. implement skill-gap analysis logic
6. add prompt tuning
7. add citations and evidence formatting
8. add refresh/rebuild workflow
9. test edge cases and fallback behavior

---

## 18) Week 2 testing plan
Test the system with the following categories:

### A. Answer relevance
- Does the response actually answer the question?
- Is it grounded in the correct section or JD chunk?

### B. Citation correctness
- Do the cited chunks really support the answer?
- Are citations from the most relevant source?

### C. Edge queries
- vague question
- no-answer question
- out-of-scope question
- mixed query

### D. Fallback handling
- missing JD
- missing resume
- weak retrieval
- empty section

### E. Comparison quality
- matched skills are accurate
- missing skills are not invented
- summary is concise and honest

---

## 19) Acceptance criteria for Week 2
Week 2 is complete only if all of the following are true:

- the system can compare resume and JD or at least support JD-based analysis with the existing resume knowledge base
- retrieval is more relevant than Week 1
- citations are shown correctly
- prompt tuning improves answer quality
- edge queries are handled gracefully
- refresh/rebuild workflow works
- the system remains stable and explainable

---

## 20) Quality standards
The Week 2 implementation should be:
- accurate
- explainable
- section-aware
- citation-backed
- concise
- stable
- easy to demo
- easy to defend in viva

Prefer clarity over cleverness.

---

## 21) Non-goals for Week 2
Do not spend time on:
- complex multi-agent orchestration
- heavy memory systems
- long-term personalization
- production auth systems
- enterprise-scale infrastructure
- unnecessary AI wrappers
- overcomplicated chain logic

The assignment expects a working, explainable project, not a research platform.

---

## 22) Week 2 engineering mindset
Think in this order:

**Can the app compare resume and JD meaningfully?**

**Can it explain why a skill matches or does not match?**

**Can it cite the right evidence?**

**Can it handle missing or weak data without hallucinating?**

If the answer to all four is yes, Week 2 is successful.

---

## 23) Notes for future weeks
Week 2 should prepare the project for:
- report generation
- user progress history
- exportable analysis
- UI polish
- final demo stability

Do not design Week 2 in a way that blocks Week 3.


