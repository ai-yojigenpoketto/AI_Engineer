# Job Skills Analysis Search Guideline
*A systematic methodology for collecting and analyzing job market skill requirements*

---

## Core Principle

**Be hypothesis-driven, not data-driven.**

Don't just collect whatever generic searches return. Instead:
1. Identify what **should** be in the requirements (based on industry trends)
2. Search specifically to validate or discover those requirements
3. Question gaps and iterate

---

## Pre-Search Phase: Context Building

### Step 1: Identify Current Industry Trends (15-20 minutes)

**Search Queries:**
```
1. "[Role] trends [Current Year]"
   Example: "AI Engineer trends 2025"

2. "What's new in [Field] [Current Year]"
   Example: "What's new in AI 2025"

3. "[Field] word of the year [Current Year]"
   Example: "AI word of the year 2025"

4. "Emerging [Field] technologies [Current Year]"
   Example: "Emerging AI technologies 2025"

5. "Future of [Role]"
   Example: "Future of AI engineering"
```

**Goal**: Build a hypothesis list of technologies/skills that **should** appear in job requirements.

**Output**: Create a checklist
```
Expected Skills (to validate):
☐ [Technology/Framework A]
☐ [Technology/Framework B]
☐ [Emerging trend C]
☐ [Tool/Platform D]
```

### Step 2: Identify Key Players (10 minutes)

**Find:**
- Companies building the cutting-edge frameworks/tools
- AI-native startups in the space
- Consulting firms doing implementation work
- Research labs pushing boundaries

**Search Queries:**
```
1. "Top [Field] companies [Current Year]"
2. "[Specific Technology] company"
   Example: "LangGraph company" → LangChain
3. "Who created [Framework]"
4. "[Technology] startups [Current Year]"
```

**Goal**: Know where to look for bleeding-edge requirements.

---

## Phase 1: Framework/Technology-Specific Searches (High Priority)

### Why This Matters
Specific technology searches find **actual usage requirements**, not generic summaries.

### Search Query Template

```
For each technology/framework identified in pre-search:

1. "[Technology] job requirements"
2. "[Technology] engineer skills"
3. "site:greenhouse.io [Technology]"
4. "site:lever.co [Technology]"
5. "[Technology] developer position"
6. "experience with [Technology]" job description
```

### Example: If "LangGraph" was identified as emerging trend

```
✓ "LangGraph job requirements"
✓ "LangGraph engineer skills"
✓ "site:greenhouse.io LangGraph"
✓ "site:lever.co LangGraph"
✓ "experience with LangGraph" job
```

### Coverage Strategy
- Search for **at least 5-7 specific technologies** per role
- Mix established (TensorFlow) and emerging (LangGraph) technologies
- Include adjacent technologies (if researching AI agents, also search for vector databases)

---

## Phase 2: AI-Native Company Job Boards (High Priority)

### Target Sources (in priority order)

**Tier 1: Companies Building the Technology**
```
Examples for AI Agents:
- CrewAI careers
- LangChain hiring
- Anthropic careers
- OpenAI careers
- Mistral AI jobs
```

**Search Pattern:**
1. Find the company building the framework
2. Go directly to their careers page
3. Read actual job descriptions for that role

**Why**: These companies know exactly what skills are needed because they're inventing them.

### Tier 2: Startup Job Boards
```
1. site:greenhouse.io "[Role]"
2. site:lever.co "[Role]"
3. site:ashbyhq.com "[Role]"
```

**Why**: Startups adopt new technologies faster than enterprises.

### Tier 3: Specialized Job Sites
```
1. YCombinator jobs - ycombinator.com/companies (filter by role)
2. AI-specific job boards
3. Tech-specific communities (HackerNews "Who's Hiring" threads)
```

---

## Phase 3: Cutting-Edge Consulting & Services (Medium-High Priority)

### Target Firms
Consulting firms implementing bleeding-edge technology for clients:

```
Search Queries:
1. "Deloitte [Emerging Tech] engineer"
   Example: "Deloitte agentic AI engineer"

2. "Accenture [Technology] consultant"
3. "McKinsey [Field] roles"
4. "[Big4] [Technology] job"
```

### Why This Works
- Consultants work with multiple clients
- They see cross-industry skill demands
- Job descriptions reflect what clients are requesting
- Often have specialized roles (e.g., "Agentic AI Engineer")

---

## Phase 4: Big Tech Companies (Lower Priority for Emerging Skills)

### Use Cases
- Validating established skills
- Understanding baseline requirements
- Checking if emerging skills have reached mainstream

### Search Approach
```
1. site:careers.google.com "[Role]"
2. site:amazon.jobs "[Role]"
3. site:microsoft.com/careers "[Role]"
4. "[BigTech] [Role] requirements"
```

### Warning
⚠️ **Big Tech job descriptions lag 6-12 months behind cutting edge**
- Use for baseline, not for discovering emerging trends
- Don't rely solely on these

---

## Phase 5: Job Description Aggregators (Validation Only)

### When to Use
- **After** completing Phases 1-4
- For validation and statistical frequency
- For understanding baseline/established requirements

### Sources
```
1. Indeed job templates
2. LinkedIn job insights
3. Glassdoor salary guides (if needed)
4. General career sites
```

### Warning
⚠️ **Never start here** - these lag significantly behind reality

---

## Phase 6: Technical Communities & Forums (Context)

### Sources
```
1. Reddit: r/MachineLearning, r/LocalLLaMA, r/ArtificialIntelligence
2. HackerNews: "Who's Hiring" threads
3. Twitter/X: Follow key practitioners and companies
4. Discord/Slack: Framework-specific communities
5. GitHub: Check popular repos for "We're hiring" in README
```

### What to Look For
- What technologies practitioners are excited about
- What frameworks are trending in discussions
- What skills experienced engineers recommend
- Real-world usage patterns

---

## Search Query Formulas

### Formula 1: Technology Intersection
```
"[Role]" "[Tech A]" OR "[Tech B]" OR "[Tech C]" requirements

Example:
"AI Engineer" "LangGraph" OR "AutoGen" OR "CrewAI" requirements
```

### Formula 2: Emerging Terminology
```
"[Emerging Term]" engineer OR developer job description

Example:
"Agentic AI" engineer OR developer job description
```

### Formula 3: Framework-Specific with Context
```
"[Framework]" "[Related Skill]" job

Example:
"LangGraph" "multi-agent" job
"AutoGen" "orchestration" engineer
```

### Formula 4: Negative Filtering (Important!)
```
"[Role]" -"[Excluded Role]" requirements

Example:
"AI Engineer" -"Machine Learning Engineer" requirements

Use this to avoid contamination from related but different roles.
```

### Formula 5: Recency Filter
```
"[Role]" requirements [Current Year]
"[Role]" skills [Current Year]

Example:
"AI Engineer" requirements 2025
```

### Formula 6: Production/Applied Focus
```
"[Role]" "production" OR "applied" OR "deployed"

Example:
"AI Engineer" "production LLM" OR "deployed agents"
```

---

## Red Flags & Warning Signs

### Signs Your Search is Missing Critical Information

🚩 **Red Flag 1**: All results are from aggregator sites (Indeed, Workable, etc.)
- **Action**: Search for specific company job boards

🚩 **Red Flag 2**: No mention of frameworks/tools released in past 12 months
- **Action**: Search specifically for those technologies by name

🚩 **Red Flag 3**: All results are from Big Tech companies
- **Action**: Search startup job boards (Greenhouse, Lever)

🚩 **Red Flag 4**: Generic skills only (Python, ML, Cloud)
- **Action**: Search for specialized/emerging skills

🚩 **Red Flag 5**: No specialized roles (all generic "AI Engineer")
- **Action**: Search for variations: "Agentic AI Engineer", "GenAI Engineer", "LLM Engineer"

🚩 **Red Flag 6**: Same content across multiple sources
- **Action**: Sources are likely copying each other; find primary sources

🚩 **Red Flag 7**: No mention of technologies you know are trending
- **Action**: Search explicitly for those technologies

---

## Critical Evaluation Checklist

After completing your search, ask:

### Trend Alignment
- [ ] Do results reflect known industry trends?
- [ ] Are emerging technologies mentioned?
- [ ] Are frameworks from the past year included?

### Source Diversity
- [ ] Did I check AI-native companies?
- [ ] Did I check startup job boards?
- [ ] Did I search for specific technologies?
- [ ] Did I balance new and established companies?

### Depth vs. Breadth
- [ ] Did I go deep on specific technologies, not just skim surface?
- [ ] Did I read actual job postings, not just summaries?
- [ ] Did I fetch full job descriptions, not just titles?

### Hypothesis Validation
- [ ] Did I validate my initial hypothesis of what should be included?
- [ ] Did I search specifically for gaps I identified?
- [ ] Did I iterate when something seemed missing?

### Recency
- [ ] Are results from current year?
- [ ] Did I filter out outdated information?
- [ ] Did I prioritize recently posted jobs?

---

## Iteration Strategy

### When to Iterate

**Trigger 1**: Known technology is missing
```
Action: Search explicitly for that technology
Example: "No agent frameworks found" → Search "LangGraph job" specifically
```

**Trigger 2**: Results seem generic/outdated
```
Action: Switch to more specific sources
Example: Generic "ML skills" → Search AI-native startup postings
```

**Trigger 3**: Lack of specialized roles
```
Action: Search for role variations
Example: Generic "AI Engineer" → Add "Agentic AI Engineer", "LLM Engineer"
```

**Trigger 4**: Source homogeneity
```
Action: Diversify source types
Example: All templates → Search actual company career pages
```

### Iteration Pattern

```
Initial Search → Evaluate → Identify Gap → Targeted Search → Re-evaluate

Example Flow:
1. Generic "AI Engineer skills" search
2. Notice no agents mentioned
3. Search "AI agents engineer requirements"
4. Find agentic AI roles
5. Search for specific frameworks (LangGraph, AutoGen)
6. Comprehensive results achieved
```

---

## Common Pitfalls to Avoid

### ❌ Pitfall 1: "Breadth Over Depth"
**Mistake**: Collecting many generic sources
**Fix**: Deep dive into specific technologies and specialized roles

### ❌ Pitfall 2: "Template Trap"
**Mistake**: Relying on job description templates
**Fix**: Read actual active job postings from real companies

### ❌ Pitfall 3: "Big Tech Bias"
**Mistake**: Only checking FAANG companies
**Fix**: Prioritize AI-native startups and consulting firms

### ❌ Pitfall 4: "Generic Query Syndrome"
**Mistake**: Using only broad searches like "AI Engineer skills"
**Fix**: Search for specific frameworks and technologies by name

### ❌ Pitfall 5: "Aggregator Dependence"
**Mistake**: Starting with Indeed, Glassdoor summaries
**Fix**: Use these for validation only, after primary research

### ❌ Pitfall 6: "Accepting First Results"
**Mistake**: Not questioning what might be missing
**Fix**: Always ask "What's conspicuously absent?"

### ❌ Pitfall 7: "Historical Data"
**Mistake**: Using data from 6+ months ago
**Fix**: Force recency in searches, target current postings

### ❌ Pitfall 8: "Role Contamination"
**Mistake**: Including related but different roles (ML Engineer when researching AI Engineer)
**Fix**: Use negative filters in searches

---

## Special Considerations by Field

### For Fast-Moving Fields (AI, Web3, etc.)

**Accelerated Timeline**
- What's 6 months old may already be outdated
- Framework landscape changes rapidly
- New job titles emerge constantly

**Search Adjustments**
1. Increase weight on Phases 1-2 (technology-specific, AI-native companies)
2. Reduce weight on Phase 5 (aggregators)
3. Check technical communities weekly for new frameworks
4. Search for framework-specific job requirements immediately when new tools launch

### For Established Fields (Backend Engineering, DevOps, etc.)

**Stable Core + Emerging Edge**
- Core skills change slowly
- But tooling and practices evolve

**Search Adjustments**
1. Balance Phase 4 (Big Tech) and Phase 2 (Startups)
2. Look for tooling updates (new monitoring tools, IaC platforms, etc.)
3. Focus on "modern X engineer" to find updated practices

---

## Output Structure Template

After completing search, structure findings as:

### 1. Executive Summary
- Role definition
- Scope (what's included/excluded)
- Key findings (2-3 sentences)
- Critical emerging trends

### 2. Core Technical Skills
Organized by category with frequency data:
- Programming languages (with %)
- Frameworks (established vs. emerging)
- Tools and platforms
- Specializations

### 3. Emerging/Critical Skills Section ⭐
**Explicitly call out:**
- What's new in the past 12 months
- What's trending but not yet in all postings
- What industry experts are emphasizing
- The "gap" between generic postings and cutting edge

### 4. Experience & Education
- Degree requirements
- Years of experience by level
- Specific experience areas

### 5. Ranked Summary
- Top 10 technical skills
- Top 10 knowledge areas
- Critical differentiators

### 6. Skills by Specialization
- Different role variants within the title
- What distinguishes each

### 7. Reflection Section
**Include:**
- Source diversity analysis
- Gaps identified and how addressed
- Confidence level in findings
- Limitations of the analysis

---

## Time Budget Recommendations

For comprehensive analysis of a single role:

```
Pre-Search Phase:          30 minutes
Phase 1 (Tech-specific):   45 minutes
Phase 2 (AI-native cos):   30 minutes
Phase 3 (Consulting):      20 minutes
Phase 4 (Big Tech):        15 minutes
Phase 5 (Aggregators):     15 minutes
Phase 6 (Communities):     15 minutes
Iteration (2-3 rounds):    30 minutes
Analysis & Compilation:    45 minutes
---
Total:                     ~4 hours
```

### Quick Analysis (1 hour)
Focus on:
- Phases 1 & 2 (30 min)
- One iteration (15 min)
- Compilation (15 min)

### Deep Analysis (8+ hours)
Add:
- Fetching and reading full job descriptions
- Cross-referencing multiple sources per finding
- Competitive analysis across similar roles
- Historical trend analysis

---

## Quality Indicators

### High-Quality Analysis Includes:

✅ Specific framework/tool names with versions
✅ Emerging technologies from past 12 months
✅ Mix of source types (startup, enterprise, consulting, native)
✅ Clear distinction between established and emerging skills
✅ Actual job posting examples cited
✅ Explanation of why certain skills matter
✅ Identification of gaps in generic postings
✅ Confidence levels for different findings

### Low-Quality Analysis Indicators:

❌ Only generic skills (Python, Cloud, ML)
❌ All sources are templates or aggregators
❌ No specific framework names
❌ No technologies from past year
❌ No emerging trends identified
❌ Can't cite actual job postings
❌ No critical evaluation of sources

---

## Tool Use Guidelines

### When to Use Different Tools

**WebSearch Tool**:
- Initial trend identification
- Framework-specific searches
- Company-specific queries
- Validation searches

**WebFetch Tool**:
- Reading full job descriptions from direct URLs
- Extracting detailed requirements from career pages
- Getting complete information from specific postings

**Task Tool (job-market-analyzer agent)**:
- When you need to scrape many postings at scale
- For statistical analysis across large datasets
- When building comprehensive databases
- **Note**: May hit rate limits; have manual fallback ready

### Parallel vs. Sequential

**Run in Parallel**:
- Multiple WebSearch queries for different technologies
- Different source types (startup vs. consulting)
- Variations of same query

**Run Sequentially**:
- Initial search → Evaluate gaps → Targeted follow-up
- Fetch specific URLs (after getting URLs from search)
- Iteration rounds

---

## Validation Checklist

Before finalizing analysis:

### Source Validation
- [ ] Did I check at least 3 different source types?
- [ ] Did I read at least 5 actual job postings (not templates)?
- [ ] Did I verify emerging technologies from multiple sources?

### Completeness Validation
- [ ] Are established frameworks covered? (TensorFlow, PyTorch, etc.)
- [ ] Are emerging frameworks covered? (LangGraph, AutoGen, etc.)
- [ ] Are related infrastructure skills covered? (Vector DBs, cloud, etc.)
- [ ] Are soft skills mentioned?

### Trend Validation
- [ ] Did I search for "[Field] trends [Year]"?
- [ ] Do my findings align with known industry trends?
- [ ] Did I identify and explain gaps?

### Recency Validation
- [ ] Are sources from current or past year?
- [ ] Did I filter out outdated information?
- [ ] Are emerging technologies from past 12 months included?

### Critical Thinking
- [ ] Did I question my initial results?
- [ ] Did I iterate when something seemed missing?
- [ ] Can I explain why certain skills are important?
- [ ] Did I distinguish between "nice to have" and "required"?

---

## Example: Applying This Guideline

### Scenario: Analyzing "AI Engineer" Skills

**Pre-Search**:
```
✓ Search "AI trends 2025" → Find: "Agentic AI is the word of the year"
✓ Hypothesis: Should include agent frameworks
```

**Phase 1 (Tech-specific)**:
```
✓ "LangGraph job requirements"
✓ "AutoGen engineer position"
✓ "CrewAI developer skills"
✓ "RAG implementation" job
✓ "Vector database" engineer
```

**Phase 2 (AI-native)**:
```
✓ LangChain careers page
✓ Anthropic careers
✓ site:greenhouse.io "AI Engineer"
✓ CrewAI jobs
```

**Phase 3 (Consulting)**:
```
✓ "Deloitte agentic AI engineer"
✓ "Accenture GenAI consultant"
```

**Evaluation**:
```
Found: Agents, LangGraph, AutoGen, CrewAI
Missing: Nothing major
Iterate: Search for "function calling" to confirm tool use skills
```

**Result**: Comprehensive analysis including emerging agent skills

---

## Meta-Guideline: Updating This Document

This guideline should be updated when:

1. **New search patterns prove effective**
   - Add to search query formulas section
   - Document what worked and why

2. **New job boards/sources emerge**
   - Add to source hierarchy
   - Classify by tier (1, 2, 3)

3. **New pitfalls identified**
   - Add to common pitfalls section
   - Include how to avoid

4. **Field-specific nuances discovered**
   - Add to special considerations
   - Document unique approaches

---

## Quick Reference Card

**Before Every Search:**
1. What are current industry trends? (Search "[Field] trends [Year]")
2. What specific technologies should I search for?
3. Which companies are building those technologies?

**Core Search Pattern:**
1. Technology-specific searches (Highest priority)
2. AI-native company job boards
3. Cutting-edge consulting firms
4. Big Tech (for baseline)
5. Aggregators (for validation only)

**Red Flags:**
- All generic skills (Python, ML, Cloud)
- No frameworks from past year
- Only template sources
- Known trends are missing

**Always Ask:**
- What's missing?
- Why might this be absent?
- Where would cutting-edge requirements appear?

**Remember:**
🎯 Hypothesis-driven > Data-driven
🎯 Specific > Generic
🎯 Primary sources > Aggregators
🎯 Question everything > Accept first results

---

**Document Version**: 1.0
**Created**: December 2025
**Purpose**: Systematic methodology for discovering true job market requirements, not just generic templates
**Key Principle**: Be relentlessly specific and question everything
