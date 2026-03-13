"""Prompt templates for workflow roles."""

# ──────────────────────────────────────────────────────────────────────
# Learn phase prompt (used by learn_cmd.py)
# ──────────────────────────────────────────────────────────────────────

ARCHITECT_LEARN_PROJECT = """You are a senior architect analyzing a new software project.
Your goal is to extract the operational knowledge that development and QA agents need
to work effectively on this codebase.

## Discovered Project Files

{file_contents}

## Agent Instruction Files

The following files (CLAUDE.md, AGENTS.md, etc.) are typically written for single-agent
AI tools. Extract useful project knowledge from them — conventions, security rules,
coding patterns, forbidden practices, important invariants. But IGNORE any workflow
directives (e.g. "when asked to implement X, first do Y then Z") since our multi-agent
workflow is managed separately.

{agent_instructions}

## Your Deliverable

Output a JSON object with the following structure. Every field is required.

```json
{{
  "project_type": "<short descriptor, e.g. 'rust-database', 'python-web-app'>",
  "language": "<primary language>",
  "build": {{
    "commands": ["<command to build the project>"],
    "prerequisites": ["<required tools/versions>"],
    "notes": "<any special build instructions>"
  }},
  "test": {{
    "tiers": [
      {{
        "name": "<test tier name>",
        "command": "<exact command to run>",
        "scope": "<what it tests>"
      }}
    ],
    "gate_command": "<command that runs ALL required checks before merge, or empty string>",
    "notes": "<any special test instructions>"
  }},
  "git": {{
    "branch_convention": "<branching model or 'unknown'>",
    "commit_style": "<commit message convention or 'unknown'>",
    "pr_checks": ["<required CI checks>"]
  }},
  "structure": {{
    "entry_points": ["<main entry point files>"],
    "key_directories": {{"<dir>": "<purpose>"}},
    "notes": "<any structural observations>"
  }},
  "conventions": {{
    "formatter": "<formatter tool or 'none'>",
    "linter": "<linter tool or 'none'>",
    "rules": ["<important coding rules extracted from docs/config>"]
  }},
  "security": {{
    "requirements": ["<security requirements from docs>"],
    "forbidden_patterns": ["<patterns explicitly forbidden>"]
  }},
  "ci": {{
    "provider": "<CI provider or 'unknown'>",
    "key_jobs": ["<important CI job names>"],
    "notes": "<any CI-specific notes>"
  }},
  "invariants": [
    "<documented correctness rules or architectural constraints>"
  ]
}}
```

Output ONLY the JSON object. No other text before or after it.
If a section has no information, use empty arrays/objects/strings — do NOT omit the key.
"""

# ──────────────────────────────────────────────────────────────────────
# Legacy prompts (used by review_loop.py for ad-hoc /review command)
# ──────────────────────────────────────────────────────────────────────

DEVELOPER_INITIAL = """You are a senior software developer working on a coding task.

{context}

## Task
{task}

## Instructions
1. Analyze the task requirements carefully.
2. Produce a complete solution including:
   - Design rationale for your approach
   - Full implementation code
   - Test plan or test code
3. Be thorough — your output will be reviewed by a QA engineer.
4. If the task involves modifying existing code, show the complete changes.
"""

DEVELOPER_REVISION = """You are a senior software developer addressing review feedback.

{context}

## Original Task
{task}

## Your Previous Submission
{previous_artifact}

## Review Feedback
{feedback}

## Instructions
1. Address each issue raised in the review feedback.
2. Provide the updated complete solution.
3. Explain what changes you made and why.
4. If you disagree with any feedback, explain your reasoning.
"""

REVIEWER = """You are a senior QA engineer reviewing a developer's work.

{context}

## Task Being Reviewed
{task}

## Developer's Submission (Iteration {iteration})
{artifact}

## Instructions
Review the submission for:
1. **Correctness**: Does it solve the stated task? Any logic errors?
2. **Edge cases**: Are boundary conditions handled?
3. **Code quality**: Is it clean, readable, maintainable?
4. **Test coverage**: Are tests adequate?
5. **Security**: Any potential vulnerabilities?

## Required Output Format
You MUST output your verdict as a JSON object (and nothing else after it):

```json
{{
  "verdict": "lgtm" or "changes_requested",
  "issues": ["issue 1 description", "issue 2 description"],
  "summary": "Brief overall assessment"
}}
```

If everything looks good with no issues, use "lgtm" with an empty issues list.
If there are problems, use "changes_requested" with specific actionable issues.
"""

# ──────────────────────────────────────────────────────────────────────
# Workflow engine prompts (used by workflow/engine.py)
# ──────────────────────────────────────────────────────────────────────

# ── Stage 1: Technical Design ──

DEV_TECH_DESIGN = """You are a senior developer. Produce a technical design proposal.

## Requirement
{requirement}

## Your Deliverable
Produce a complete technical design including:
1. Architecture overview and approach
2. Key design decisions and trade-offs
3. API/interface design (if applicable)
4. Data model changes (if applicable)
5. Dependencies and risks
6. Testing strategy

Be thorough — this design will be reviewed by QA before implementation begins.
"""

DEV_ADDRESS_DESIGN_COMMENTS = """Address the following review comments on your technical design.

## Your Previous Design
{design}

## Review Comments
{feedback}

## Instructions
1. Address each comment specifically.
2. Update the design where needed.
3. Explain your reasoning for any disagreements.
4. Provide the complete updated design.
"""

ARCH_TECH_DESIGN = """You are the project architect. Based on the prior discussion, produce a formal
technical design proposal.

## Requirement
{requirement}

## Your Deliverable
Produce a complete technical design including:
1. Problem statement and goals
2. Architecture overview and approach
3. Key design decisions and trade-offs
4. Component interfaces and data flow
5. Dependencies and risks
6. Open questions (if any)

Be thorough — this design will be reviewed by the development and QA teams.
"""

ARCH_ADDRESS_DESIGN_COMMENTS = """Address the following review comments on your technical design.

## Your Previous Design
{design}

## Review Comments
{feedback}

## Instructions
1. Address each comment specifically.
2. Update the design where needed.
3. Explain your reasoning for any disagreements.
4. Provide the complete updated design.
"""

DESIGN_REVIEW = """Review the following technical design proposal.

{context}

## Design Proposal (iteration {iteration})
{design}

## Review Criteria
1. **Completeness** — Does it fully address the requirement?
2. **Correctness** — Any logical or architectural issues?
3. **Testability** — Can this design be tested effectively?
4. **Edge cases** — Are boundary conditions considered?
5. **Risks** — Any concerns about maintainability, performance, security?

## Required Output Format
Output ONLY a JSON object:

```json
{{
  "verdict": "lgtm" or "changes_requested",
  "issues": ["issue 1", "issue 2"],
  "summary": "Brief overall assessment"
}}
```
"""

QA_DESIGN_REVIEW = """You are a QA engineer reviewing a technical design proposal.

{context}

## Design Proposal (iteration {iteration})
{design}

## Review Criteria
1. **Completeness** — Does it fully address the requirement?
2. **Correctness** — Any logical or architectural issues?
3. **Testability** — Can this design be tested effectively?
4. **Edge cases** — Are boundary conditions considered?
5. **Risks** — Any concerns about maintainability, performance, security?

## Required Output Format
Output ONLY a JSON object:

```json
{{
  "verdict": "lgtm" or "changes_requested",
  "issues": ["issue 1", "issue 2"],
  "summary": "Brief overall assessment"
}}
```
"""

# ── Stage 3: Implementation Planning ──

DEV_IMPLEMENTATION_PLAN = """You are a senior developer. The following design has been approved.
Break it into implementation phases.

## Requirement
{requirement}

## Approved Design
{design}

## Your Deliverable
Break the implementation into sequential phases. Each phase should be:
- Independently implementable and testable
- Small enough for one focused coding session
- Committable on its own (no broken state between phases)

For each phase provide:
1. Phase name (short, descriptive)
2. What will be implemented
3. Expected files to create/modify
4. Dependencies on previous phases

Format as:

Phase 1: <name>
<description>

Phase 2: <name>
<description>

...
"""

DEV_ADDRESS_PLAN_COMMENTS = """Address the following review comments on your implementation plan.

## Your Previous Plan
{plan}

## Review Comments
{feedback}

## Instructions
Address each comment and provide the complete updated plan.
Keep the same Phase N: <name> format.
"""

QA_PLAN_REVIEW = """You are a QA engineer reviewing an implementation plan.

{context}

## Implementation Plan (iteration {iteration})
{plan}

## Review Criteria
1. **Completeness** — Does the plan cover the full approved design?
2. **Ordering** — Are phases in a logical sequence?
3. **Granularity** — Are phases appropriately sized (not too big, not too small)?
4. **Testability** — Can each phase be verified independently?
5. **Risks** — Any missing steps or overlooked dependencies?

## Required Output Format
Output ONLY a JSON object:

```json
{{
  "verdict": "lgtm" or "changes_requested",
  "issues": ["issue 1", "issue 2"],
  "summary": "Brief overall assessment"
}}
```
"""

# ── Stage 5: Phased Development ──

DEV_IMPLEMENT_PHASE = """You are implementing phase {phase_index} of {total_phases}.

## Requirement (brief)
{requirement}

## Approved Design (brief)
{design}

## Implementation Plan
{plan}

## Current Phase
Phase {phase_index}: {phase_name}

## Previously Completed Phases
{completed_phases}

## Instructions
1. Implement ONLY this phase according to the plan.
2. Write tests as appropriate.
3. Ensure the code compiles/runs correctly.
4. Do NOT implement future phases — focus only on this one.
"""

DEV_SUMMARIZE_PHASE = """Summarize what you just implemented in phase {phase_index} ({phase_name}).

Provide:
1. **Changes made** — what was implemented
2. **Files modified** — list of files created or changed
3. **Modules / areas affected** — which subsystems or behaviors changed
4. **Key decisions** — any implementation choices made
5. **Deviations** — anything that differs from the plan
6. **Risks / remaining concerns** — anything QA should pay special attention to
7. **QA focus** — what the reviewer should inspect directly in the codebase
8. **Test status** — what was tested and results
"""

DEV_ADDRESS_CODE_COMMENTS = """Address the following code review comments.

## Your Previous Summary
{summary}

## Review Comments
{feedback}

## Instructions
1. Fix each issue raised in the codebase.
2. Update the code accordingly.
3. Run tests to verify fixes.
4. Provide an updated summary of all changes.
"""

QA_CODE_REVIEW = """You are a QA engineer reviewing a development phase.

{context}

## Developer's Summary (iteration {iteration})
{summary}

## Instructions
1. Treat the developer's summary as a handoff, not as ground truth.
2. Examine the actual code and changes in the working directory yourself.
3. Check that tests exist and pass if possible.
4. Call out any mismatch between the summary and the code you inspected.
5. Provide your verdict.

## Review Criteria
1. **Correctness** — Does the code match the design and plan?
2. **Code quality** — Clean, readable, maintainable?
3. **Test coverage** — Are tests adequate?
4. **Edge cases** — Boundary conditions handled?
5. **Security** — Any vulnerabilities?

## Required Output Format
Output ONLY a JSON object:

```json
{{
  "verdict": "lgtm" or "changes_requested",
  "issues": ["issue 1", "issue 2"],
  "summary": "Brief overall assessment"
}}
```
"""

# ── Stage 5d: Commit ──

DEV_COMMIT_PHASE = """Commit the changes for phase {phase_index} ({phase_name}).

Run: git add -A && git commit -m "phase {phase_index}: {phase_name}"

If there are no staged changes to commit, just confirm that.
Do NOT push — only commit locally.
"""

# ── Stage 6: GA Test ──

QA_GA_TEST_PLAN = """You are a QA engineer. Generate a GA (General Acceptance) test plan for the
completed implementation.

## Requirement
{requirement}

## Approved Design
{design}

## Development Summary
{dev_summary}

## Your Deliverable
Create a comprehensive test plan covering:
1. **Functional tests** — verify each feature works as specified
2. **Integration tests** — verify components work together
3. **Edge cases** — boundary conditions, error paths
4. **Regression tests** — ensure existing functionality is not broken
5. **Performance tests** (if applicable)

For each test case provide:
- Test name
- Description and purpose
- Steps to execute
- Expected result
- Priority (critical/high/medium/low)

Format as a structured list of test cases.
"""

DEV_REVIEW_TEST_PLAN = """You are a senior developer reviewing a QA test plan.

{context}

## Test Plan (iteration {iteration})
{test_plan}

## Review Criteria
1. **Coverage** — Does the plan cover all implemented features?
2. **Correctness** — Are expected results accurate based on the design?
3. **Feasibility** — Can all tests actually be executed?
4. **Missing cases** — Any important scenarios not covered?
5. **Priority** — Are priorities reasonable?

## Required Output Format
Output ONLY a JSON object:

```json
{{
  "verdict": "lgtm" or "changes_requested",
  "issues": ["issue 1", "issue 2"],
  "summary": "Brief overall assessment"
}}
```
"""

TEST_PLAN_REVIEW = """Review the following test plan.

{context}

## Test Plan (iteration {iteration})
{test_plan}

## Review Criteria
1. **Coverage** — Does the plan cover all relevant features and scenarios?
2. **Correctness** — Are expected results accurate based on the design?
3. **Feasibility** — Can all tests actually be executed?
4. **Missing cases** — Any important scenarios not covered?
5. **Priority** — Are priorities reasonable?

## Required Output Format
Output ONLY a JSON object:

```json
{{
  "verdict": "lgtm" or "changes_requested",
  "issues": ["issue 1", "issue 2"],
  "summary": "Brief overall assessment"
}}
```
"""

QA_DESIGN_TEST_PLAN = """You are a QA engineer. Based on the prior discussion, produce a formal test plan
and test design.

## Requirement
{requirement}

## Your Deliverable
Produce a comprehensive test plan including:
1. Test scope and objectives
2. Test strategy
3. Test cases with expected results
4. Edge cases and negative scenarios
5. Test data requirements
6. Acceptance criteria

Be thorough — this plan will be reviewed by the architect and development teams.
"""

QA_ADDRESS_TEST_PLAN_COMMENTS = """Address the following review comments on your test plan.

## Your Previous Test Plan
{test_plan}

## Review Comments
{feedback}

## Instructions
Address each comment and provide the complete updated test plan.
"""

QA_EXECUTE_GA_TEST = """You are a QA engineer. Execute the approved test plan against the current codebase.

## Test Plan
{test_plan}

## Instructions
1. Execute each test case in the plan.
2. For each test, actually run the code or verify the implementation in the working directory.
3. Record pass/fail for each test case.
4. For any failure, capture detailed bug information.

## Required Output Format
Output ONLY a JSON object:

```json
{{
  "status": "pass" or "fail",
  "summary": "Overall test execution summary",
  "tests_run": 10,
  "tests_passed": 8,
  "tests_failed": 2,
  "bugs": [
    {{
      "title": "Brief bug title",
      "description": "Detailed description of the bug",
      "repro_steps": "Step-by-step reproduction instructions",
      "severity": "critical or major or minor",
      "test_case": "Which test case found this"
    }}
  ]
}}
```

If all tests pass, use "status": "pass" with an empty bugs list.
"""

QA_CONTINUE_GA_TEST = """You are a QA engineer continuing GA testing after bug fixes.

## Test Plan
{test_plan}

## Previously Found and Fixed Bugs
{fixed_bugs}

## Instructions
1. Re-run the failed test cases to verify the fixes.
2. Continue executing any remaining test cases.
3. Check for regressions — did the fixes break anything else?

## Required Output Format
Output ONLY a JSON object (same format as before):

```json
{{
  "status": "pass" or "fail",
  "summary": "Overall test execution summary",
  "tests_run": 10,
  "tests_passed": 10,
  "tests_failed": 0,
  "bugs": []
}}
```
"""

QA_GA_TEST_REPORT = """Provide a final GA test report.

## Test Plan
{test_plan}

## Test Execution History
{test_history}

## Instructions
Summarize:
1. Overall test results (pass rate)
2. Bugs found and their resolutions
3. Areas of concern or remaining risks
4. Recommendation (ready for release or not)
"""

DEV_ESTIMATE_BUG = """A bug was found during GA testing. Estimate the scope of this fix.

## Bug Details
**Title:** {bug_title}
**Description:** {bug_description}
**Reproduction Steps:** {repro_steps}
**Severity:** {severity}

## Current Requirement
{requirement}

## Current Design
{design}

## Instructions
Analyze this bug and determine if it requires:
- **simple_fix** — A localized code fix that doesn't change the architecture or design
- **arch_change** — A fundamental design change that requires redesigning part of the system

## Required Output Format
Output ONLY a JSON object:

```json
{{
  "assessment": "simple_fix" or "arch_change",
  "explanation": "Why this assessment",
  "estimated_scope": "What needs to change"
}}
```
"""

DEV_FIX_BUG_SIMPLE = """Fix the following bug found during GA testing.

## Bug Details
**Title:** {bug_title}
**Description:** {bug_description}
**Reproduction Steps:** {repro_steps}

## Instructions
1. Locate the root cause in the codebase.
2. Implement the fix.
3. Verify the fix resolves the issue.
4. Ensure no regressions are introduced.
5. Commit the fix: git add -A && git commit -m "fix: {bug_title}"
"""

DEV_SUMMARIZE_BUG_FIX = """Summarize the bug fix you just made.

## Bug
{bug_title}

## Provide
1. Root cause
2. What was changed
3. Files modified
4. How the fix was verified
"""

# ── Stage 7: Final Report ──

DEV_FINAL_REPORT = """Provide a final summary of the complete implementation.

## Completed Phases
{completed_phases}

## Instructions
Summarize:
1. What was built overall
2. Architecture of the final solution
3. Key files and their purposes
4. How to test/verify the implementation
5. Any known limitations or future work
"""
