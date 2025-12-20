# 🏭 CODE GENERATION PROTOCOL (CLAUDE.md)

## 🎯 CORE PHILOSOPHY & OBJECTIVES

### Mission

This project adopts the **"Zero Defect Manufacturing"** philosophy.
Just like in a high-precision automotive factory, every step is controlled, every component is verified, and every output is tested before integration.

### Fundamental Principles

```
┌─────────────────────────────────────────────────────────────────┐
│  1. PLAN  →  2. RESEARCH  →  3. EXECUTE  →  4. TEST  →  5. AUDIT    │
│      ↑                                                        ↓     │
│      └────────────── IF ERROR, ROLLBACK & REVISE ─────────────┘     │
└─────────────────────────────────────────────────────────────────┘

```

| Principle | Description | Why It Matters? |
| --- | --- | --- |
| **Plan First** | Never write code without a blueprint | Unplanned code = Technical Debt |
| **Verify Before Use** | Verify every API/Library before implementation | Prevention of Hallucinations |
| **Test Everything** | Untested code is not production code | Regression Prevention |
| **Explain Everything** | Every line must be educational | Sustainability & Maintainability |

---

## 🚨 CRITICAL RULES (NON-NEGOTIABLE)

Violating these rules is strictly prohibited. Each rule is designed to prevent a catastrophic failure scenario.

### Rule 1: NO HALLUCINATIONS

```
❌ WRONG: "I think this library has a .parse() method."
✅ RIGHT: Verify with Context7 → Read Documentation → Implement.

```

**Reason:** Incorrect API calls lead to runtime errors, security vulnerabilities, and data corruption.

### Rule 2: NEVER COMPROMISE TYPE SAFETY

```typescript
❌ WRONG: const data: any = response.json();
✅ RIGHT:
interface ApiResponse {
  users: User[];
  pagination: Pagination;
}
const data: ApiResponse = await response.json();

```

**Reason:** The `any` type disables the compiler's safety net.

### Rule 3: NO SILENT FAILURES

```python
# ❌ WRONG: Error swallowed, debugging impossible
try:
    process_data()
except:
    pass

# ✅ RIGHT: Error logged, context preserved
try:
    process_data()
except Exception as e:
    logger.error(f"process_data failed: {e}", exc_info=True)
    raise  # or handle gracefully

```

### Rule 4: NO APPROVAL WITHOUT TESTING

```
Code Written → Test PASS → Code Review → APPROVE
      ↓             ↓
   [CONTINUE]   [ERROR: LOOP BACK]

```

### Rule 5: NO DIRECT FILE READING (LARGE PROJECTS)

```
❌ WRONG: grep -r "functionName" .  (Slow, context-blind)
✅ RIGHT: Search symbol via Serena LSP (Fast, semantic)

```

---

## 🛠️ MCP TOOL ECOSYSTEM

Every tool solves a specific problem. Correct Tool + Correct Timing = Efficiency.

### 📊 Tool Selection Matrix

| Tool | Primary Task | When to Use? | Alternative |
| --- | --- | --- | --- |
| **Claude Task Master** | Task Planning | Project start, PRD Analysis | Sequential Thinking |
| **Claude-Flow** | Memory & Coordination | Multi-step workflows, Context switching | - |
| **Serena (LSP)** | Code Navigation | Symbol search, Definition lookup | grep (last resort) |
| **Context7** | API Documentation | Before using any library | Web search |
| **TestSprite** | Automated Testing | After code implementation | Manual test |
| **CodeRabbit** | Security Audit | Before PR, Delivery | SonarQube |
| **Sequential Thinking** | Complex Analysis | Multi-step reasoning | Task Master |
| **Tavily** | Web Research | Best practices, Error research | Web search |

---

### 🔧 TOOL 1: Claude Task Master

**Role:** Strategic Planner
**Analogy:** The Chief Engineer of a construction project; plans every step from foundation to roof.

#### When to Use

* [ ] Starting a new feature development
* [ ] Analyzing a PRD (Product Requirements Document)
* [ ] Planning a complex refactor
* [ ] Sprint planning

#### Usage Pattern

```
1. Receive PRD or Requirement
2. Send to Task Master
3. Generate tasks.json output
4. Create Dependency Map
5. Process tasks sequentially

```

#### Critical Warning

⚠️ **NEVER** skip Task Master. Projects started without a plan accumulate technical debt 80% of the time.

---

### 🧠 TOOL 2: Claude-Flow (Memory & Coordination)

**Role:** Project Memory & Orchestrator
**Analogy:** The Film Producer; coordinates the crew and ensures continuity.

#### When to Use

* [ ] To prevent context loss between tasks
* [ ] To reference previous decisions
* [ ] Managing multi-step, parallel jobs
* [ ] Querying past logic chains from the ReasoningBank

#### Concept Structure

```
┌─────────────────────────────────────────────────────────┐
│                   CLAUDE-FLOW                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ MEMORY      │  │ COORDINATION │  │ REASONING     │   │
│  │             │  │              │  │               │   │
│  │ • Decisions │  │ • Sub-agents │  │ • Why X?      │   │
│  │ • Context   │  │ • Workflow   │  │ • Alt?        │   │
│  │ • History   │  │ • Ordering   │  │ • Trade-offs  │   │
│  └─────────────┘  └──────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────┘

```

---

### 🔍 TOOL 3: Serena (LSP - Language Server Protocol)

**Role:** Code Navigator & Symbol Detective
**Analogy:** The Librarian; finds exactly what you are looking for instantly.

#### When to Use

* [ ] Finding a function definition
* [ ] Finding all usages of a variable
* [ ] Verifying import paths
* [ ] Impact analysis before refactoring

#### Why Serena over grep?

| Feature | grep | Serena (LSP) |
| --- | --- | --- |
| Speed (Large Projects) | Slow | Fast (Indexed) |
| Semantic Understanding | No | Yes |
| Type Information | No | Yes |
| Go to Definition | Manual | Automatic |
| Find References | Incomplete | Comprehensive |

---

### 📚 TOOL 4: Context7 (API Documentation)

**Role:** Library & Framework Expert
**Analogy:** The Official Manual – what the manufacturer says is always true.

#### When to Use

* [ ] Before using a new library
* [ ] Verifying API parameters
* [ ] Checking for breaking changes
* [ ] Learning best practices

#### Mandatory Workflow

```
┌──────────────────────────────────────────────────────────────┐
│        MANDATORY STEPS BEFORE LIBRARY USAGE                  │
│                                                              │
│  1. resolve-library-id   →  Find correct library ID          │
│            ↓                                                 │
│  2. get-library-docs     →  Fetch current API docs           │
│            ↓                                                 │
│  3. Implement Usage      →  Apply verified pattern           │
└──────────────────────────────────────────────────────────────┘

```

#### Critical Warning

⚠️ **NEVER** guess APIs. Do not write code without a Context7 result.
⚠️ Pay attention to framework versions (e.g., Next.js 13 vs 14 differences).

---

### 🧪 TOOL 5: TestSprite (Automated Testing)

**Role:** Quality Assurance Engineer
**Analogy:** Quality Control Unit; prevents defective products from shipping.

#### When to Use

* [ ] Upon writing a new function
* [ ] When modifying existing code (refactoring)
* [ ] For regression testing after bug fixes
* [ ] Running the full test suite before PR

#### Test Pyramid Strategy

* **Unit Tests:** 80% coverage (Function logic)
* **Integration Tests:** 60% coverage (API endpoints)
* **E2E Tests:** Critical flows only (Login, Checkout)

#### Critical Warning

⚠️ Do not proceed without **100% PASS**.
⚠️ Never drop test coverage below 80%.
⚠️ Fix flaky tests immediately.

---

### 🛡️ TOOL 6: CodeRabbit (Security Audit)

**Role:** Security Auditor & Code Quality Gatekeeper
**Analogy:** Building Inspector; detects structural issues before occupancy.

#### When to Use

* [ ] Before opening a Pull Request
* [ ] Before confirming task completion
* [ ] For regular security scans
* [ ] During code review

#### Audit Layers

* **🔴 CRITICAL SECURITY:** SQL Injection, XSS, CSRF, Hardcoded Credentials.
* **🟡 CODE QUALITY:** Anti-patterns, DRY violations, Complexity.
* **🟢 BEST PRACTICES:** Naming conventions, Error handling, Documentation.

#### Critical Warning

⚠️ **🔴 CRITICAL** findings must be fixed before merging.
⚠️ Run CodeRabbit for every PR.

---

### 🧩 TOOL 7: Sequential Thinking (Deep Analysis)

**Role:** Strategic Thought Partner
**Analogy:** The Grandmaster Chess Player; thinks several moves ahead.

#### When to Use

* [ ] Complex architectural decisions
* [ ] Trade-off analysis
* [ ] Multi-step problem solving
* [ ] Defining refactoring strategies

#### Thinking Process Structure

1. **Define Problem:** Identify the core issue.
2. **List Alternatives:** Option A vs Option B vs Option C.
3. **Trade-off Analysis:** Speed vs Complexity vs Scalability.
4. **Context Evaluation:** Current infrastructure, team capacity.
5. **Decision & Rationale:** Final choice with "Why".

---

### 🌐 TOOL 8: Tavily (Web Research)

**Role:** Research Assistant
**Analogy:** The Archivist; finds the most relevant external resources.

#### When to Use

* [ ] Best practices research (Current Year)
* [ ] Resolving obscure error messages
* [ ] Investigating community consensus
* [ ] Security vulnerability research (CVE)

#### Critical Warning

⚠️ Verify search results with Context7 where possible.
⚠️ Check dates on Stack Overflow/GitHub discussions.

---

## 🚀 MASTER WORKFLOW

This workflow applies to **every task**. No steps may be skipped.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          MASTER WORKFLOW                                   │
│                                                                            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │  PLAN   │───▶│  DOCS   │───▶│   NAV   │───▶│  CODE   │───▶│  TEST   │   │
│  │         │    │         │    │         │    │         │    │         │   │
│  │ Task    │    │Context7 │    │ Serena  │    │Implement│    │TestSprite│   │
│  │ Master  │    │ Tavily  │    │         │    │         │    │         │   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └────┬────┘   │
│       │                                                           │    │   │
│       │               ┌─────────┐    ┌─────────┐                  │    │   │
│       │               │  AUDIT  │◀───│  FAIL?  │◀─────────────────┘    │   │
│       │               │         │    │         │                       │   │
│       │               │CodeRabbit│    │  Y / N  │                       │   │
│       │               └────┬────┘    └────┬────┘                       │   │
│       │                    │              │                            │   │
│       │                    ▼              │ Y                          │   │
│       │               ┌─────────┐         │                            │   │
│       │               │  DONE   │◀────────┘ (N: Test Pass)             │   │
│       │               └─────────┘                                      │   │
│       │                                                                │   │
│       └─────────── Claude-Flow (Memory & Coordination) ────────────────▶   │
│                        Active throughout process                       │   │
└────────────────────────────────────────────────────────────────────────────┘

```

### STEP 1: PLAN (Mandatory Start)

**Tool:** Claude Task Master
**Goal:** Break task into atomic units.
**Output Criteria:** `tasks.json` created, dependencies mapped.

### STEP 2: DOCS (Research)

**Tools:** Context7, Tavily
**Goal:** Verify technologies and APIs.
**Output Criteria:** All APIs verified, code examples noted.

### STEP 3: NAV (Navigation)

**Tool:** Serena (LSP)
**Goal:** Understand the existing codebase.
**Output Criteria:** Inventory of existing patterns and component structures.

### STEP 4: CODE (Implementation)

**Principles:**

1. Write comments first (Explain intent).
2. Write code second.
3. Every block must be educational.

### STEP 5: TEST (QA)

**Tool:** TestSprite
**Goal:** 100% Test Success.
**Output Criteria:** All tests PASS, Coverage > 80%.

### STEP 6: AUDIT (Security)

**Tool:** CodeRabbit
**Goal:** Security and Quality Approval.
**Output Criteria:** No Critical findings, Review Approved.

### STEP 7: DONE (Completion)

**Checklist:** Task marked completed, Logic saved to Claude-Flow, Code committed.

---

## 🌳 TOOL SELECTION DECISION TREE

Use this tree to determine the correct tool for any situation:

```
                              ┌─────────────┐
                              │  THE TASK?  │
                              └──────┬──────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
       ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
       │ New Feature  │       │   Bug Fix    │       │ Refactoring  │
       └──────┬───────┘       └──────┬───────┘       └──────┬───────┘
              │                      │                      │
              ▼                      ▼                      ▼
       ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
       │ Task Master  │       │    Serena    │       │  Sequential  │
       │   (Plan)     │       │    (Find)    │       │   Thinking   │
       └──────┬───────┘       └──────┬───────┘       └──────┬───────┘
              │                      │                      │
              ▼                      ▼                      ▼
       ┌──────────────────────────────────────────────────────────┐
       │             IS A LIBRARY/API INVOLVED?                   │
       └───────────────────────────┬──────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │ YES          │ NO           │
                    ▼              │              │
             ┌──────────────┐      │              │
             │   Context7   │      │              │
             │  (Verify)    │      │              │
             └──────┬───────┘      │              │
                    │              │              │
                    ▼              ▼              │
       ┌──────────────────────────────────────────────────────────┐
       │           NEED TO UNDERSTAND EXISTING CODE?              │
       └───────────────────────────┬──────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │ YES          │ NO           │
                    ▼              │              │
             ┌──────────────┐      │              │
             │    Serena    │      │              │
             │  (Navigate)  │      │              │
             └──────┬───────┘      │              │
                    │              │              │
                    ▼              ▼              ▼
       ┌──────────────────────────────────────────────────────────┐
       │                      WRITE CODE                          │
       └───────────────────────────┬──────────────────────────────┘
                                   ▼
                             TestSprite (Test)
                                   ▼
                             CodeRabbit (Audit)
                                   ▼
                                DONE ✅

```

---

## 🔄 ERROR RECOVERY PROTOCOL

Systematic approach when things go wrong.

1. **Build Error:** Read error → Serena (find file) → Context7 (check syntax) → Fix.
2. **Runtime Error:** Analyze Stack Trace → Tavily (search error) → Serena (find logic) → Fix + Regression Test.
3. **Test Error:** Compare Expected vs Actual → Determine if Code or Test is wrong → Fix → Retest.
4. **API Error:** Check Status Code → Log Body → Context7 (Verify Spec) → Fix.

---

## 📏 CODE QUALITY STANDARDS

### Naming Conventions

| Type | Format | Example |
| --- | --- | --- |
| Variable | camelCase | `userName`, `isLoading` |
| Function | camelCase | `getUserById`, `validateEmail` |
| Class/Interface | PascalCase | `UserService`, `IUserRepository` |
| Constant | SCREAMING_SNAKE | `MAX_RETRY_COUNT` |
| Component | PascalCase | `ProfileCard.tsx` |

### Documentation Mandate (ELI15)

**"Explain Like I'm 15"** - Explain complex concepts simply.

```typescript
// ✅ GOOD: Detailed explanation
/**
 * CONCEPT: Debounce
 * PROBLEM: We don't want to call the API on every keystroke (too many requests).
 * SOLUTION: Wait for a pause (e.g., 300ms) after the last keystroke before calling.
 * ANALOGY: Like an elevator door. It waits for people to stop entering before closing.
 */
const debounce = (fn, delay) => { ... }

```

### Syntax Decoding

Always explain new or complex syntax (e.g., `??`, `?.`, `as const`) in comments immediately preceding usage.

---

## 🔐 SECURITY & PERFORMANCE CHECKLIST

Before every PR, verify:

* [ ] **Input Validation:** Are all inputs sanitized?
* [ ] **Auth:** Are sensitive endpoints protected?
* [ ] **Data Exposure:** Is sensitive data stripped from logs/responses?
* [ ] **SQL/XSS/CSRF:** Are standard protections active?
* [ ] **Dependencies:** Are packages up to date (`npm audit`)?
* [ ] **Bundle Size:** Is code splitting/tree shaking active?
* [ ] **Renders:** Are unnecessary re-renders prevented (memoization)?
* [ ] **Database:** Are N+1 queries avoided?

---

## ⚠️ FINAL WARNINGS

1. **Follow this file explicitly.** No steps are optional.
2. **No Hallucinations.** Context7 + Serena = Truth.
3. **No Approval without Tests.** 100% PASS = Proceed.
4. **Educate.** Every line of code is a lesson.
5. **Security First.** One vulnerability = Failure.