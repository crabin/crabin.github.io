---
layout: single
author_profile: true
title: "AI Debate — Multi-Agent Debate System"
collection: projects
category: ai
permalink: /projects/2026-03-22-AI-Debate-System
excerpt: 'Multi-agent debate simulation system with 9 independent AI agents following standard debate competition rules.'
date: 2026-03-22
---

# Multi-Agent AI Debate System

## Overview

This project implements a complete debate competition simulation system using 9 independent AI agents (8 debaters + 1 judge). The system follows standard four-stage debate rules — opening statements, cross-examination, free debate, and closing arguments — with real-time streaming output via a colorful terminal UI. Each agent operates with dedicated system prompts and can use different LLM providers (Zhipu GLM, DeepSeek, OpenAI-compatible APIs) simultaneously.

## Agent Architecture

### Hybrid Pattern (Lightweight Orchestrator + Message Pool)

The system uses a hybrid agent architecture combining:

- **StageController**: Lightweight orchestrator managing stage flow and speaking order
- **MessagePool**: Dual-layer message pool handling message routing and permission isolation
- **Independent Agents**: Each holds its own prompt + context from visible channels + generates speeches

### Message Channels

| Channel   | Readable By       | Writable By          |
|-----------|-------------------|----------------------|
| `public`  | All agents        | All agents           |
| `team_pro`    | Pro team (4)      | Pro team             |
| `team_con`    | Con team (4)      | Con team             |
| `judge_notes` | Judge             | Judge                |

### Agent Visibility

| Agent Identity   | Visible Messages              |
|------------------|-------------------------------|
| Pro debaters     | `public` + `team_pro`         |
| Con debaters     | `public` + `team_con`         |
| Judge            | `public` + `judge_notes`      |

## Agent Design

### BaseAgent

```python
class BaseAgent:
    name: str              # "Pro 1st Debater"
    agent_id: str          # "pro_1"
    team: str              # "pro" / "con" / "judge"
    system_prompt: str     # Core prompt
    llm: BaseLLM           # Configurable LLM instance

    def build_context(self, message_pool, stage) -> str:
        """Build context from message pool by visibility"""

    def speak(self, system_prompt, context, instruction) -> str:
        """Generate speech using LLM"""
```

### DebaterAgent

All debaters share the `DebaterAgent` class, differentiated by configuration. Prompt structure:

```
[Identity Layer]  You are {team} {position}, stance: {stance}
[Personality Layer]  Debate style: {personality} (configurable)
[Role Layer]  Injected by position (1-4)
[Rules Layer]  Debate competition rules
[Context]  Current stage + visible historical speeches
```

**Position Responsibilities**:

| Position | Core Responsibilities |
|----------|---------------------|
| 1st Debater | Opening statement (750 words/3 min) + Cross-exam summary (500 words/2 min) |
| 2nd Debater | Cross-exam questions (125 words/30 sec) + Answers (250 words/1 min) + Free debate |
| 3rd Debater | Cross-exam questions (125 words/30 sec) + Answers (250 words/1 min) + Free debate |
| 4th Debater | Free debate + Closing statement (750 words/3 min) |

### JudgeAgent

Independent design, not inheriting from Debater's role layer.

**Scoring Dimensions** (1-10 points):
- Logic — Weight 0.25
- Persuasion — Weight 0.25
- Expression — Weight 0.20
- Teamwork — Weight 0.15
- Rule Compliance — Weight 0.15

**Violation Detection**:
- `counter_question`: Defender asks questions back
- `not_direct_answer`: Fails to answer directly
- `attacker_answered`: Attacker answers questions
- `off_topic`: Seriously deviates from topic
- `scripted_summary`: Cross-exam summary is scripted
- `personal_attack`: Personal attacks

## Debate Stages

### Stage 1: Opening Statements

Fixed order: Pro 1st (3 min/750 words) → Con 1st (3 min/750 words)

### Stage 2: Cross-Examination

4 rounds of cross-examination:
1. Pro 2nd asks questions → Select Con 2nd or 3rd to answer
2. Con 2nd asks questions → Select Pro 2nd or 3rd to answer
3. Pro 3rd asks questions → Select Con 2nd or 3rd to answer
4. Con 3rd asks questions → Select Pro 2nd or 3rd to answer

Each round: Questions 30 sec (125 words, 3+ questions) + Answers 1 min (250 words)

After cross-exam: Pro 1st summary (2 min/500 words) → Con 1st summary (2 min/500 words)

### Stage 3: Free Debate

- Pro speaks first, then teams alternate
- 4 minutes total per team (estimated by word count)
- Each debater must speak at least once
- Same team cannot speak consecutively

**Team Coordination Mechanism**:

When it's a team's turn to speak, StageController uses a single LLM call to simulate "captain" decision:

```
Captain Prompt:
  [Task] Decide who speaks this round, give response direction suggestion.
  [Team messages] {team_messages}
  [Public debate records] {recent_public_messages}
  [Speak counts] {speak_counts}
  [Unspoken debaters] {unspeaking_debaters}
  [Time remaining] {time_left} sec

  Output format (strict JSON):
  {"speaker": "pro_2", "direction": "Refute opponent's employment argument"}
```

Flow: Captain LLM call → Parse JSON for speaker + direction → Inject direction into selected debater's instruction → Selected debater calls speak() → Generate speech.

Prioritize unspoken debaters to ensure each speaks at least once.

### Stage 4: Closing Statements

Fixed order: Con 4th (3 min/750 words) → Pro 4th (3 min/750 words)

Judge scores each closing statement. After all stages, Judge provides final review.

## Scoring System

### Three-Layer Scoring

1. **Real-time Scoring** (after each speech) — Judge agent scores
2. **Rule Engine Auto-Penalty** (hardcoded):
   - Overtime: Team -3, Individual -2
   - Counter-question in cross-exam: Individual -2
   - Attacker answers in cross-exam: Individual -2
   - Same team consecutive speeches: Team -3
3. **Stage Summary + Final Summary**

### Score Formula

```
Single Speech Score = logic * 0.25 + persuasion * 0.25 + expression * 0.20
                     + teamwork * 0.15 + rule_compliance * 0.15

Individual Total = sum(all speech scores) + individual penalties (negative)
Team Total = sum(4 individuals' totals) + team penalties (negative)

No lower limit (can be negative, reflecting severe violation penalties)
Best Debater = Individual with highest total
```

## LLM Abstraction Layer

```python
class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], temperature: float) -> str:
        """Unified chat interface"""

class ZhipuLLM(BaseLLM):
    """Zhipu AI implementation, default glm-4.7"""

class OpenAICompatibleLLM(BaseLLM):
    """OpenAI-compatible API implementation (DeepSeek, Claude, etc.)"""

LLM_FACTORY = {"zhipu": ZhipuLLM, "openai_compatible": OpenAICompatibleLLM, ...}
```

## Configuration

### Multi-Model Configuration

Each role (pro, con, judge) can use different LLM providers via `.env`:

```bash
# Global default
LLM_PROVIDER=zhipu
LLM_MODEL=glm-4.7
ZAI_API_KEY=your-key

# Pro uses DeepSeek
PRO_LLM_PROVIDER=openai_compatible
PRO_LLM_MODEL=deepseek-chat
PRO_LLM_BASE_URL=https://api.deepseek.com/v1
PRO_LLM_API_KEY=sk-...

# Con uses Zhipu
CON_LLM_PROVIDER=zhipu
CON_LLM_MODEL=glm-4.7

# Judge uses Claude
JUDGE_LLM_PROVIDER=openai_compatible
JUDGE_LLM_MODEL=claude-sonnet-4-6
JUDGE_LLM_BASE_URL=https://api.anthropic.com/v1
JUDGE_LLM_API_KEY=sk-ant-...
```

Priority: Role-prefixed env vars > Global env vars > Config defaults

### Debater Personalities

5 debate styles defined in `config/personalities.yaml`:

| Style      | Name             | Characteristics                          |
|------------|------------------|------------------------------------------|
| `logical`      | Logical | Syllogism, causal chains, rigorous argumentation |
| `emotional`    | Emotional | Vivid examples, emotional resonance      |
| `data_driven`  | Data-Driven | Research reports, statistical data       |
| `aggressive`   | Aggressive | Find loopholes, relentless pursuit       |
| `diplomatic`   | Diplomatic | Defuse attacks, clever transformation    |

## Project Structure

```
AI_debate/
├── config/
│   ├── default.yaml          # LLM, timer, scoring config
│   ├── topics.yaml           # Built-in topics
│   └── personalities.yaml    # Debater personality templates
├── src/
│   ├── cli.py                # CLI entry
│   ├── config.py             # YAML config loader
│   ├── export.py             # JSON export
│   ├── agents/               # Agents
│   │   ├── base.py           # Base agent
│   │   ├── debater.py        # Debater agent
│   │   ├── judge.py          # Judge agent
│   │   └── prompts.py        # System prompt templates
│   ├── engine/               # Engine
│   │   ├── message_pool.py   # Message pool (public/team/judge channels)
│   │   ├── scorer.py         # Three-layer scoring engine
│   │   └── timer.py          # Timer and overtime tracking
│   ├── llm/                  # LLM abstraction layer
│   │   ├── __init__.py       # LLM factory + role env var parser
│   │   ├── base.py           # Abstract base class
│   │   ├── zhipu.py          # Zhipu GLM provider
│   │   └── openai_compatible.py  # OpenAI-compatible provider
│   ├── stages/               # Debate stages
│   │   ├── controller.py     # Stage controller
│   │   ├── opening.py        # Opening statements
│   │   ├── cross_exam.py     # Cross-examination
│   │   ├── free_debate.py    # Free debate (sequential + concurrent)
│   │   └── closing.py        # Closing statements
│   └── display/
│       └── terminal.py       # Rich terminal UI
├── tests/                    # Test suite
├── docs/                     # Design docs
├── .env.example              # Environment variable template
└── pyproject.toml
```

## Technical Stack

| Package         | Usage                              |
|-----------------|-----------------------------------|
| `zai`           | Zhipu GLM SDK                      |
| `openai`        | OpenAI-compatible API support      |
| `rich >= 13.0`  | Terminal UI, panels, tables, live streaming |
| `pyyaml >= 6.0` | YAML config loading                |
| `python-dotenv >= 1.0` | .env environment variable loading |
| `dataclasses`   | Immutable data structures          |

## Showcase

Running the debate system:

```bash
python -m src.cli
```

![AI Debate System Terminal Output](AI_debate/1774363272915.png)

The terminal displays:
- Color-coded speeches (Blue for Pro, Red for Con, Yellow for Judge)
- Real-time word count and timing
- Live scoring after each speech
- Scoreboard panel
- Overtime/violation warnings
- Final scoreboard with verdict

## Applications

- AI education and research — Multi-agent collaboration study
- Debate training — Simulated practice for debaters
- LLM evaluation — Compare different models' reasoning capabilities
- Prompt engineering — System prompt design patterns
- Competitive intelligence — Simulating opposing viewpoints

## Source Code

[GitHub Repository](https://github.com/crabin/AI_debate)

The project has been open-sourced and pushed to GitHub for community collaboration.
