<h1 align="center">Gecko: A Simulation Environment with Stateful Feedback for Refining Agent Tool Calls</h1>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-EB3ECC)](https://camel-ai.github.io/camel/index.html)
[![Discord](https://img.shields.io/discord/1082486657678311454?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb)](https://discord.camel-ai.org/)
[![X](https://img.shields.io/twitter/follow/CamelAIOrg?style=social)](https://x.com/CamelAIOrg)
[![CAMEL Framework](https://img.shields.io/badge/CAMEL-Framework-black)](https://github.com/camel-ai/camel)
[![CAMEL-AI](https://img.shields.io/badge/CAMEL--AI-Website-blue)](https://www.camel-ai.org/)

</div>

<hr>

<div align="center">
<h4 align="center">

[Project Page](https://camel-ai.github.io/gecko/) |
Paper (TBD) |
[Installation](#️-installation) |
[Citation](#️-cite) |
[CAMEL-AI](https://www.camel-ai.org/)

</h4>
</div>

![overview](overview.png)

## Abstract
The ability to use tools is fundamental for large language model (LLM) agents. Given a task, existing systems use LLMs to plan and generate tool calls, which are executed by real-world tools to complete the task. However, tool calls are prone to errors because they are derived merely from LLM intrinsic capabilities. What is more, while it is useful to let LLMs iteratively refine the tool-call sequence using execution results from real tools, this process can be expensive and lead to unsafe results.

To improve LLM tool calls and address issues caused by using real tools for refinement, we introduce Gecko, a comprehensive environment that simulates tool responses using a combination of rules and LLMs. Specifically, Gecko checks the validity of tool calls including input arguments and tool names, synthesizes reasonable responses that adhere to the output schema, and assesses whether all task objectives have been achieved. These three types of feedback provided by Gecko allow LLMs to refine their tool calls, forming a simple yet effective test-time scaling method named GATS. On BFCLv3 and $\tau^2$-bench, GATS consistently improves the tool-calling performance of various LLMs.

## 🛠️ Installation

### 1. Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure env
```bash
cp .env.example .env
# then set OPENAI_API_KEY in .env
```

`run_gecko_server.py` loads `.env` automatically from the repository root.

## 🚀 Quick Start

### 1. Run Gecko
```bash
python run_gecko_server.py --schemas_dir data/openapi --port 8000
```

### 2. Smoke test
```bash
curl -s http://localhost:8000/session-id
```

## Core Endpoints
- `GET /session-id`: create a new session
- `POST /set-session-state`: set initial or current state
- `GET /get-session-state`: fetch latest state
- `GET /get-session-history`: fetch request/response history
- `POST /update-state-from-real`: sync real tool results into state

## Route Format
Both route styles are supported:
- `POST /api_name/endpoint`
- `POST /endpoint`

## Notes
- Default schemas path is `data/openapi`.
- Session state is persisted in local `sessions.db`.

## 📚 Exploring CAMEL Dependency
Gecko is built on top of the [CAMEL](https://github.com/camel-ai/camel) framework. You can inspect CAMEL source code directly to understand the base abstractions used by Gecko.

### Accessing CAMEL Source Code
```bash
git clone https://github.com/camel-ai/camel.git
cd camel
```

## ⏱️ Future Plans
- [ ] Release GATS code (planned for March 2026).

## 🖊️ Cite
If you find this repo useful, please cite:

```bibtex
@misc{zhang2026gecko,
  title={Gecko: A Simulation Environment with Stateful Feedback for Refining Agent Tool Calls},
  author={Zeyu Zhang and Guohao Li and Zhenchang Xing and Alexandros Apostolopoulos and Yu Lin Lee and Liang Zheng},
  year={2026},
  eprint={TBD},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={TBD}
}
```
