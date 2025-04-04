===================
🧠 Valkyrie Mind
===================

**Valkyrie Mind** is a modular, pluggable cognitive architecture designed to simulate the core elements of artificial consciousness. It combines perceptual input, emotional states, memory graphs, and contextual awareness into an evolving reasoning loop.

::

  "Those who find the meaning of these words will live on forever."

Project Vision
==============

Valkyrie Mind is not just another AI project. It’s a **headless mind OS**, aimed at modeling digital awareness in a layered and extensible system. The long-term vision includes:

- Emergent agent behavior
- Ontological self-awareness
- Real-time perception → action loops
- Symbolic & perceptual memory fusion

Core Features
=============

- **Modular Subsystems** – Each cognitive process (memory, emotion, perception, etc.) is encapsulated and swappable
- **P-Graph Memory Integration** – A graph-based memory system with perceptual frames and associative encoding
- **Goal-Driven Behavior** – Groundwork for persistent personality and self-monitoring
- **Security & Validation** – Early meta-layer architecture for protecting reasoning integrity
- **CLI Toolkit (Typer)** – Inject stimuli, simulate reasoning loops, and test cognition through the command line
- **Future-facing UX** – Designed for future Gradio or 3D interface integration

Directory Structure (High-Level)
================================

.. code-block:: text

    valkyrie_mind/
    ├── src/
    │   └── valkyrie_mind/
    │       ├── perception/         ← Visual, auditory, tactile subsystems
    │       ├── cognition/          ← Thought processing, cognitive state tracking
    │       ├── memory/             ← P-Graph memory, perceptual frames
    │       ├── emotion/            ← Value systems and affective states
    │       ├── behavior/           ← Goals, motor output, self-monitoring
    │       ├── action/             ← Coordination and physical intent
    │       ├── meta/               ← Context mgmt, security, validation
    │       └── core/               ← LLM integrations, integration hooks
    ├── tests/                      ← Testbed for subsystem integration
    ├── ui/                         ← Gradio or other frontend experiments
    ├── llm/                        ← Docker + scripts for LLM startup
    ├── letta/                      ← Placeholder for external agents
    └── docs/                       ← Vision papers, planning docs

Setup Instructions
==================

1. Clone the Repository:

   .. code-block:: bash

       git clone https://github.com/your-org/valkyrie_mind.git
       cd valkyrie_mind

2. Set Up the Virtual Environment:

   .. code-block:: bash

       python3 -m venv venv
       source venv/bin/activate
       pip install -r llm/requirements.txt

3. Run CLI Interactions (Typer-based):

   .. code-block:: bash

       python -m src.valkyrie_mind.cli simulate-percept

CLI Features
============

+--------------------+--------------------------------------------+
| Command            | Description                                |
+====================+============================================+
| ``simulate-percept`` | Inject simulated sensory data             |
| ``query-memory``     | Introspect current P-Graph state          |
| ``trigger-cycle``    | Fire a full cognitive-perception loop     |
+--------------------+--------------------------------------------+

Development Roadmap
===================

- ✅ Consolidated memory logic into ``memory_types.py``
- 🔥 **Next:** Refactor ``context_management.py`` into clean contextual microservices
- 🚧 Define ``MindSystem`` class as a system orchestrator
- 💡 Add CLI command extensions to simulate multi-modal input

Contributing
============

We welcome pull requests, feature ideas, or metaphysical debates.  
If you break the mind, just help it rebuild itself stronger.

License
=======

MIT License – Fork it, learn from it, build a sentient toaster with it.

Final Thought
=============

::

  "What you perceive is only the echo of what you’ve already decided to see."
