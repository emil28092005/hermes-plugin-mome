# MoME — Mixture of Memory Experts for Hermes Agent

Sparse-gated personal memory with an online-learning router. Replaces monolithic memory context with expert-routed retrieval — only relevant memory experts are activated per query.

## Features

- **4 Experts**: `identity`, `knowledge`, `projects`, `preferences`
- **Sparse Activation**: Only 1–2 experts are queried per turn (determined by the router)
- **Online Learning**: SGD classifier router learns which expert to route to based on usage patterns
- **Local-Only**: 100% offline, no API keys needed
- **Auto-Learning**: Regex-based fact extraction from user queries
- **Persistent**: JSON-based storage, survives restarts

## Installation

```bash
# 1. Clone the plugin into Hermes plugins directory
git clone https://github.com/emil28092005/hermes-plugin-mome.git \
    ~/.hermes/hermes-agent/plugins/memory/mome

# 2. Install dependencies
pip install numpy scikit-learn

# 3. Activate via Hermes memory setup
hermes memory setup
```

Select `mome` from the list of available memory providers.

## Usage

Once activated, MoME provides three tools:

### `mome_search`
Search memory across experts. The router automatically selects which experts to query.

```text
mome_search(query="what projects am I working on?", top_k=3)
mome_search(query="Python skills", expert="knowledge", top_k=5)
```

### `mome_store`
Store a fact directly into a specific expert.

```text
mome_store(expert="identity", fact="I prefer dark mode")
```

### `mome_status`
Show expert sizes and router training state.

```text
mome_status()
```

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│   Router    │ ← SGDClassifier (online learning)
│  (predict)  │
└──────┬──────┘
       │ top-2 experts selected
       ▼
┌──────┴──────┐
│   Experts   │
│  identity   │ ─── cosine similarity search
│  knowledge  │
│  projects   │
│ preferences │
└──────┬──────┘
       │ relevant memories
       ▼
┌─────────────┐
│   Context   │ → injected into system prompt
└─────────────┘
```

### Fact Extraction (Auto-Learning)

On each turn, MoME automatically extracts facts from user queries via regex:

- `меня зовут X` → identity
- `я живу в X` → identity
- `работаю над X` → projects
- `у меня проект X` → projects
- `знаю/использую X` → knowledge
- `нравится X` → preferences

The router learns over time which experts to activate for which types of queries.

## Dependencies

- Python ≥ 3.10
- `numpy` — vector operations
- `scikit-learn` — SGD router classifier

Zero external API dependencies. Works fully offline.

## Development

```bash
# Test the engine standalone
python -c "from plugins.memory.mome.engine import MomeEngine; e = MomeEngine('/tmp/test_mome'); e.store_fact('identity', 'Test fact'); print(e.query('test'))"
```

## Author

**Emil Shanaty** — [github.com/emil28092005](https://github.com/emil28092005)

## License

MIT
