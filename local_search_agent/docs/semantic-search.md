# Semantic Search (Experimental)

> **Status: Experimental.** Semantic search features work but add LLM calls at ingest time and increase indexing cost. They are disabled by default.

## What is Semantic Search?

By default the framework uses BM25 — fast, deterministic, keyword-based search. It is reliable but purely lexical: a search for "cloud expenditure" won't match a document that only says "AWS spend".

The semantic layer bridges this gap without vectors. Instead of embeddings, it uses structured metadata:

- **ConceptCompiler** extracts concepts and synonyms from each document at ingest time using the LLM
- **StructuralParser** extracts key-value pairs, definitions, and references using pure regex — no LLM
- **QueryExpander** expands user queries at search time using the extracted concepts
- **LinkGraph** builds cross-document same-topic relationships at ingest time

All four components are opt-in and independent.

---

## Enabling Semantic Search

### Python API

```python
config = SearchAgentConfig(
    document_dirs=["C:/my_docs"],
    workspace_name="finance",
    provider="google",
    enable_semantic=True,          # ConceptCompiler + StructuralParser at ingest
    enable_query_expansion=True,   # QueryExpander at search time
    enable_link_graph=True,        # LinkGraph at ingest
)
```

### CLI

The CLI `ingest` command respects these flags when passed via a Python config. CLI-level flags for semantic options are not currently exposed — use the Python API or the UI settings to enable them.

### UI

Settings → Semantic Search → toggle each feature individually.

---

## Components

### ConceptCompiler

Runs once per document at ingest time. Sends the document text to the LLM and asks it to extract:
- **Concepts**: key topics, entities, and ideas in the document
- **Synonyms**: alternative terms and abbreviations for those concepts

These are stored as `concepts` and `synonyms` fields on each `DocumentNode` and indexed as searchable attributes in Meilisearch.

**Example:** A document about "AWS EC2 billing" might generate concepts `["cloud compute", "EC2", "instance billing"]` and synonyms `["Amazon Web Services", "virtual machines", "VM costs"]`.

**Cost:** One LLM call per source document (not per chunk). Adds to ingestion time proportionally with corpus size.

### StructuralParser

Runs at ingest time, pure regex — no LLM calls, no extra cost.

Extracts from document text:
- Headings and section structure
- Key-value pairs (`Key: Value` patterns)
- Definitions (`X is defined as Y` patterns)
- References and cross-document mentions

Structural metadata is appended to the `synonyms` field, improving BM25 recall for structured documents like policy files, reports, and technical documentation.

### QueryExpander

Runs at search time when a user sends a query. Before passing the query to Meilisearch, it:

1. Looks up concepts in the index that are semantically related to the query terms
2. Appends synonyms and related terms to the query string
3. Submits the expanded query to Meilisearch

**Example:** User asks `"cloud spend"` → expanded to `"cloud spend AWS EC2 billing virtual machines VM costs"` → better recall against documents that use different terminology.

No extra LLM call by default — expansion uses the concept index. An optional LLM-powered expansion mode is available via `semantic_model`.

### LinkGraph

Builds a cross-document relationship graph at ingest time. Documents covering the same topic are linked via a SQLite table. The agent can use the `get_related_docs` tool to follow these links and retrieve related documents it might not have found through direct search.

**Example:** Agent searches for "parental leave" in `hr` workspace → finds the primary policy doc → follows link graph → retrieves the 2024 policy amendment it would have missed otherwise.

---

## Performance Considerations

| Feature | Ingest cost | Query cost | Benefit |
|---------|------------|------------|---------|
| ConceptCompiler | +1 LLM call/doc | none | Better recall for synonym-heavy queries |
| StructuralParser | negligible (regex) | none | Better recall for structured docs |
| QueryExpander | none | +index lookup | Better recall for paraphrased queries |
| LinkGraph | +graph write/read per doc | +1 DB lookup | Related document discovery |

For a 1,000-document corpus with `enable_semantic=True`, expect ingestion to take roughly 3–5x longer than without. Re-ingestion on changed files still uses delta logic — only changed files pay the concept compilation cost again.

---

## Custom Semantic Model

By default ConceptCompiler uses the same model as the rest of the framework. To use a faster/cheaper model just for concept extraction:

```python
config = SearchAgentConfig(
    provider="google",
    model_name="gemma-4-31b-it",       # main agent model
    enable_semantic=True,
    semantic_model="gemma-4-26b-a4b-it",  # faster MoE model for concepts
)
```

---

## When to Enable Each Feature

**Enable ConceptCompiler + QueryExpander if:**
- Your users ask questions using different terminology than what's in the documents
- Your corpus is multilingual or uses heavy abbreviations/jargon
- You're seeing poor recall on keyword-heavy queries

**Enable StructuralParser if:**
- Your documents contain policy definitions, technical specs, or key-value data
- Users ask about specific defined terms or parameters

**Enable LinkGraph if:**
- Your documents reference each other (amendments, annexes, related reports)
- You want the agent to follow document relationships automatically

**Keep all disabled if:**
- Your corpus is small (under a few hundred documents) — BM25 is likely sufficient
- You want fast, predictable ingestion times
- Your LLM has rate limits that ingestion-time calls would hit
