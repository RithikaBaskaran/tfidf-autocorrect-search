# LiteSearch: A Tiny Text Search Engine (with Autocorrect + Subword BPE)

## Overview
LiteSearch is a compact, single-file search engine for large plain-text corpora (e.g., public-domain eBooks). It: 

Splits raw text into paragraph-level documents
Learns simple BPE-style subword merges to handle OOV terms
Builds an inverted index + TF-IDF vectors
Ranks results with cosine similarity
Autocorrects misspelled queries and suggests a “[hint] Did you mean: …”

The goal: be small, readable, and surprisingly capable.

---

## Features
Paragraph indexing for dense, meaningful snippets

Greedy BPE tokenizer learned from the corpus (no external models)

Inverted index + TF-IDF (sparse)

Autocorrect via Levenshtein distance + frequency-aware candidate ranking

One-file pipeline: build artifacts and search using a single script

## Project Structure
.
├── search_engine.py        # single script: build + search
├── data/                   # (you provide) .txt files
└── artifacts/              # generated: paragraphs, merges, vocab, index, tfidf, norms, word_vocab
Artifacts created during build:

paragraphs.jsonl, books.jsonl

merges.txt (BPE merges), vocab.txt (BPE tokens)

word_vocab.txt (word frequency list for autocorrect)

inverted_index.json

tfidf.jsonl, tfidf_norms.json

## Requirements
Python ≥ 3.8

Standard library only (no hard external deps)

Tip: If you want progress bars or timing, you can layer in tqdm later—this repo doesn’t require it.

## Getting Started
1) Put data in place

Drop your .txt files into ./data/. Example:

data/
  pg345.txt
  pg1661.txt
  ...

2) Build the index
python3 search_engine.py --data_dir ./data --artifact_dir ./artifacts --build


This:

Splits books into paragraphs

Trains a small BPE (greedy merges)

Builds an inverted index

Computes TF-IDF vectors

Writes a word-level vocab for autocorrect

3) Run a single query
python3 search_engine.py --data_dir ./data --artifact_dir ./artifacts --query "your query"

4) Interactive search (REPL)
python3 search_engine.py --data_dir ./data --artifact_dir ./artifacts --search

Example Queries (from my runs)

These illustrate behavior on a public-domain book corpus:

Multi-token misspelling
Query: alce picturs
Result: [hint] Did you mean: alice pictures and paragraphs referencing Alice + pictures.

Classic name misspelling
Query: sherlock holms
Result: [hint] Did you mean: sherlock holmes and Sherlock-rich paragraphs.

Single-token phonetic misspelling
Query: drakula
Result: [hint] Did you mean: dracula with strong Dracula passages.

Baseline correct query
Query: dracula harker
Result: Direct hits connecting Count Dracula and Jonathan Harker.

Multi-term, character-focused query
Query: dracula jonathan harker mina
Result: Chapters and diary headers listing Jonathan Harker and Mina together.

(Exact ranking may vary slightly with your corpus.)

## How It Works (Brief)
BPE training: learns frequent symbol merges on a sampled subset of the corpus; then tokenizes query + paragraphs consistently.

Indexing: builds an inverted index (token → {paragraph: term freq}) and TF-IDF vectors per paragraph, plus L2 norms.

Autocorrect: for each alphanumeric query token:

Generate 1-edit candidates (insertions, substitutions, deletions), prefer those in the word vocab by frequency rank.

If none fit, fall back to bounded Levenshtein against length-banded vocab entries.

If corrected query ≠ original, print a [hint] line.

Ranking: cosine similarity between query TF-IDF and each candidate paragraph vector.

## Reproducing My Results
1) Use a corpus of public-domain .txt files (e.g., Project Gutenberg).

2) Run the build command above.

3) Try the example queries; you should see similar behavior (not identical text).

## Roadmap (Ideas)
Positional scoring / BM25

Character n-grams for robust OOV matching

Synonym/alias lists for names (e.g., “Holms” → “Holmes”)

Web UI / FastAPI service

## Attribution
Texts used should be public domain (e.g., Project Gutenberg). Please follow their usage guidelines.

## Contact
Questions or ideas? Open an issue or ping me here.

```Note: Outputs will vary slightly depending on ranking, but the examples above illustrate the system’s capabilities.```
