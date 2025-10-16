# LiteSearch: A Tiny Text Search Engine (with Autocorrect + Subword BPE)

**LiteSearch** is a compact, single-file search engine for large plain-text corpora (e.g., public-domain eBooks).  
It:

- Splits raw text into paragraph-level documents  
- Learns simple BPE-style subword merges to handle OOV terms  
- Builds an inverted index + TF-IDF vectors  
- Ranks results with cosine similarity  
- Autocorrects misspelled queries and suggests a “`[hint] Did you mean: …`”

The goal: be **small**, **readable**, and **surprisingly capable**.

---

## ✨ Features

- Paragraph indexing for dense, meaningful snippets  
- Greedy BPE tokenizer learned from the corpus (no external models)  
- Inverted index + TF-IDF (sparse)  
- Autocorrect via Levenshtein distance + frequency-aware candidate ranking  
- One-file pipeline: build artifacts and search using a single script  

---
## 🧩 Project Structure
.
├── search_engine.py # single script: build + search
├── data/ # (you provide) .txt files
└── artifacts/ # generated: paragraphs, merges, vocab, index, tfidf, norms, word_vocab

Artifacts created during build:
- `paragraphs.jsonl`, `books.jsonl`  
- `merges.txt` (BPE merges), `vocab.txt` (BPE tokens)  
- `word_vocab.txt` (word frequency list for autocorrect)  
- `inverted_index.json`  
- `tfidf.jsonl`, `tfidf_norms.json`  

---

## ⚙️ Requirements
- Python ≥ 3.8  
- Standard library only (no heavy external dependencies)

💡 *Tip:* You can add `tqdm` for progress bars if desired — not required.

---

## 🚀 Getting Started

### 1️⃣ Put data in place
Drop your `.txt` files into `./data/`:

data/
pg345.txt
pg1661.txt
...

### 2️⃣ Build the index
```bash
python3 search_engine.py --data_dir ./data --artifact_dir ./artifacts --build
```
This:
- Splits books into paragraphs
- Trains a small BPE (greedy merges)
- Builds an inverted index
- Computes TF-IDF vectors
- Writes a word-level vocab for autocorrect

### 3️⃣ Run a single query
```bash
python3 search_engine.py --data_dir ./data --artifact_dir ./artifacts --query "your query"
```

### 4️⃣ Interactive search (REPL)
```bash
python3 search_engine.py --data_dir ./data --artifact_dir ./artifacts --search
```
Example Queries (from my runs)

| Type                              | Query                          | Behavior                                                              |
| --------------------------------- | ------------------------------ | --------------------------------------------------------------------- |
| Multi-token misspelling           | `alce picturs`                 | Suggests `alice pictures`; retrieves *Alice in Wonderland* paragraphs |
| Classic name misspelling          | `sherlock holms`               | Suggests `sherlock holmes`; retrieves Sherlock passages               |
| Single-token phonetic misspelling | `drakula`                      | Suggests `dracula`; returns Dracula passages                          |
| Baseline correct query            | `dracula harker`               | Direct hits linking Count Dracula & Jonathan Harker                   |
| Multi-term character query        | `dracula jonathan harker mina` | Retrieves diary and chapter sections featuring both                   |

(Exact ranking may vary slightly with your corpus.)

## How It Works (Brief)
**BPE Training:** Learns frequent symbol merges on sampled text, tokenizing queries + paragraphs consistently.

**Indexing:** Builds inverted index (token → {paragraph: term frequency}) and TF-IDF vectors per paragraph.

**Autocorrect:**
- Generates 1-edit candidates (insert, delete, substitute).
- Prefers known vocab words by frequency.
- Falls back to bounded Levenshtein search.
- Prints [hint] Did you mean: … when correction differs.

**Ranking:** Cosine similarity between query TF-IDF and paragraph vectors.

## 🧪 Reproducing Results

Use a corpus of public-domain .txt files (e.g., Project Gutenberg).
Run the build command above.
Try the example queries — you’ll get similar behavior.

## Roadmap (Ideas)
- Positional scoring / BM25
- Character n-grams for robust OOV matching
- Synonym/alias lists for names (e.g., “Holms” → “Holmes”)
- Web UI / FastAPI service

## Attribution
Texts used should be public domain (e.g., Project Gutenberg). Please follow their usage guidelines.

## Contact
Questions or ideas?
→ Open an issue
 or connect with me here on GitHub!
 
---

```Note: Outputs will vary slightly depending on ranking, but the examples above illustrate the system’s capabilities.```
