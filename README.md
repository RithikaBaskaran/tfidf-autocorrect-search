# Homework 1 – Search Engine

## Overview
This project implements a simple text-based search engine over a collection of eBooks.  
The system supports:
- Parsing and indexing text into paragraphs
- Querying the collection with ranking based on similarity scores
- Handling misspelled queries using autocorrect with suggestions

The deliverables for this homework include:
1. `search_engine.py` – the main Python script  
2. `README.md` – this instruction file  
3. `Homework1_Writeup.pdf` – analysis and discussion of results  

---

## Requirements
- Python 3.8 or higher
- Standard Python libraries:
  - `argparse`
  - `json`
  - `collections`
  - `os`
  - `re`
  - `math`
- Additional libraries:
  - `tqdm`
  - `textblob` (for autocorrect spelling suggestions)

You can install missing packages with:
```bash
pip install tqdm textblob
```

## Data Setup
1. Place your raw text eBooks inside a data/ directory.
2.	The script will process these into artifacts (e.g., paragraphs.jsonl) for efficient searching.
3.	Artifacts are stored in an artifacts/ folder (or artifacts_fullword/ if you rebuild with full-word indexing).

## How to Run

## Index the Data
If you are running for the first time (or after changing the dataset), build the artifacts:

```bash 
python3 search_engine.py --data_dir ./data --artifact_dir ./artifacts --build
```
This creates the processed files needed for searching.

## Run a Query
To search the collection:

```bash 
python3 search_engine.py --data_dir ./data --artifact_dir ./artifacts --query "your query here"
```

Example:
```bash 
python3 search_engine.py --data_dir ./data --artifact_dir ./artifacts --query "sherlock holms"
```

If the query contains spelling mistakes, the script will suggest corrections:
[hint] Did you mean: sherlock holmes

## Example Queries and Outputs

The following queries illustrate the system’s behavior:
1.	Multi-token misspelling
	•   Query: `alce picturs`
	•	System suggests alice pictures and retrieves paragraphs from Alice in Wonderland with picture references.
2.	Classic name misspelling
	•	Query: `sherlock holms`
	•	System suggests sherlock holmes and retrieves Sherlock Holmes passages.
3.	Single-token phonetic misspelling
	•	Query: `drakula`
	•	System suggests dracula and retrieves strong Dracula-related passages.
4.	Baseline correct query
	•	Query: `dracula harker`
	•	No correction needed. Retrieves passages directly connecting Count Dracula and Jonathan Harker.
5.	Multi-term, character-focused query
	•	Query: `dracula jonathan harker mina`
	•	No correction needed. Retrieves passages listing Jonathan Harker and Mina together, often in chapter headers or diaries.


```Note: Outputs will vary slightly depending on ranking, but the examples above illustrate the system’s capabilities.```