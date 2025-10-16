from pathlib import Path
import argparse
import json
import math
import re
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable, Set, Any, Optional

# ---------- Optional polish knobs ----------
WHOLE_WORD_MIN_MATCHES = 1     # set to 0 to disable filtering
WHOLE_WORD_MATCH_BONUS  = 0.06 # additive bonus per whole-word hit
# ------------------------------------------

# ----------------------------------------------------
# Levenshtein edit distance + word-level autocorrect
# ----------------------------------------------------

def levenshtein_distance(a: str, b: str, max_dist: Optional[int] = None) -> int:
    """
    Standard Wagner–Fischer Levenshtein distance.
    If max_dist is given and exceeded, returns max_dist+1 (early exit).
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)

    # Quick bound: if even the length gap exceeds threshold, bail.
    if max_dist is not None and abs(la - lb) > max_dist:
        return max_dist + 1

    # Ensure a is the longer for a tiny speed bump
    if la < lb:
        a, b = b, a
        la, lb = lb, la

    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        row_min = cur[0]  # for early exit
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost  # substitution
            )
            if cur[j] < row_min:
                row_min = cur[j]
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[lb]


def load_word_vocab(artifacts: "Artifacts", top_n: Optional[int] = None) -> List[str]:
    """Load word-level vocab; optionally truncate to top_n. Cache list, set, and rank map."""
    if not artifacts.word_vocab.exists():
        load_word_vocab._list = []
        load_word_vocab._set = set()
        load_word_vocab._rank = {}
        return []
    words = artifacts.word_vocab.read_text(encoding="utf-8").splitlines()
    if top_n is not None:
        words = words[:top_n]
    # caches
    load_word_vocab._list = words
    load_word_vocab._set  = set(words)
    load_word_vocab._rank = {w: i for i, w in enumerate(words)}  # lower i = more frequent
    return words

# initialize caches
load_word_vocab._list = []
load_word_vocab._set  = set()
load_word_vocab._rank = {}


def autocorrect_token(word: str, word_vocab: List[str], max_dist: int = 2) -> str:
    """
    Edit-type priority autocorrect:
      1) Generate all 1-edit neighbors grouped by operation: insertions > substitutions > deletions.
         Within the first non-empty group, pick the most frequent vocab member.
      2) If none found, do a bounded Levenshtein sweep (<= max_dist) over a length-banded shortlist.
      3) If the original is in-vocab but rare, allow override only for a clearly more frequent candidate.
    """
    w = word.lower()
    if len(w) < 3 or not w.isalpha():
        return word

    vocab_list = getattr(load_word_vocab, "_list", word_vocab) or word_vocab
    vocab_set  = getattr(load_word_vocab, "_set", set(vocab_list))
    rank_map   = getattr(load_word_vocab, "_rank", {t: i for i, t in enumerate(vocab_list)})

    # Frequency guardrails
    RANK_TRUST    = 20_000   # common words: don't “fix”
    CLEAR_WIN     = 5_000    # replacement must beat rare original by at least this much in rank

    in_vocab = w in vocab_set
    w_rank = rank_map.get(w, 10**9)
    if in_vocab and w_rank <= RANK_TRUST:
        return word

    letters = "abcdefghijklmnopqrstuvwxyz"
    L = len(w)

    # 1-edit neighbors by operation
    insertions = set()
    for i in range(L + 1):
        for ch in letters:
            insertions.add(w[:i] + ch + w[i:])
    substitutions = set()
    for i in range(L):
        for ch in letters:
            if ch != w[i]:
                substitutions.add(w[:i] + ch + w[i+1:])
    deletions = { w[:i] + w[i+1:] for i in range(L) }

    def best_by_rank(cands):
        present = [c for c in cands if c in vocab_set]
        if not present:
            return None
        present.sort(key=lambda x: rank_map.get(x, 10**12))
        return present[0]

    # Prefer insertion > substitution > deletion
    for group in (insertions, substitutions, deletions):
        pick = best_by_rank(group)
        if pick:
            # If original is rare but in vocab, require a clear frequency win
            if in_vocab:
                if rank_map.get(w, 10**9) - rank_map.get(pick, 10**12) >= CLEAR_WIN:
                    return pick
                else:
                    return word
            return pick

    # Fallback: bounded DP sweep over near-length words
    best, best_d, best_rank = w, max_dist + 1, 10**12
    for v in vocab_list:
        if abs(len(v) - L) > max_dist:
            continue
        d = levenshtein_distance(w, v, max_dist=max_dist)
        if d <= max_dist:
            r = rank_map.get(v, 10**12)
            if (d < best_d) or (d == best_d and r < best_rank):
                best, best_d, best_rank = v, d, r
                if d == 1 and r <= 10_000:
                    break

    if best_d <= max_dist:
        if in_vocab:
            if rank_map.get(w, 10**9) - best_rank >= CLEAR_WIN:
                return best
            else:
                return word
        return best

    return word

# ----------------------------
# Utilities / data containers
# ----------------------------

class Artifacts:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

        self.paragraphs  = self.root / "paragraphs.jsonl"
        self.bookmeta    = self.root / "books.jsonl"
        self.index       = self.root / "inverted_index.json"
        self.vocab       = self.root / "vocab.txt"           # BPE token vocab
        self.merges      = self.root / "merges.txt"          # BPE merges
        self.tfidf       = self.root / "tfidf.jsonl"         # per-paragraph tfidf
        self.tfidf_norms = self.root / "tfidf_norms.json"    # per-paragraph norms
        self.norms       = self.tfidf_norms                  # alias used elsewhere
        self.word_vocab  = self.root / "word_vocab.txt"      # word-level vocab for autocorrect


# -----------------------
# Step 1: Load the data
# -----------------------

def load_books(data_dir: Path) -> Iterable[Tuple[str, Path]]:
    for p in sorted(data_dir.rglob("*.txt")):
        yield p.stem, p


def read_book_text(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[warn] failed reading {fp}: {e}", file=sys.stderr)
        return ""


# -----------------------------------------------
# Step 2: Split into paragraphs & save as JSONL
# -----------------------------------------------

def split_paragraphs(raw_text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", raw_text)
    paras = []
    for s in parts:
        s = s.strip()
        if len(s) >= 120:  # keep non-trivial paragraphs
            paras.append(s)
    return paras


def build_and_save_paragraphs(data_dir: Path, artifacts: Artifacts) -> None:
    with artifacts.paragraphs.open("w", encoding="utf-8") as wp, \
         artifacts.bookmeta.open("w", encoding="utf-8") as wm:
        pid = 0
        for book_id, fp in load_books(data_dir):
            raw = read_book_text(fp)
            paras = split_paragraphs(raw)
            wm.write(json.dumps({"book_id": book_id, "title": book_id}) + "\n")
            for para in paras:
                wp.write(json.dumps({"paragraph_id": pid, "book_id": book_id, "text": para}) + "\n")
                pid += 1
    print(f"[ok] wrote: {artifacts.paragraphs} and {artifacts.bookmeta}")


# ---------------------------------------------
# Step 3: Train a tiny BPE tokenizer (greedy)
# ---------------------------------------------

def train_bpe(artifacts: Artifacts, vocab_size: int = 3000, min_freq: int = 10,
              sample_paragraphs: int = 20000, top_k_words: int = 50000) -> None:
    """
    Train a tiny BPE from a sample for speed.
    - Reuse merges if they already exist and are non-empty.
    - Sample paragraphs deterministically.
    - Keep only the top-K most frequent words from the sample.
    """
    import random

    if artifacts.merges.exists() and artifacts.merges.stat().st_size > 0:
        print(f"[ok] using existing merges: {artifacts.merges}")
        return

    print(f"[bpe] training on a sample: {sample_paragraphs} paragraphs; "
          f"top {top_k_words} words; vocab_size={vocab_size}; min_freq={min_freq}")

    rng = random.Random(42)
    sample_lines: List[str] = []
    with artifacts.paragraphs.open("r", encoding="utf-8") as f:
        k = sample_paragraphs
        for i, line in enumerate(f, 1):
            if i <= k:
                sample_lines.append(line)
            else:
                j = rng.randint(1, i)
                if j <= k:
                    sample_lines[j - 1] = line

    # word frequencies from sample
    word_freq = Counter()
    for line in sample_lines:
        rec = json.loads(line)
        for w in re.findall(r"\w+", rec["text"].lower()):
            word_freq[w] += 1

    most_common = word_freq.most_common(top_k_words)
    word_freq = Counter(dict(most_common))

    # initialize vocab as sequences of chars + </w>
    def to_syms(w: str):
        return tuple(list(w) + ["</w>"])
    vocab = {to_syms(w): c for w, c in word_freq.items()}

    def pair_counts_fn(vocab_map):
        pc = defaultdict(int)
        for syms, freq in vocab_map.items():
            for i in range(len(syms) - 1):
                pc[(syms[i], syms[i + 1])] += freq
        return pc

    def merge_once(vocab_map, pair):
        a, b = pair
        new_vocab = {}
        for syms, freq in vocab_map.items():
            i = 0
            out = []
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            t = tuple(out)
            new_vocab[t] = new_vocab.get(t, 0) + freq
        return new_vocab

    merges: List[Tuple[str, str]] = []
    for _ in range(vocab_size):
        pc = pair_counts_fn(vocab)
        if not pc:
            break
        (best_pair, cnt) = max(pc.items(), key=lambda x: x[1])
        if cnt < min_freq:
            break
        merges.append(best_pair)
        vocab = merge_once(vocab, best_pair)

    with artifacts.merges.open("w", encoding="utf-8") as w:
        for a, b in merges:
            w.write(f"{a} {b}\n")
    print(f"[ok] trained BPE: {len(merges)} merges -> {artifacts.merges}")


def bpe_tokenize(text: str, merges_path: Path) -> List[str]:
    """
    Apply learned BPE merges to text (greedy).
    """
    # cache merges on function object
    if not hasattr(bpe_tokenize, "_merges"):
        merges = []
        if merges_path.exists():
            for line in merges_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                a, b = line.split()
                merges.append((a, b))
        bpe_tokenize._merges = {(a, b): a + b for (a, b) in merges}

    merges = bpe_tokenize._merges

    pieces = re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
    out_tokens: List[str] = []

    for piece in pieces:
        if not re.match(r"\w+$", piece):
            out_tokens.append(piece)
            continue

        symbols = list(piece) + ["</w>"]
        merged = True
        while merged and len(symbols) > 1:
            merged = False
            i = 0
            new_syms = []
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) in merges:
                    new_syms.append(merges[(symbols[i], symbols[i + 1])])
                    i += 2
                    merged = True
                else:
                    new_syms.append(symbols[i])
                    i += 1
            symbols = new_syms

        for s in symbols:
            if s != "</w>":
                out_tokens.append(s)

    return out_tokens


# -----------------------------------------------
# Step 5: Build inverted index & token/word vocab
# -----------------------------------------------

def build_inverted_index_and_vocab(artifacts: Artifacts) -> None:
    """
    Build:
      - inverted index: BPE token -> {para_id: term_frequency}
      - vocab.txt: all unique BPE tokens (one per line)
      - word_vocab.txt: word-level vocab (for autocorrect) with min frequency
    """
    index: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    bpe_vocab: Set[str] = set()
    word_vocab_counter = Counter()

    with artifacts.paragraphs.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["paragraph_id"]
            text = rec["text"]

            for w in re.findall(r"\w+", text.lower()):
                word_vocab_counter[w] += 1

            toks = bpe_tokenize(text, artifacts.merges)
            for t in toks:
                bpe_vocab.add(t)
            for t, cnt in Counter(toks).items():
                index[t][pid] += cnt

    # save inverted index (json with str keys)
    with artifacts.index.open("w", encoding="utf-8") as w:
        json.dump({tok: {str(pid): tf for pid, tf in postings.items()}
                   for tok, postings in index.items()}, w)

    # save BPE token vocab
    artifacts.vocab.write_text("\n".join(sorted(bpe_vocab)), encoding="utf-8")

    # save word-level vocab for autocorrect (freq >= 1)
    MIN_FREQ = 1
    common_words = [w for (w, c) in word_vocab_counter.items() if c >= MIN_FREQ]
    common_words.sort(key=lambda w: word_vocab_counter[w], reverse=True)

    artifacts.word_vocab.write_text("\n".join(common_words), encoding="utf-8")

    print(f"[ok] wrote: {artifacts.index} and {artifacts.vocab}")
    print(f"[ok] wrote: {artifacts.word_vocab} (word-level vocab for autocorrect)")


# ------------------------------------
# Step 6: Compute TF-IDF (sparse vecs)
# ------------------------------------

def compute_tfidf(artifacts: Artifacts) -> None:
    with artifacts.index.open("r", encoding="utf-8") as f:
        index = json.load(f)

    inv_index: Dict[str, Dict[int, int]] = {
        tok: {int(pid): int(tf) for pid, tf in postings.items()}
        for tok, postings in index.items()
    }

    para_ids: Set[int] = set()
    for postings in inv_index.values():
        para_ids.update(postings.keys())
    N = len(para_ids)

    df: Dict[str, int] = {tok: len(postings) for tok, postings in inv_index.items()}
    idf: Dict[str, float] = {tok: math.log((N + 1) / (df_val + 1)) + 1.0
                             for tok, df_val in df.items()}

    by_para: Dict[int, Dict[str, int]] = defaultdict(dict)
    for tok, postings in inv_index.items():
        for pid, tf in postings.items():
            by_para[pid][tok] = tf

    norms: Dict[int, float] = {}
    with artifacts.tfidf.open("w", encoding="utf-8") as w:
        for pid, tf_map in by_para.items():
            vec: Dict[str, float] = {tok: tf * idf[tok] for tok, tf in tf_map.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            norms[pid] = norm
            w.write(json.dumps({"paragraph_id": pid, "tfidf": vec}) + "\n")

    with artifacts.norms.open("w", encoding="utf-8") as wn:
        json.dump({str(pid): val for pid, val in norms.items()}, wn)

    print(f"[ok] wrote: {artifacts.tfidf} and {artifacts.norms} (N={N} paragraphs)")


# ------------------------------------------------
# Step 8: Query processing, ranking & UI helpers
# ------------------------------------------------

def cosine_similarity(qvec: Dict[str, float], dvec: Dict[str, float], dnorm: float) -> float:
    dot = 0.0
    for tok, qv in qvec.items():
        dv = dvec.get(tok)
        if dv:
            dot += qv * dv
    return dot / (dnorm or 1.0)


def query_to_vec(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = Counter(tokens)
    vec = {tok: tf[tok] * idf.get(tok, 0.0) for tok in tf}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def load_tfidf_and_norms(artifacts: Artifacts) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float]]:
    tfidf_by_para: Dict[int, Dict[str, float]] = {}
    with artifacts.tfidf.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tfidf_by_para[int(rec["paragraph_id"])] = {k: float(v) for k, v in rec["tfidf"].items()}
    norms = {int(k): float(v) for k, v in json.loads(artifacts.norms.read_text(encoding="utf-8")).items()}
    return tfidf_by_para, norms


def load_index_and_idf(artifacts: Artifacts) -> Tuple[Dict[str, Dict[int, int]], Dict[str, float]]:
    index_raw = json.loads(artifacts.index.read_text(encoding="utf-8"))
    inv_index = {tok: {int(pid): int(tf) for pid, tf in postings.items()}
                 for tok, postings in index_raw.items()}
    para_ids: Set[int] = set()
    for postings in inv_index.values():
        para_ids.update(postings.keys())
    N = len(para_ids)
    df = {tok: len(postings) for tok, postings in inv_index.items()}
    idf = {tok: math.log((N + 1) / (df_val + 1)) + 1.0 for tok, df_val in df.items()}
    return inv_index, idf


def candidate_paragraphs(query_tokens: List[str], inv_index: Dict[str, Dict[int, int]]) -> Set[int]:
    cand: Set[int] = set()
    for tok in query_tokens:
        postings = inv_index.get(tok)
        if postings:
            cand.update(postings.keys())
    return cand


def format_snippet(text: str, max_len: int = 240) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + " …"


def search_once(query: str, artifacts: Artifacts, top_k: int = 10) -> List[Dict[str, Any]]:
    inv_index, idf = load_index_and_idf(artifacts)
    tfidf_by_para, norms = load_tfidf_and_norms(artifacts)

    # 1) word-level autocorrect first (distance=3)
    word_vocab = load_word_vocab(artifacts)
    parts = re.findall(r"\w+|[^\w\s]", query.lower(), flags=re.UNICODE)
    corrected_parts: List[str] = [
        autocorrect_token(tok, word_vocab, max_dist=3) if re.match(r"\w+$", tok) else tok
        for tok in parts
    ]
    corrected_query = " ".join(corrected_parts)
    if corrected_query != query.lower():
        print(f"[hint] Did you mean: {corrected_query}")

    # 2) BPE tokenize corrected query
    tokens = bpe_tokenize(corrected_query, artifacts.merges)

    # 3) retrieve candidates and prepare paragraph text lookup (for whole-word overlap)
    cand = candidate_paragraphs(tokens, inv_index)
    qvec = query_to_vec(tokens, idf)

    titles: Dict[str, str] = {}
    if artifacts.bookmeta.exists():
        with artifacts.bookmeta.open("r", encoding="utf-8") as bm:
            for line in bm:
                r = json.loads(line)
                titles[r["book_id"]] = r.get("title", r["book_id"])

    pid_to_text: Dict[int, Tuple[str, str]] = {}
    if artifacts.paragraphs.exists():
        with artifacts.paragraphs.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                pid_to_text[r["paragraph_id"]] = (r["book_id"], r["text"])

    # whole-word overlap helpers
    query_words = {w for w in re.findall(r"\w+", corrected_query) if w}

    def whole_word_hits(pid: int) -> int:
        _, txt = pid_to_text.get(pid, ("", ""))
        if not txt or not query_words:
            return 0
        para_words = set(re.findall(r"\w+", txt.lower()))
        return len(query_words & para_words)

    # 4) score with optional filter/bonus
    scored: List[Tuple[int, float]] = []
    for pid in cand:
        hits = whole_word_hits(pid)

        # optional FILTER
        if WHOLE_WORD_MIN_MATCHES and hits < WHOLE_WORD_MIN_MATCHES:
            continue

        sim = cosine_similarity(qvec, tfidf_by_para.get(pid, {}), norms.get(pid, 1.0))

        # optional BOOST
        if WHOLE_WORD_MATCH_BONUS:
            sim += hits * WHOLE_WORD_MATCH_BONUS

        if sim > 0:
            scored.append((pid, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    # 5) hydrate results
    results = []
    for pid, score in top:
        book_id, text = pid_to_text.get(pid, ("?", ""))
        title = titles.get(book_id, book_id)
        results.append({
            "paragraph_id": pid,
            "book_id": book_id,
            "title": title,
            "score": round(score, 4),
            "snippet": format_snippet(text)
        })
    return results


def interactive_loop(artifacts: Artifacts) -> None:
    print("\nType your query (or 'exit' to quit):")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            return
        if q.lower() in {"exit", "quit"}:
            print("bye!")
            return
        if not q:
            continue
        hits = search_once(q, artifacts, top_k=10)
        if not hits:
            print("No paragraph found.")
            continue
        for i, h in enumerate(hits, 1):
            print(f"{i:2d}. [pid={h['paragraph_id']}] {h['title']}  (score={h['score']})")
            print(f"    {h['snippet']}\n")


# ---------------------------
# Orchestration (build/main)
# ---------------------------

def build_pipeline(data_dir: Path, artifacts: Artifacts) -> None:
    print("[1/4] Paragraphizing …")
    build_and_save_paragraphs(data_dir, artifacts)

    print("[2/4] Training BPE …")
    train_bpe(artifacts, vocab_size=10000, min_freq=2)

    print("[3/4] Building inverted index & vocab …")
    build_inverted_index_and_vocab(artifacts)

    print("[4/4] Computing TF-IDF …")
    compute_tfidf(artifacts)

    print("[ok] Build complete.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mini search engine with autocorrect")
    p.add_argument("--data_dir", type=Path, required=True,
                   help="Directory containing .txt books (start with ~100; later 3000)")
    p.add_argument("--artifact_dir", type=Path, default=Path("./artifacts"),
                   help="Where to store paragraphs/index/bpe/vocab/tfidf")
    p.add_argument("--build", action="store_true", help="Run the offline build pipeline")
    p.add_argument("--search", action="store_true", help="Enter interactive search mode")
    p.add_argument("--query", type=str, default=None, help="Run a single search and exit")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    artifacts = Artifacts(args.artifact_dir)

    if args.build:
        build_pipeline(args.data_dir, artifacts)

    if args.query:
        results = search_once(args.query, artifacts, top_k=10)
        if not results:
            print("No paragraph found.")
        else:
            for i, h in enumerate(results, 1):
                print(f"{i:2d}. [pid={h['paragraph_id']}] {h['title']}  (score={h['score']})")
                print(f"    {h['snippet']}\n")

    if args.search:
        interactive_loop(artifacts)

    if not (args.build or args.search or args.query):
        print("Nothing to do. Try --build and/or --search. See --help.", file=sys.stderr)


if __name__ == "__main__":
    main()