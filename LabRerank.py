"""Lab: Re-ranking a RAG pipeline.

Two-stage retrieval over SciFact, a set of 5,183 biomedical abstracts and 300
test claims with known relevant abstracts. Stage 1 (given) is a fast bi-encoder
search. Stage 2 is a cross-encoder that re-scores the stage 1 candidates.

Search for "TODO" to find the four places you write code.

STEP 4 and 7: Checking Claims
    Improvement - No green checks before re-ranking; 1 green check after.
    • 'In adult tissue, most T cells are memory T cells'
    • 'Incidence rates of cervical cancer have decreased'

    Best improvement - No green checks before re-ranking; 2 green checks after.
    • 'Obesity is determined solely by environmental factors'

    No improvement - No green checks before or after re-ranking.
    • '0-dimensional biomaterials show inductive properties'
    • 'Ivermectin is used to treat onchocerciasis'

STEP 5
Why does build_index() put the title in front of every chunk?
    When an abstract is cut into multiple chunks, the later chunks might not
    say what the paper is about. Putting the title in front of every
    chunk gives each one the topic, so the embedding knows which paper it
    belongs to and the search can still match it.

Why is to_doc_ids() needed?
    to_doc_ids() turns ranked chunks into ranked abstract ids and removes repeats, 
    so we compare the right things and one abstract can't fill the whole top 3 relevant results.

STEP 10
Why does re-ranking help so little at 5 candidates, and what does it cost at 100?
    • After setting the slider to 5, the 'Obesity' claim results 
      were exactly the same before and after re-ranking.
    • At 10, 1 green check mark after re-ranking.
    • At 30, 2 green check marks.
    • At 100, still 2.

    • The re-ranker only re-sorts what stage 1 already found. With 5 candidates
    there is barely anything to re-sort, and if the right abstract is not in
    those 5, the re-ranker won't ever find it. At 100 candidates it has many
    more to choose from, but since it has to read 100 claim + chunk pairs 
    for every claim, the latency goes up a lot.
    • Recall@3 stayed at 0.736 for 5 candidates and went from 0.741 to 0.767 at 100. 
        For 5 candidates, the re-ranking took 174 ms, and for 100 candidates, it took 3925 ms, 
        so much slower than stage 1.

STEP 13
    • Claim: 'Bone marrow cells contribute to adult macrophage compartments'

    • Which one is right: REFUTED. The claim says bone marrow cells contribute to
    adult macrophage compartments, but all three abstracts say adult tissue
    macrophages mostly come from before birth and maintain themselves with
    minimal input from bone marrow or circulating monocytes. The NOT ENOUGH
    EVIDENCE answer was too cautious, because the abstracts do not just leave
    the question open, they point against the claim.

    • What made the difference: Both runs used the same three abstracts
    (12827098, 7521113, 22406695), but re-ranking put them in a different
    order. The LLM reads the abstracts in the order it is given, and the
    different order changed how it weighed the evidence.

    • Answers:
        Without re-ranking
        REFUTED: The abstracts consistently indicate that adult tissue-resident 
        macrophages are primarily established before birth and maintain themselves 
        independently of circulating monocytes or bone marrow-derived cells in the steady state. 
        There is minimal contribution from circulating monocytes or bone marrow cells to adult 
        macrophage compartments under normal conditions [12827098, 7521113, 22406695].

        With re-ranking
        NOT ENOUGH EVIDENCE: The abstracts indicate that most adult tissue macrophages originate 
        during embryonic development and maintain themselves locally without significant contribution 
        from circulating monocytes or bone marrow-derived cells under steady-state conditions 
        [12827098, 22406695, 7521113]. However, they do not explicitly address whether bone marrow 
        cells contribute to adult macrophage compartments under other conditions or to any extent.
"""

import json
import time

import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

DATA_DIR = "data/scifact"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHAT_MODEL = "gpt-4.1-mini"

CHUNK_WORDS = 120    # words per chunk
CHUNK_OVERLAP = 30   # words shared by neighbouring chunks
CANDIDATES = 30      # chunks stage 1 hands to the reranker
CONTEXT_K = 3        # abstracts that go into the prompt
EVAL_QUERIES = 100   # test claims used in Part C

# TODO (Part D): write the system prompt. It must
#   - tell the model to use only the abstracts in {context}
#   - ask for a verdict: SUPPORTED, REFUTED, or NOT ENOUGH EVIDENCE
#   - ask it to cite the abstracts it used by their [id]
SYSTEM_TEMPLATE = """You are a careful assistant that checks medical claims
against research abstracts.

Use ONLY the abstracts given below. Do not use any outside knowledge.
Each abstract starts with its id in square brackets, like [12345].

Give exactly one verdict:
- SUPPORTED: the abstracts show the claim is true
- REFUTED: the abstracts show the claim is false
- NOT ENOUGH EVIDENCE: the abstracts do not settle it

Start your answer with the verdict. Then explain in one to three short
sentences, and cite every abstract you used by its [id].

Abstracts:
{context}"""


# ---------- Given: loading ----------
@st.cache_data
def load_scifact(data_dir=DATA_DIR):
    """Return (corpus, queries, qrels).

    corpus:  doc_id -> {"title", "text"}
    queries: query_id -> claim text (the 300 test claims)
    qrels:   query_id -> set of relevant doc_ids (the answer key)
    """
    corpus = {}
    with open(f"{data_dir}/corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = {"title": row["title"], "text": row["text"]}

    queries = {}
    with open(f"{data_dir}/queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[row["_id"]] = row["text"]

    qrels = {}
    with open(f"{data_dir}/qrels/test.tsv", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            query_id, doc_id, _ = line.strip().split("\t")
            qrels.setdefault(query_id, set()).add(doc_id)

    return corpus, queries, qrels


# ---------- Given: chunking ----------
def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Split text into windows of `size` words that overlap by `overlap` words."""
    words = text.split()
    step = size - overlap
    starts = range(0, max(len(words) - overlap, 1), step)
    return [" ".join(words[start:start + size]) for start in starts]


# ---------- Given: embedding ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def build_index(size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Chunk every abstract, embed every chunk, keep a chunk -> doc_id map."""
    corpus, _, _ = load_scifact()
    doc_ids, texts = [], []
    for doc_id, doc in corpus.items():
        for piece in chunk_text(doc["text"], size, overlap):
            doc_ids.append(doc_id)
            texts.append(f"{doc['title']}. {piece}")

    vectors = load_embedder().encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    return {"doc_ids": doc_ids, "texts": texts, "vectors": vectors}


# ---------- Given: stage 1 retrieval ----------
def retrieve(index, query, k=CANDIDATES):
    """Return the top k chunks as [(chunk_index, cosine_score), ...]."""
    query_vector = load_embedder().encode(query, normalize_embeddings=True)
    scores = index["vectors"] @ query_vector
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]


def to_doc_ids(index, hits):
    """Collapse ranked chunks into ranked abstracts (first chunk wins)."""
    ranked = []
    for chunk_index, _ in hits:
        doc_id = index["doc_ids"][chunk_index]
        if doc_id not in ranked:
            ranked.append(doc_id)
    return ranked


# ---------- Part B: stage 2 re-ranking ----------
@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANK_MODEL)


def rerank(index, query, hits):
    """Re-score stage 1 hits with the cross-encoder, best first.

    TODO (Part B):
      1. Build a list of (query, chunk_text) pairs, one per hit.
         The chunk text is index["texts"][chunk_index].
      2. Score all pairs at once with load_reranker().predict(pairs).
      3. Return the hits as [(chunk_index, reranker_score), ...], highest first.
    """
    # (query, chunk text) pair for every hit
    pairs = [(query, index["texts"][chunk_index]) for chunk_index, _ in hits]

    # scoring all the pairs in 1 predict() call
    scores = load_reranker().predict(pairs)

    # same (chunk_index, score) format, but with the new scores
    reranked = [(hits[n][0], float(scores[n])) for n in range(len(hits))]
    reranked.sort(key=lambda hit: hit[1], reverse=True)
    return reranked


# ---------- Given: metrics ----------
def recall_at_k(ranked, relevant, k):
    """Share of the relevant abstracts that appear in the top k."""
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked, relevant):
    """1 / rank of the first relevant abstract (0 if none was retrieved)."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1 / rank
    return 0.0


# ---------- Part C: measure before and after ----------
def evaluate(index, n_queries=EVAL_QUERIES, candidates=CANDIDATES, k=CONTEXT_K):
    """Average recall@k, MRR and latency, with and without re-ranking.

    TODO (Part C): for the first n_queries claims in qrels,
      - run retrieve(), then rerank(), timing each with time.perf_counter()
      - turn both hit lists into doc ids with to_doc_ids()
      - score both with recall_at_k() and reciprocal_rank()
    Return a dict of averages with these keys:
      recall_before, recall_after, rr_before, rr_after, retrieve_ms, rerank_ms
    """
    _, queries, qrels = load_scifact()

    recall_before, recall_after = [], []
    rr_before, rr_after = [], []
    retrieve_times, rerank_times = [], []

    for query_id in list(qrels)[:n_queries]:
        claim = queries[query_id]
        relevant = qrels[query_id]

        # stage 1, timed
        start = time.perf_counter()
        hits = retrieve(index, claim, candidates)
        retrieve_times.append((time.perf_counter() - start) * 1000)

        # stage 2, timed
        start = time.perf_counter()
        reranked = rerank(index, claim, hits)
        rerank_times.append((time.perf_counter() - start) * 1000)

        # chunks -> abstracts
        ranked_before = to_doc_ids(index, hits)
        ranked_after = to_doc_ids(index, reranked)

        # score both
        recall_before.append(recall_at_k(ranked_before, relevant, k))
        recall_after.append(recall_at_k(ranked_after, relevant, k))
        rr_before.append(reciprocal_rank(ranked_before, relevant))
        rr_after.append(reciprocal_rank(ranked_after, relevant))

    return {
        "recall_before": float(np.mean(recall_before)),
        "recall_after": float(np.mean(recall_after)),
        "rr_before": float(np.mean(rr_before)),
        "rr_after": float(np.mean(rr_after)),
        "retrieve_ms": float(np.mean(retrieve_times)),
        "rerank_ms": float(np.mean(rerank_times)),
    }


# ---------- Part D: generation ----------
def build_context(doc_ids):
    corpus, _, _ = load_scifact()
    return "\n\n".join(
        f"[{doc_id}] {corpus[doc_id]['title']}\n{corpus[doc_id]['text']}"
        for doc_id in doc_ids
    )


def verify_claim(claim, doc_ids):
    """TODO (Part D): send SYSTEM_TEMPLATE (filled with build_context(doc_ids))
    and the claim to CHAT_MODEL, and return the model's reply text."""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    system_prompt = SYSTEM_TEMPLATE.format(context=build_context(doc_ids))
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Claim: {claim}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# ---------- UI ----------
def show_docs(doc_ids, relevant):
    corpus, _, _ = load_scifact()
    for rank, doc_id in enumerate(doc_ids, start=1):
        mark = " ✅" if doc_id in relevant else ""
        st.markdown(f"**{rank}. [{doc_id}]**{mark} {corpus[doc_id]['title']}")


def main():
    st.title("Lab: Re-ranking a RAG Pipeline")
    corpus, queries, qrels = load_scifact()

    candidates = st.sidebar.slider("Stage 1 candidates", 5, 100, CANDIDATES, step=5)
    k = st.sidebar.slider("Abstracts sent to the LLM", 1, 5, CONTEXT_K)

    with st.spinner("Chunking and embedding 5,183 abstracts (first run only)..."):
        index = build_index()
    st.caption(f"{len(corpus):,} abstracts, {len(index['texts']):,} chunks indexed")

    # ----- Part A: pick a claim, look at stage 1 -----
    test_ids = list(qrels)
    query_id = st.selectbox(
        "Pick a test claim", test_ids, format_func=lambda qid: queries[qid]
    )
    custom = st.text_input("...or type your own claim (no answer key)")
    claim = custom or queries[query_id]
    relevant = set() if custom else qrels[query_id]

    hits = retrieve(index, claim, candidates)
    before = to_doc_ids(index, hits)

    # ----- Part B: side by side -----
    start = time.perf_counter()
    after = to_doc_ids(index, rerank(index, claim, hits))
    rerank_ms = (time.perf_counter() - start) * 1000

    left, right = st.columns(2)
    with left:
        st.subheader("Without re-ranking")
        show_docs(before[:k], relevant)
    with right:
        st.subheader("With re-ranking")
        show_docs(after[:k], relevant)
    st.caption(f"Re-ranking {candidates} candidates took {rerank_ms:.0f} ms. ✅ = in the answer key.")

    # ----- Part C -----
    # TODO (Part C): add a button that runs evaluate(index, EVAL_QUERIES, candidates, k)
    # and shows Recall@k, MRR and added latency with st.metric (use delta= for the change).
    st.divider()
    st.header("Measure before and after")
    if st.button(f"Run evaluation on {EVAL_QUERIES} claims"):
        with st.spinner("Evaluating... (more candidates takes longer)"):
            r = evaluate(index, EVAL_QUERIES, candidates, k)

        c1, c2, c3 = st.columns(3)
        c1.metric(
            f"Recall@{k}",
            f"{r['recall_after']:.3f}",
            delta=f"{r['recall_after'] - r['recall_before']:+.3f}",
        )
        c2.metric(
            "MRR",
            f"{r['rr_after']:.3f}",
            delta=f"{r['rr_after'] - r['rr_before']:+.3f}",
        )
        c3.metric(
            "Extra time per claim",
            f"{r['rerank_ms']:.0f} ms",
            delta=f"+{r['rerank_ms']:.0f} ms (stage 1 alone: {r['retrieve_ms']:.0f} ms)",
            delta_color="inverse",
        )
        st.caption(
            f"Before re-ranking: Recall@{k} = {r['recall_before']:.3f}, "
            f"MRR = {r['rr_before']:.3f}. Candidates = {candidates}."
        )

    # ----- Part D -----
    # TODO (Part D): add a button that calls verify_claim() twice, once with before[:k]
    # and once with after[:k], and shows the two answers in side-by-side columns.
    st.divider()
    st.header("Ask the LLM")
    if st.button("Verify this claim"):
        with st.spinner("Asking the LLM..."):
            answer_before = verify_claim(claim, before[:k])
            answer_after = verify_claim(claim, after[:k])

        a1, a2 = st.columns(2)
        with a1:
            st.subheader("Without re-ranking")
            st.write(answer_before)
        with a2:
            st.subheader("With re-ranking")
            st.write(answer_after)


if __name__ == "__main__":
    main()