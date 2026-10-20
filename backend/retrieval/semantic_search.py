"""
PhoenixAI hybrid retrieval.
Combines:
1. Dense semantic search using the existing E5 embedding index.
2. BM25-style lexical search for exact Hebrew terms.
3. Reciprocal Rank Fusion (RRF) to merge rankings.
4. Source diversity and irrelevant-page filtering.
The existing embeddings.npy file is reused.
No crawling or index rebuilding is required.
"""
import json
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"
INDEX_FILE = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
MODEL_NAME = "intfloat/multilingual-e5-small"
logger = logging.getLogger("phoenixai.search")
LOG_LEVEL = os.getenv("RETRIEVAL_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
TOKEN_PATTERN = re.compile(r"[\w\u0590-\u05FF]+", re.UNICODE)
HEBREW_PREFIXES = ("ו", "ה", "ב", "כ", "ל", "מ", "ש")
STOPWORDS = {
    "אני", "את", "אתה", "אתם", "אתן", "היא", "הוא",
    "של", "על", "עם", "או", "גם", "אם", "מה", "מי",
    "איך", "האם", "יש", "אין", "זה", "זו", "זאת",
    "לי", "לך", "לכם", "שלי", "שלך", "שלכם",
    "כדי", "אפשר", "ניתן", "צריך", "צריכה",
    "רוצה", "רוצים", "מתי", "כמה", "איפה",
    "אצל", "אחרי", "לפני", "דרך", "יותר",
    "הפניקס", "באתר",
}
BLOCKED_PATHS = (
    "/404/",
    "/career/",
    "/accessibility/",
    "/terms-of-use/",
    "/privacy-policy/",
    "/contact-us/call-back/",
)
BLOCKED_TITLE_TERMS = (
    "שגיאה |",
    "הגשת קורות חיים",
    "מדיניות שמירת הפרטיות",
    "תנאי שימוש",
    "הסדרי הנגישות",
)
CLAIMS_TERMS = {
    "תביעה", "תביעות", "הגשת", "להגיש",
    "נזק", "נזקי", "פיצוי",
}
HOME_TERMS = {
    "דירה", "בדירה", "דירת", "בית",
    "מים", "נזילה", "צנרת", "רטיבות",
    "שרברב", "אינסטלציה", "הצפה",
}
HOME_PATH_HINTS = (
    "/apartment-insurance/",
    "/claims-appartment-insurance/",
)
CLAIMS_PATH_HINTS = (
    "/claims-customer-service/",
    "/claims-",
)
RRF_K = 60
@lru_cache(maxsize=1)
def get_model():
    """Load the same multilingual E5 model through compact ONNX inference."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    model_dir = PROJECT_ROOT / "models" / "e5-onnx"
    model_path = model_dir / "onnx" / "model_quantized.onnx"
    tokenizer_path = model_dir / "tokenizer.json"
    if not model_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(
            "ONNX model files are missing. Set Render Build Command to: "
            "pip install -r requirements-render.txt && python scripts/download_e5_onnx.py"
        )
    logger.info("Loading local quantized ONNX E5 model: %s", model_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    session = ort.InferenceSession(
        model_path, sess_options=options, providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=256)
    return E5OnnxEncoder(session, tokenizer)


class E5OnnxEncoder:
    """Match SentenceTransformer E5 query prefix, mean pooling and L2 normalization."""

    def __init__(self, session, tokenizer):
        self.session = session
        self.tokenizer = tokenizer
        self.input_names = {item.name for item in session.get_inputs()}

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=8):
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        vectors = []
        for offset in range(0, len(values), batch_size):
            batch = values[offset:offset + batch_size]
            encoded = self.tokenizer.encode_batch(batch)
            width = max(len(item.ids) for item in encoded)
            ids = np.zeros((len(encoded), width), dtype=np.int64)
            mask = np.zeros((len(encoded), width), dtype=np.int64)
            for row, item in enumerate(encoded):
                length = len(item.ids)
                ids[row, :length] = item.ids
                mask[row, :length] = item.attention_mask
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            feeds = {key: value for key, value in feeds.items() if key in self.input_names}
            output = self.session.run(None, feeds)[0]
            # Attention-mask-aware mean pooling, as used by E5 SentenceTransformer.
            weights = mask[..., None].astype(np.float32)
            pooled = (output * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1)
            if normalize_embeddings:
                pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            vectors.extend(pooled.astype(np.float32))
        result = np.asarray(vectors, dtype=np.float32)
        return result[0] if single else result


@lru_cache(maxsize=1)
def load_chunks() -> list[dict]:
    start = time.perf_counter()
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    logger.info(
        "Loaded %d chunks in %.2fs",
        len(chunks),
        time.perf_counter() - start,
    )
    return chunks
@lru_cache(maxsize=1)
def load_embeddings() -> np.ndarray:
    start = time.perf_counter()
    embeddings = np.load(INDEX_FILE, allow_pickle=False, mmap_mode="r")
    logger.info(
        "Loaded embedding index: shape=%s in %.2fs",
        embeddings.shape,
        time.perf_counter() - start,
    )
    return embeddings
def build_index() -> None:
    """Index building is disabled in the lean runtime to preserve existing vectors."""
    raise RuntimeError("Keep existing embeddings.npy. Rebuild only in the original PyTorch environment.")


def tokenize(text: str) -> list[str]:
    """Tokenize Hebrew and English text."""
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if len(token) > 1
    ]
def normalize_token(token: str) -> str:
    """
    Apply conservative normalization.
    Remove a single common Hebrew prefix only when the
    remaining token is sufficiently long.
    """
    token = token.lower()
    if (
        len(token) >= 5
        and token.startswith(HEBREW_PREFIXES)
    ):
        return token[1:]
    return token
def expand_tokens(text: str) -> list[str]:
    """Include original and normalized forms."""
    result = []
    for token in tokenize(text):
        if token in STOPWORDS:
            continue
        result.append(token)
        normalized = normalize_token(token)
        if normalized != token and normalized not in STOPWORDS:
            result.append(normalized)
    return result
def is_valid_source(chunk: dict) -> bool:
    """Allow only relevant public Phoenix website pages."""
    url = str(chunk.get("url", "")).strip()
    title = str(chunk.get("title", "")).strip()
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if (
        parsed.scheme != "https"
        or parsed.hostname != "www.fnx.co.il"
    ):
        return False
    path = parsed.path.lower()
    if any(blocked in path for blocked in BLOCKED_PATHS):
        return False
    if any(term in title for term in BLOCKED_TITLE_TERMS):
        return False
    content = str(chunk.get("content", "")).strip()
    return len(content) >= 80
@lru_cache(maxsize=1)
def get_lexical_index() -> dict:
    """Build a lightweight in-memory BM25 index."""
    start = time.perf_counter()
    chunks = load_chunks()
    term_frequencies = []
    document_lengths = []
    document_frequencies = defaultdict(int)
    valid_indices = []
    for index, chunk in enumerate(chunks):
        if not is_valid_source(chunk):
            term_frequencies.append(Counter())
            document_lengths.append(0)
            continue
        title = str(chunk.get("title", ""))
        content = str(chunk.get("content", ""))
        # Repeating title terms improves page-title matching.
        tokens = (
            expand_tokens(title) * 3
            + expand_tokens(content)
        )
        frequencies = Counter(tokens)
        term_frequencies.append(frequencies)
        document_lengths.append(len(tokens))
        valid_indices.append(index)
        for term in frequencies:
            document_frequencies[term] += 1
    average_length = (
        sum(document_lengths[index] for index in valid_indices)
        / max(len(valid_indices), 1)
    )
    result = {
        "term_frequencies": term_frequencies,
        "document_lengths": document_lengths,
        "document_frequencies": dict(document_frequencies),
        "valid_indices": valid_indices,
        "average_length": max(average_length, 1),
        "document_count": len(valid_indices),
    }
    logger.info(
        "Lexical index ready: %d valid chunks, %.2fs",
        len(valid_indices),
        time.perf_counter() - start,
    )
    return result
def lexical_search(
    query: str,
    top_k: int = 40,
) -> list[tuple[int, float]]:
    """Rank chunks using BM25-style lexical relevance."""
    index = get_lexical_index()
    query_terms = set(expand_tokens(query))
    if not query_terms:
        return []
    k1 = 1.5
    b = 0.75
    total_documents = index["document_count"]
    average_length = index["average_length"]
    scores = []
    for document_index in index["valid_indices"]:
        frequencies = index["term_frequencies"][document_index]
        document_length = index["document_lengths"][document_index]
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = index[
                "document_frequencies"
            ].get(term, 0)
            idf = math.log(
                1
                + (
                    total_documents - document_frequency + 0.5
                ) / (document_frequency + 0.5)
            )
            denominator = (
                frequency
                + k1 * (
                    1 - b
                    + b * document_length / average_length
                )
            )
            score += idf * (
                frequency * (k1 + 1) / denominator
            )
        if score > 0:
            scores.append((document_index, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]
def semantic_search(
    query: str,
    top_k: int = 40,
) -> list[tuple[int, float]]:
    """Search the existing normalized E5 embedding index."""
    chunks = load_chunks()
    embeddings = load_embeddings()
    if len(chunks) != len(embeddings):
        raise ValueError(
            "Chunk count does not match embedding count. "
            "Rebuild the index."
        )
    model = get_model()
    query_embedding = model.encode(
        f"query: {query}",
        normalize_embeddings=True,
    )
    scores = embeddings @ query_embedding
    candidate_count = min(
        max(top_k * 5, 100),
        len(chunks),
    )
    indices = np.argsort(scores)[::-1][:candidate_count]
    results = []
    for index in indices:
        index = int(index)
        if not is_valid_source(chunks[index]):
            continue
        results.append((index, float(scores[index])))
        if len(results) >= top_k:
            break
    return results
def get_topic_boost(query: str, chunk: dict) -> float:
    """Add a small domain-specific ranking bonus."""
    query_tokens = set(expand_tokens(query))
    title = str(chunk.get("title", "")).lower()
    url = str(chunk.get("url", "")).lower()
    boost = 0.0
    has_home_topic = bool(query_tokens & HOME_TERMS)
    has_claims_topic = bool(query_tokens & CLAIMS_TERMS)
    if has_home_topic and any(
        hint in url for hint in HOME_PATH_HINTS
    ):
        boost += 0.018
    if has_claims_topic and any(
        hint in url for hint in CLAIMS_PATH_HINTS
    ):
        boost += 0.018
    if has_home_topic and (
        "דירה" in title
        or "מים" in title
        or "צנרת" in title
    ):
        boost += 0.008
    if has_claims_topic and (
        "תביעה" in title
        or "תביעות" in title
    ):
        boost += 0.008
    return boost
def search(
    query: str,
    top_k: int = 4,
) -> list[dict]:
    """
    Perform hybrid retrieval with RRF.
    Returns the original chunk metadata plus:
    - score: fused ranking score
    - semantic_score
    - lexical_score
    """
    if not query or not query.strip():
        return []
    if top_k <= 0:
        return []
    start = time.perf_counter()
    chunks = load_chunks()
    semantic_start = time.perf_counter()
    dense_results = semantic_search(query, top_k=60)
    semantic_seconds = time.perf_counter() - semantic_start
    lexical_start = time.perf_counter()
    keyword_results = lexical_search(query, top_k=60)
    lexical_seconds = time.perf_counter() - lexical_start
    logger.info(
        "Hybrid search query=%r | semantic=%d (%.2fs) "
        "| lexical=%d (%.2fs)",
        query[:150],
        len(dense_results),
        semantic_seconds,
        len(keyword_results),
        lexical_seconds,
    )
    fused_scores = defaultdict(float)
    semantic_scores = {}
    lexical_scores = {}
    # Lexical matching receives slightly more weight to
    # prevent unrelated high-similarity pages from winning.
    for rank, (index, score) in enumerate(dense_results, 1):
        fused_scores[index] += 1.0 / (RRF_K + rank)
        semantic_scores[index] = score
    for rank, (index, score) in enumerate(keyword_results, 1):
        fused_scores[index] += 1.5 / (RRF_K + rank)
        lexical_scores[index] = score
    for index in fused_scores:
        fused_scores[index] += get_topic_boost(
            query,
            chunks[index],
        )
    ranked_indices = sorted(
        fused_scores,
        key=lambda index: fused_scores[index],
        reverse=True,
    )
    # Prefer distinct pages before repeating the same URL.
    selected_indices = []
    seen_urls = set()
    deferred_indices = []
    for index in ranked_indices:
        url = chunks[index].get("url", "")
        if url in seen_urls:
            deferred_indices.append(index)
            continue
        seen_urls.add(url)
        selected_indices.append(index)
        if len(selected_indices) >= top_k:
            break
    if len(selected_indices) < top_k:
        for index in deferred_indices:
            selected_indices.append(index)
            if len(selected_indices) >= top_k:
                break
    results = []
    for rank, index in enumerate(selected_indices, 1):
        chunk = chunks[index]
        result = {
            **chunk,
            "score": round(fused_scores[index], 6),
            "semantic_score": round(
                semantic_scores.get(index, 0.0),
                6,
            ),
            "lexical_score": round(
                lexical_scores.get(index, 0.0),
                6,
            ),
        }
        results.append(result)
        logger.info(
            "RESULT #%d | fused=%.4f | dense=%.4f "
            "| lexical=%.4f | title=%s | url=%s",
            rank,
            result["score"],
            result["semantic_score"],
            result["lexical_score"],
            str(result.get("title", ""))[:90],
            result.get("url", ""),
        )
    logger.info(
        "Hybrid search completed in %.2fs",
        time.perf_counter() - start,
    )
    return results
if __name__ == "__main__":
    build_index()
