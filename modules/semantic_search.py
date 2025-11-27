# modules/semantic_search.py
import numpy as np
from typing import List, Dict, Any
from modules.cache import cache

# Lazy import to allow graceful error message if package missing
try:
    from sentence_transformers import SentenceTransformer, util
except Exception as e:
    SentenceTransformer = None
    util = None
    IMPORT_ERROR = str(e)
else:
    IMPORT_ERROR = None


class SemanticSearch:
    """
    Builds embeddings over (1) raw texts, (2) triples (Subject - Relation - Object),
    and (3) entity labels, then performs semantic search across the combined index.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            cache.add_log("ERROR", f"sentence-transformers import failed: {IMPORT_ERROR}")
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        try:
            self.model = SentenceTransformer(model_name)
            cache.add_log("INFO", f"Loaded SentenceTransformer model '{model_name}'")
        except Exception as e:
            cache.add_log("ERROR", f"Failed to load embedding model: {str(e)}")
            raise

    # ----------------------------
    # Build index
    # ----------------------------
    def build_index(self, df_texts: Any = None, df_triples: Any = None, entities_list: List[str] = None):
        """
        Build and cache embeddings for:
          - df_texts: dataframe with a 'Text' or 'text' column (original sentences)
          - df_triples: dataframe with columns ['Subject','Relation','Object']
          - entities_list: list of entity strings
        The method is flexible: pass whichever datasets you have.
        """
        try:
            index_items = []  # list of strings
            index_meta = []   # list of dicts with type and source info

            # 1) Raw text rows
            if df_texts is not None:
                # prefer 'Text' then 'text' then 'body'
                if "Text" in df_texts.columns:
                    col = "Text"
                elif "text" in df_texts.columns:
                    col = "text"
                elif "body" in df_texts.columns:
                    col = "body"
                else:
                    col = None

                if col:
                    texts = df_texts[col].fillna("").astype(str).tolist()
                    for i, t in enumerate(texts):
                        index_items.append(t)
                        index_meta.append({"type": "text", "source": col, "index": i})

            # 2) Triples (Subject Relation Object)
            if df_triples is not None:
                expected_cols = {"Subject", "Relation", "Object"}
                if expected_cols.issubset(set(df_triples.columns)):
                    for i, row in df_triples.iterrows():
                        s = str(row["Subject"])
                        r = str(row["Relation"])
                        o = str(row["Object"])
                        triple_text = f"{s} {r} {o}"
                        index_items.append(triple_text)
                        index_meta.append({"type": "triple", "source": "triples", "index": i, "subject": s, "object": o, "relation": r})
                else:
                    cache.add_log("WARNING", "build_index: df_triples missing required columns (Subject/Relation/Object)")

            # 3) Entities (unique names)
            if entities_list:
                for i, ent in enumerate(entities_list):
                    ent_str = str(ent)
                    index_items.append(ent_str)
                    index_meta.append({"type": "entity", "source": "entities", "index": i, "entity": ent_str})

            if len(index_items) == 0:
                cache.add_log("ERROR", "build_index: Nothing to index (no texts, triples, or entities provided).")
                raise ValueError("No indexable content provided")

            # Compute embeddings
            cache.add_log("INFO", f"Computing embeddings for {len(index_items)} items...")
            embeddings = self.model.encode(index_items, convert_to_tensor=True, show_progress_bar=False)

            # Cache everything
            cache.set("semantic_index_items", index_items)
            cache.set("semantic_index_meta", index_meta)
            cache.set("semantic_embeddings", embeddings)

            cache.add_log("INFO", f"Semantic index built: {len(index_items)} items cached.")
            return True

        except Exception as e:
            cache.add_log("ERROR", f"Failed to build semantic index: {str(e)}")
            raise

    # ----------------------------
    # Search
    # ----------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search the cached index and return top_k matches with metadata + scores."""
        try:
            if not cache.has("semantic_embeddings"):
                cache.add_log("ERROR", "search called but embeddings not built.")
                raise RuntimeError("Embeddings not built. Call build_index(...) first.")

            index_items = cache.get("semantic_index_items")
            index_meta = cache.get("semantic_index_meta")
            embeddings = cache.get("semantic_embeddings")

            query_emb = self.model.encode(query, convert_to_tensor=True)
            scores = util.cos_sim(query_emb, embeddings)[0].cpu().numpy()  # numpy array

            # get top_k indices
            top_idx = np.argsort(scores)[::-1][:top_k]

            results = []
            for rank, idx in enumerate(top_idx, start=1):
                results.append({
                    "rank": rank,
                    "score": float(scores[idx]),
                    "text": index_items[idx],
                    "meta": index_meta[idx]
                })

            cache.add_log("INFO", f"Semantic search for '{query}' returned {len(results)} results.")
            return results

        except Exception as e:
            cache.add_log("ERROR", f"Semantic search failed: {str(e)}")
            raise

    # ----------------------------
    # Utility to list searchable node candidates (entities + subjects + objects)
    # ----------------------------
    @staticmethod
    def extract_entities_from_session(entities_df):
        """
        Flatten your session entities dataframe (expected structure: columns ['id','text','entities'])
        to a unique list of entity labels.
        """
        if entities_df is None:
            return []

        unique = set()
        for ent_list in entities_df.get("entities", []):
            # ent_list may be a python-list string or an actual list; handle both
            if isinstance(ent_list, str):
                # attempt to parse naive format: "[('A','TYPE'),('B','TYPE')]"
                try:
                    # use eval in a controlled way; if this is risky in your environment, skip
                    parsed = eval(ent_list)
                except Exception:
                    parsed = []
                pairs = parsed
            else:
                pairs = ent_list or []

            for p in pairs:
                if isinstance(p, (list, tuple)) and len(p) >= 1:
                    unique.add(str(p[0]))
            # fallback: if pair strings, ignore

        return sorted(unique)
