import spacy
import pandas as pd
from modules.cache import cache

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except:
    raise ValueError("spaCy model 'en_core_web_sm' is not installed. Run: python -m spacy download en_core_web_sm")


# ------------------------- ENTITY EXTRACTION -------------------------

def extract_entities_from_text(text):
    """Extract named entities from a single text record."""
    if not isinstance(text, str):
        text = str(text)

    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


def extract_entities_from_data(df):
    """Extract entities for each record in the dataset."""
    try:
        if "body" not in df.columns:
            raise KeyError("Dataset must contain a 'body' column.")

        results = []

        for idx, row in df.iterrows():
            text = row["body"]
            entities = extract_entities_from_text(text)

            results.append({
                "id": idx,
                "text": text,
                "entities": entities
            })

        cache.add_log("INFO", f"Extracted entities from {len(results)} records")
        return pd.DataFrame(results)

    except Exception as e:
        cache.add_log("ERROR", f"Entity extraction failed: {str(e)}")
        raise e


# ------------------------- RELATION EXTRACTION -------------------------

def extract_relations_from_text(text):
    """Simple rule-based relation extraction using dependency parsing."""
    if not isinstance(text, str):
        text = str(text)

    doc = nlp(text)
    triples = []

    for token in doc:
        # Main verb or key relation word
        if token.dep_ in ("ROOT", "attr") and token.pos_ == "VERB":
            subject = None
            obj = None

            # Find subject
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child.text

            # Find object
            for child in token.children:
                if child.dep_ in ("dobj", "attr", "pobj"):
                    obj = child.text

            if subject and obj:
                triples.append((subject, token.lemma_, obj))

    return triples


def extract_relations_from_data(df):
    """Extract subject–relation–object triples from each record."""
    try:
        if "body" not in df.columns:
            raise KeyError("Dataset must contain a 'body' column.")

        all_triples = []

        for idx, row in df.iterrows():
            text = row["body"]
            triples = extract_relations_from_text(text)

            for s, r, o in triples:
                all_triples.append({
                    "TextID": idx,
                    "Subject": s,
                    "Relation": r,
                    "Object": o
                })

        cache.add_log("INFO", f"Extracted {len(all_triples)} triples")
        return pd.DataFrame(all_triples)

    except Exception as e:
        cache.add_log("ERROR", f"Relation extraction failed: {str(e)}")
        raise e
