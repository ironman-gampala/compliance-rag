"""Region and document-type labels for each PDF."""

import json

from rag.config import DOCS_DIR

CATALOG_PATH = DOCS_DIR / "catalog.json"
UNSPECIFIED = "unspecified"


def load_catalog() -> dict[str, dict[str, str]]:
    if not CATALOG_PATH.exists():
        return {}
    data = json.loads(CATALOG_PATH.read_text())
    catalog = {}
    for name, meta in data.items():
        catalog[name] = {
            "region": str(meta.get("region") or UNSPECIFIED),
            "doc_type": str(meta.get("doc_type") or UNSPECIFIED),
            "title": str(meta.get("title") or name),
        }
    return catalog


def labels_for(filename: str, catalog: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    catalog = load_catalog() if catalog is None else catalog
    meta = catalog.get(filename)
    if meta is None:
        return {"region": UNSPECIFIED, "doc_type": UNSPECIFIED, "title": filename}
    return meta
