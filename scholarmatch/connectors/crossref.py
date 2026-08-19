"""CrossRef DOI Resolver and Metadata Client."""

from typing import Optional, Dict, Any
import requests

from scholarmatch.config import CROSSREF_BASE_URL, DEFAULT_USER_AGENT, REQUEST_TIMEOUT
from scholarmatch.models.schemas import Publication


class CrossRefClient:
    """Client for resolving DOI metadata via CrossRef REST API."""

    def __init__(self, email: Optional[str] = None):
        self.headers = {
            "User-Agent": f"ScholarMatch/0.1.0 ({f'mailto:{email}' if email else DEFAULT_USER_AGENT})"
        }

    def resolve_doi(self, doi: str) -> Optional[Publication]:
        """Fetch canonical metadata for a specific DOI."""
        clean_doi = doi.strip().replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
        url = f"{CROSSREF_BASE_URL}/works/{clean_doi}"

        try:
            resp = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return None
            data = resp.json().get("message", {})
            title_list = data.get("title", [])
            title = title_list[0] if title_list else "Unknown Title"
            abstract = data.get("abstract", "Abstract not indexed on CrossRef.")
            # Remove XML/JATS tags if present in CrossRef abstract
            import re
            abstract_clean = re.sub(r"<[^>]+>", "", abstract).strip()

            issued = data.get("issued", {}).get("date-parts", [[2024]])
            year = issued[0][0] if issued and issued[0] else 2024
            container = data.get("container-title", [])
            venue = container[0] if container else "Peer-Reviewed Venue"
            citations = data.get("is-referenced-by-count", 0)

            # References
            reference_entries = data.get("reference", [])
            ref_dois = [r.get("DOI") for r in reference_entries if r.get("DOI")]

            return Publication(
                title=title,
                abstract=abstract_clean,
                year=int(year),
                venue=venue,
                doi=clean_doi,
                citation_count=citations,
                keywords=data.get("subject", []),
                references=ref_dois[:10]
            )
        except Exception:
            return None
