"""OpenAlex Academic REST API Connector."""

from typing import List, Dict, Any, Optional, Tuple
import requests

from scholarmatch.config import OPENALEX_BASE_URL, DEFAULT_USER_AGENT, REQUEST_TIMEOUT
from scholarmatch.models.schemas import Publication, FacultyProfile, ActiveGrant


class OpenAlexClient:
    """Client for OpenAlex API with polite user-agent and structured schema conversion."""

    def __init__(self, email: Optional[str] = None):
        self.headers = {
            "User-Agent": f"ScholarMatch/0.1.0 ({f'mailto:{email}' if email else DEFAULT_USER_AGENT})"
        }

    def search_works(self, query: str, limit: int = 10) -> List[Publication]:
        """Search OpenAlex works for a given query."""
        url = f"{OPENALEX_BASE_URL}/works"
        params = {
            "search": query,
            "per-page": limit,
            "sort": "relevance_score:desc"
        }
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return []
            data = resp.json()
            results = data.get("results", [])
            publications: List[Publication] = []

            for item in results:
                title = item.get("title") or "Untitled Paper"
                # OpenAlex stores abstracts as inverted index
                abstract_inverted = item.get("abstract_inverted_index")
                abstract = self._reconstruct_abstract(abstract_inverted) or "Abstract not available."
                year = item.get("publication_year") or 2024
                doi = item.get("doi")
                citations = item.get("cited_by_count", 0)

                # Extract concepts / keywords
                concepts = [c.get("display_name") for c in item.get("concepts", []) if c.get("display_name")]
                referenced_works = [r.replace("https://openalex.org/", "") for r in item.get("referenced_works", [])]

                primary_location = item.get("primary_location") or {}
                source = primary_location.get("source") or {}
                venue = source.get("display_name") or "Academic Venue"

                publications.append(Publication(
                    title=title,
                    abstract=abstract,
                    year=year,
                    venue=venue,
                    doi=doi,
                    citation_count=citations,
                    keywords=concepts[:6],
                    references=referenced_works[:10]
                ))

            return publications
        except Exception:
            return []

    def search_authors(self, name_query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search authors on OpenAlex."""
        url = f"{OPENALEX_BASE_URL}/authors"
        params = {"search": name_query, "per-page": limit}
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return []
            data = resp.json()
            authors = []
            for item in data.get("results", []):
                affiliations = item.get("last_known_institutions", [])
                inst_name = affiliations[0].get("display_name") if affiliations else "Unknown Institution"
                summary_stats = item.get("summary_stats", {})
                authors.append({
                    "id": item.get("id"),
                    "name": item.get("display_name"),
                    "institution": inst_name,
                    "works_count": item.get("works_count", 0),
                    "cited_by_count": item.get("cited_by_count", 0),
                    "h_index": summary_stats.get("h_index", 0),
                    "i10_index": summary_stats.get("i10_index", 0),
                    "top_concepts": [c.get("display_name") for c in item.get("x_concepts", [])[:5]]
                })
            return authors
        except Exception:
            return []

    def _reconstruct_abstract(self, inverted_index: Optional[Dict[str, List[int]]]) -> str:
        """Reconstruct plain abstract text from OpenAlex inverted index structure."""
        if not inverted_index:
            return ""
        word_positions: List[Tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join([w[1] for w in word_positions])
