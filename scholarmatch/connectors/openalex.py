"""OpenAlex Academic REST API Connector."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import requests

from scholarmatch.config import OPENALEX_BASE_URL, DEFAULT_USER_AGENT, REQUEST_TIMEOUT
from scholarmatch.models.schemas import Publication, FacultyProfile, ActiveGrant
from scholarmatch.connectors.http_utils import get_resilient_session

logger = logging.getLogger(__name__)


class OpenAlexClient:
    """Client for OpenAlex API with polite user-agent and structured schema conversion."""

    def __init__(self, email: Optional[str] = None):
        self.headers = {
            "User-Agent": f"ScholarMatch/0.1.0 ({f'mailto:{email}' if email else DEFAULT_USER_AGENT})"
        }
        self.session = get_resilient_session(retries=3)

    def search_works(self, query: str, limit: int = 10) -> List[Publication]:
        """Search OpenAlex works for a given query."""
        url = f"{OPENALEX_BASE_URL}/works"
        params = {
            "search": query,
            "per-page": limit,
            "sort": "relevance_score:desc"
        }
        try:
            resp = self.session.get(url, headers=self.headers, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logger.warning("OpenAlex search_works returned status %d for query %s", resp.status_code, query)
                return []
            data = resp.json()
            results = data.get("results", [])
            publications: List[Publication] = []

            for item in results:
                title = item.get("title") or "Untitled Paper"
                abstract_inverted = item.get("abstract_inverted_index")
                abstract = self._reconstruct_abstract(abstract_inverted) or "Abstract not available."
                year = item.get("publication_year") or 2024
                doi = item.get("doi")
                citations = item.get("cited_by_count", 0)

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
        except requests.RequestException as e:
            logger.warning("OpenAlex search_works network request failed: %s", e)
            return []
        except Exception as e:
            logger.exception("Failed to parse OpenAlex works response: %s", e)
            return []

    def search_authors(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search OpenAlex authors by name or keywords."""
        url = f"{OPENALEX_BASE_URL}/authors"
        params = {
            "search": query,
            "per-page": limit
        }
        try:
            resp = self.session.get(url, headers=self.headers, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logger.warning("OpenAlex search_authors returned status %d for query %s", resp.status_code, query)
                return []
            data = resp.json()
            authors: List[Dict[str, Any]] = []

            for item in data.get("results", []):
                aff_info = item.get("last_known_institution") or {}
                inst_name = aff_info.get("display_name") or "Academic Institution"
                summary_stats = item.get("summary_stats") or {}
                concepts = [c.get("display_name") for c in item.get("x_concepts", []) if c.get("display_name")]

                authors.append({
                    "id": item.get("id"),
                    "name": item.get("display_name"),
                    "institution": inst_name,
                    "works_count": item.get("works_count", 0),
                    "cited_by_count": item.get("cited_by_count", 0),
                    "h_index": summary_stats.get("h_index", 0),
                    "top_concepts": concepts[:5],
                    "source": "OpenAlex"
                })

            return authors
        except requests.RequestException as e:
            logger.warning("OpenAlex search_authors network request failed: %s", e)
            return []
        except Exception as e:
            logger.exception("Failed to parse OpenAlex author response: %s", e)
            return []

    def _reconstruct_abstract(self, inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
        """Reconstruct abstract from OpenAlex inverted index dictionary."""
        if not inverted_index:
            return None

        word_pos: List[Tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_pos.append((pos, word))

        word_pos.sort(key=lambda x: x[0])
        return " ".join([w[1] for w in word_pos])
