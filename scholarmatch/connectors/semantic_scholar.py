"""Semantic Scholar (S2) Graph API Connector."""

from typing import List, Dict, Any, Optional
import requests

from scholarmatch.config import REQUEST_TIMEOUT, DEFAULT_USER_AGENT
from scholarmatch.models.schemas import FacultyProfile, Publication


class SemanticScholarClient:
    """Connector for Semantic Scholar Academic Graph API.

    Enables structured author search, h-index retrieval, paper citation graphs,
    and automatic conversion into FacultyProfile objects.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.headers = {
            "User-Agent": DEFAULT_USER_AGENT
        }
        if api_key:
            self.headers["x-api-key"] = api_key

    def search_authors(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search authors on Semantic Scholar by name."""
        url = f"{self.BASE_URL}/author/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "authorId,name,affiliations,homepage,paperCount,citationCount,hIndex"
        }

        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return []

            data = resp.json().get("data", [])
            authors: List[Dict[str, Any]] = []

            for item in data:
                affs = item.get("affiliations") or []
                inst = affs[0] if affs else "Academic Institution"
                authors.append({
                    "author_id": item.get("authorId"),
                    "name": item.get("name"),
                    "institution": inst,
                    "paper_count": item.get("paperCount", 0),
                    "citation_count": item.get("citationCount", 0),
                    "h_index": item.get("hIndex", 0),
                    "homepage": item.get("homepage"),
                    "profile_url": f"https://www.semanticscholar.org/author/{item.get('authorId')}"
                })

            return authors
        except Exception:
            return []

    def fetch_author_profile(self, author_id: str, limit_papers: int = 10) -> Optional[FacultyProfile]:
        """Fetch full author details and top papers from Semantic Scholar."""
        url = f"{self.BASE_URL}/author/{author_id}"
        params = {
            "fields": (
                "name,affiliations,homepage,paperCount,citationCount,hIndex,"
                "papers.title,papers.abstract,papers.year,papers.venue,papers.citationCount,papers.externalIds"
            )
        }

        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return None

            data = resp.json()
            name = data.get("name", "Unknown Researcher")
            affs = data.get("affiliations") or []
            inst = affs[0] if affs else "Academic Institution"
            h_index = data.get("hIndex", 0)
            total_citations = data.get("citationCount", 0)

            raw_papers = data.get("papers") or []
            publications: List[Publication] = []
            extracted_keywords = set()

            for p in raw_papers[:limit_papers]:
                title = p.get("title") or "Untitled Paper"
                abstract = p.get("abstract") or f"Published work by {name}."
                year = p.get("year") or 2024
                venue = p.get("venue") or "Peer-Reviewed Venue"
                citations = p.get("citationCount", 0)
                ext_ids = p.get("externalIds") or {}
                doi = ext_ids.get("DOI")

                publications.append(Publication(
                    title=title,
                    abstract=abstract,
                    year=year,
                    venue=venue,
                    doi=doi,
                    citation_count=citations,
                    keywords=[]
                ))

            summary = (
                f"Faculty lab of {name} at {inst}. "
                f"Published {len(raw_papers)} papers with {total_citations:,} total citations and an h-index of {h_index}."
            )

            return FacultyProfile(
                id=f"s2-{author_id}",
                name=name,
                institution=inst,
                department="Computer Science / Engineering",
                lab_name=f"{name.split()[-1]} Research Group",
                lab_website=data.get("homepage") or f"https://www.semanticscholar.org/author/{author_id}",
                research_summary=summary,
                specialties=["Artificial Intelligence", "Computational Science"],
                recent_publications=publications,
                active_grants=[],
                h_index=h_index,
                accepting_students=True
            )
        except Exception:
            return None
