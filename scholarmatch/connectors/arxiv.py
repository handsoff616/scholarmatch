"""arXiv API Open Access Preprint Connector."""

import xml.etree.ElementTree as ET
from typing import List, Optional
import requests

from scholarmatch.config import REQUEST_TIMEOUT, DEFAULT_USER_AGENT
from scholarmatch.models.schemas import Publication


class ArxivClient:
    """Connector for the open-access arXiv REST API."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self):
        self.headers = {"User-Agent": DEFAULT_USER_AGENT}

    def search_preprints(self, query: str, max_results: int = 8) -> List[Publication]:
        """Search arXiv preprints by topic, methodology, or author."""
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        try:
            resp = requests.get(self.BASE_URL, headers=self.headers, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return []

            root = ET.fromstring(resp.content)
            # Atom XML namespace
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            publications: List[Publication] = []

            for entry in entries:
                title = entry.find("atom:title", ns)
                title_text = title.text.strip().replace("\n", " ") if title is not None else "Untitled Preprint"

                summary = entry.find("atom:summary", ns)
                abstract_text = summary.text.strip().replace("\n", " ") if summary is not None else ""

                published = entry.find("atom:published", ns)
                year_val = int(published.text[:4]) if published is not None and len(published.text) >= 4 else 2024

                id_elem = entry.find("atom:id", ns)
                arxiv_url = id_elem.text.strip() if id_elem is not None else ""

                # Primary category / tags
                categories = []
                for cat in entry.findall("atom:category", ns):
                    term = cat.attrib.get("term")
                    if term:
                        categories.append(term)

                doi = None
                for link in entry.findall("atom:link", ns):
                    if link.attrib.get("title") == "doi":
                        doi = link.attrib.get("href", "").replace("http://dx.doi.org/", "")

                publications.append(Publication(
                    title=title_text,
                    abstract=abstract_text,
                    year=year_val,
                    venue="arXiv Preprint",
                    doi=doi or arxiv_url,
                    citation_count=0,
                    keywords=categories[:4]
                ))

            return publications
        except Exception:
            return []
