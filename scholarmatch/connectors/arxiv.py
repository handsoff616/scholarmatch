"""arXiv API Open Access Preprint Connector."""

import logging
from typing import List
import requests

try:
    import defusedxml.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from scholarmatch.config import REQUEST_TIMEOUT, DEFAULT_USER_AGENT
from scholarmatch.models.schemas import Publication
from scholarmatch.connectors.http_utils import get_resilient_session

logger = logging.getLogger(__name__)


class ArxivClient:
    """Connector for the open-access arXiv REST API."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self):
        self.headers = {"User-Agent": DEFAULT_USER_AGENT}
        self.session = get_resilient_session(retries=3)

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
            resp = self.session.get(self.BASE_URL, headers=self.headers, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logger.warning("arXiv API returned non-200 status code: %d", resp.status_code)
                return []

            root = ET.fromstring(resp.content)
            # Atom XML namespace
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            publications: List[Publication] = []

            for entry in entries:
                title = entry.find("atom:title", ns)
                title_text = title.text.strip().replace("\n", " ") if title is not None and title.text else "Untitled Preprint"

                summary = entry.find("atom:summary", ns)
                abstract_text = summary.text.strip().replace("\n", " ") if summary is not None and summary.text else ""

                published = entry.find("atom:published", ns)
                year_val = int(published.text[:4]) if published is not None and published.text and len(published.text) >= 4 else 2024

                id_elem = entry.find("atom:id", ns)
                arxiv_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""

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
        except requests.RequestException as e:
            logger.warning("arXiv network request failed: %s", e)
            return []
        except Exception as e:
            logger.exception("Failed to parse arXiv API response: %s", e)
            return []
