"""DBLP Computer Science Bibliography API Connector."""

import logging
from typing import List, Dict, Any, Optional
import requests

from scholarmatch.config import REQUEST_TIMEOUT, DEFAULT_USER_AGENT
from scholarmatch.connectors.http_utils import get_resilient_session

logger = logging.getLogger(__name__)


class DBLPClient:
    """Connector for the open DBLP computer science bibliography service."""

    BASE_URL = "https://dblp.org/search/author/api"

    def __init__(self):
        self.headers = {"User-Agent": DEFAULT_USER_AGENT}
        self.session = get_resilient_session(retries=3)

    def search_authors(self, name_query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search DBLP authors by name."""
        params = {
            "q": name_query,
            "format": "json",
            "h": limit
        }
        try:
            resp = self.session.get(self.BASE_URL, headers=self.headers, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logger.warning("DBLP search returned non-200 status %d for query %s", resp.status_code, name_query)
                return []

            result_data = resp.json().get("result", {}).get("hits", {}).get("hit", [])
            if isinstance(result_data, dict):
                result_data = [result_data]

            authors: List[Dict[str, Any]] = []
            for hit in result_data:
                info = hit.get("info", {})
                author_name = info.get("author")
                url = info.get("url")
                notes_raw = info.get("notes", {}).get("note", []) if isinstance(info.get("notes"), dict) else []
                if isinstance(notes_raw, dict):
                    notes_list = [notes_raw]
                elif isinstance(notes_raw, list):
                    notes_list = notes_raw
                elif isinstance(notes_raw, str):
                    notes_list = [{"text": notes_raw}]
                else:
                    notes_list = []

                affiliations = []
                for n in notes_list:
                    if isinstance(n, dict):
                        text = n.get("text") or n.get("#text") or ""
                        if text:
                            affiliations.append(text)
                    elif isinstance(n, str):
                        affiliations.append(n)

                authors.append({
                    "name": author_name,
                    "dblp_url": url,
                    "affiliations": affiliations,
                    "source": "DBLP"
                })

            return authors
        except requests.RequestException as e:
            logger.warning("DBLP network request failed: %s", e)
            return []
        except Exception as e:
            logger.exception("Failed to parse DBLP API response: %s", e)
            return []
