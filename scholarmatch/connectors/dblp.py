"""DBLP Computer Science Bibliography API Connector."""

from typing import List, Dict, Any, Optional
import requests

from scholarmatch.config import REQUEST_TIMEOUT, DEFAULT_USER_AGENT


class DBLPClient:
    """Connector for the open DBLP computer science bibliography service."""

    BASE_URL = "https://dblp.org/search/author/api"

    def __init__(self):
        self.headers = {"User-Agent": DEFAULT_USER_AGENT}

    def search_authors(self, name_query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search DBLP authors by name."""
        params = {
            "q": name_query,
            "format": "json",
            "h": limit
        }
        try:
            resp = requests.get(self.BASE_URL, headers=self.headers, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
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
                    "dblp_id": hit.get("@id")
                })

            return authors
        except Exception:
            return []
