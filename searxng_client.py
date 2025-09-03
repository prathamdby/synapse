"""SearXNG client for the Telegram bot."""

import os
import logging
import aiohttp
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class SearxngClient:
    """Client for interacting with a self-hosted SearXNG instance."""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("SEARXNG_BASE_URL")
        if not self.base_url:
            raise ValueError("SearXNG base URL is required")

        # Ensure the base URL doesn't end with a slash
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

    async def search(
        self,
        query: str,
        categories: List[str] = None,
        engines: List[str] = None,
        language: str = "en",
        format: str = "json",
        page: int = 1,
        time_range: str = None,
        max_results: int = 10,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Perform a search using SearXNG API.

        Args:
            query: Search query
            categories: List of categories to search in
            engines: List of engines to use
            language: Language code (default: "en")
            format: Response format (default: "json")
            page: Page number for pagination (default: 1)
            time_range: Time range filter (default: None)
            max_results: Maximum number of results to return (default: 10)

        Returns:
            List of search results or None if error occurred
        """
        try:
            # Build query parameters
            params = {
                "q": query,
                "format": format,
                "language": language,
                "pageno": page,
            }

            # Add optional parameters
            if categories:
                params["categories"] = ",".join(categories)
            if engines:
                params["engines"] = ",".join(engines)
            if time_range:
                params["time_range"] = time_range

            # Construct the full URL
            url = f"{self.base_url}/search"

            # Make the request
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Extract results and limit to max_results
                    results = data.get("results", [])
                    return results[:max_results] if results else []

        except aiohttp.ClientError as e:
            logger.error(f"Error making request to SearXNG: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during SearXNG search: {e}")
            return None

    async def search_with_summary(
        self,
        query: str,
        categories: List[str] = None,
        language: str = "en",
        max_results: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        Perform a search and return results with a summary.

        Args:
            query: Search query
            categories: List of categories to search in
            language: Language code (default: "en")
            max_results: Maximum number of results to return (default: 5)

        Returns:
            Dictionary with search results and summary, or None if error occurred
        """
        results = await self.search(
            query=query,
            categories=categories,
            language=language,
            max_results=max_results,
        )

        if results is None:
            return None

        # Create a summary of the results
        summary = {"query": query, "total_results": len(results), "results": []}

        for result in results:
            summary["results"].append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": (
                        result.get("content", "")[:200] + "..."
                        if len(result.get("content", "")) > 200
                        else result.get("content", "")
                    ),
                    "engine": result.get("engine", ""),
                }
            )

        return summary
