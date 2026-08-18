import sys
import os

sys.path.append(os.getcwd())
import time
import langid
import asyncio
import requests
import aiolimiter
from typing import Optional, Union, Dict
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor

from .base_tool import BaseTool
from .cache_manager import PreprocessCacheManager


class BingSearchTool(BaseTool):
    """BingSearchTool"""

    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    def __init__(
        self,
        api_key: str,
        zone: str = "your_zone",
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "cn",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        requests_per_second: float = 2.0,
        search_cache_file="",
        wikipedia_fallback: bool = True,
        wikipedia_timeout: int = 15,
    ):
        self._api_key = api_key
        self._zone = zone
        self._max_results = max_results
        self._result_length = result_length
        self._location = location
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._wikipedia_fallback = wikipedia_fallback
        self._wikipedia_timeout = wikipedia_timeout

        self._limiter = aiolimiter.AsyncLimiter(
            max_rate=requests_per_second, time_period=1.0
        )
        self.search_cache_manager = PreprocessCacheManager(search_cache_file)

    @property
    def name(self) -> str:
        return "bing_search"

    @property
    def trigger_tag(self) -> str:
        return "search"

    def _call_request(self, query, headers, payload, timeout):
        error_cnt = 0
        while True:
            if error_cnt >= self._max_retries:
                print(
                    f"query: {query} has tried {error_cnt} times without success, just skip it."
                )
                break
            try:
                response = requests.post(
                    "https://api.brightdata.com/request",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                search_results = response.json()
                return search_results
            except requests.exceptions.Timeout:
                error_cnt += 1
                print(
                    f"error_cnt: {error_cnt}, Bing Web Search request timed out ({timeout} seconds) for query: {query}"
                )
                time.sleep(5)
            except requests.exceptions.RequestException as e:
                error_cnt += 1
                print(
                    f"error_cnt: {error_cnt}, Error occurred during Bing Web Search request: {e}, payload: {payload}"
                )
                time.sleep(5)
        return None

    def _pack_query(self, query):
        if langid.classify(query)[0] == "zh":
            mkt, setLang = "zh-CN", "zh"
        else:
            mkt, setLang = "en-US", "en"
        input_obj = {"q": query, "mkt": mkt, "setLang": setLang}
        encoded_query = urlencode(input_obj)
        return encoded_query

    def _make_request(self, query: str, timeout: int):
        """
        Send a request to the Brightdata API.

        Args:
            query: The search query.
            timeout: Request timeout in seconds.
            
        Returns:
            The response object.
        """
        encoded_query = self._pack_query(query)
        target_url = f"https://www.bing.com/search?{encoded_query}&brd_json=1&cc={self._location}"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"zone": self._zone, "url": target_url, "format": "raw"}
        result = self._call_request(query, headers, payload, timeout)
        return result

    async def postprocess_search_result(self, query, response, **kwargs) -> str:
        data = response
        if "organic" not in data:
            data["chunk_content"] = []
            result = self._format_results(data)
        else:
            chunk_content_list = []
            seen_snippets = set()
            for result_item in data["organic"]:
                snippet = result_item.get("description", "").strip()
                if snippet and snippet not in seen_snippets:
                    chunk_content_list.append(snippet)
                    seen_snippets.add(snippet)
            data["chunk_content"] = chunk_content_list
            result = self._format_results(data)
        return result

    async def execute(self, query: str, timeout: int = 60, **kwargs) -> str:
        """
        Execute a Bing search query with support for cache and semantic similarity cache hits.

        Args:
            query: The search query text.
            timeout: Request timeout in seconds.
            model: SBERT model used for semantic search.
            threshold: Minimum similarity threshold.
            top_k: Number of top most similar cached entries to consider.

        Returns:
            A string containing the search result.
        """
        hit_cache = self.search_cache_manager.hit_cache(query)
        from_cache = False
        if hit_cache:
            print("hit cache: ", query)
            response = hit_cache
            from_cache = True
        else:
            loop = asyncio.get_event_loop()

            async with self._limiter:
                response = await loop.run_in_executor(
                    self._executor, lambda: self._make_request(query, timeout)
                )
            if response is None:
                fallback = await self._wikipedia_fallback_search(
                    query, "Bing request failed", timeout
                )
                return fallback or f"Bing search failed: {query}"
        assert response is not None
        if not self._has_bing_results(response):
            fallback = await self._wikipedia_fallback_search(
                query, "Bing returned no organic results", timeout
            )
            if fallback:
                return fallback
        if not from_cache:
            await self.search_cache_manager.add_to_cache(query, response)
        try:
            result = await self.postprocess_search_result(query, response, **kwargs)
        except Exception as exc:
            fallback = await self._wikipedia_fallback_search(
                query, f"Bing postprocess failed: {exc}", timeout
            )
            if fallback:
                return fallback
            raise
        if self._is_empty_result(result):
            fallback = await self._wikipedia_fallback_search(
                query, "Bing returned no usable snippets", timeout
            )
            if fallback:
                return fallback
        return result or f"Bing search failed: {query}"

    def _has_bing_results(self, response: Dict) -> bool:
        organic = response.get("organic") if isinstance(response, dict) else None
        if not organic:
            return False
        return any(item.get("description", "").strip() for item in organic)

    def _is_empty_result(self, result: Optional[str]) -> bool:
        if result is None:
            return True
        normalized = result.strip().lower()
        return normalized in {"", "none", "no results found.", "no search results found."}

    async def _wikipedia_fallback_search(
        self, query: str, reason: str, timeout: int
    ) -> Optional[str]:
        if not self._wikipedia_fallback:
            return None
        print(f"{reason}; trying Wikipedia fallback for query: {query}")
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                lambda: self._search_wikipedia(
                    query, timeout=min(timeout, self._wikipedia_timeout)
                ),
            )
        except Exception as exc:
            print(f"Wikipedia fallback failed for query {query!r}: {exc}")
            return None
        if self._is_empty_result(result):
            print(f"Wikipedia fallback returned no results for query: {query}")
            return None
        return result

    def _search_wikipedia(self, query: str, timeout: int) -> str:
        lang = "zh" if langid.classify(query)[0] == "zh" else "en"
        endpoint = f"https://{lang}.wikipedia.org/w/api.php"
        headers = {"User-Agent": "cbpo-bing-wikipedia-fallback/1.0"}
        search_response = requests.get(
            endpoint,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": self._max_results,
                "format": "json",
            },
            headers=headers,
            timeout=timeout,
        )
        search_response.raise_for_status()
        hits = search_response.json().get("query", {}).get("search", [])
        if not hits:
            return "No search results found."

        page_ids = [str(hit["pageid"]) for hit in hits[: self._max_results]]
        extract_response = requests.get(
            endpoint,
            params={
                "action": "query",
                "pageids": "|".join(page_ids),
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "format": "json",
            },
            headers=headers,
            timeout=timeout,
        )
        extract_response.raise_for_status()
        pages = extract_response.json().get("query", {}).get("pages", {})
        formatted = []
        for page_id in page_ids:
            page = pages.get(page_id, {})
            title = " ".join(str(page.get("title", "")).split())
            extract = " ".join(str(page.get("extract", "")).split())
            if not extract:
                continue
            snippet = extract[: self._result_length]
            prefix = f"{title}: " if title else ""
            formatted.append(f"Page {len(formatted) + 1}: {prefix}{snippet}")
        return "\n".join(formatted) if formatted else "No search results found."

    def _format_results(self, results: Dict) -> Union[str, None]:
        """Format the search results."""
        if not results.get("chunk_content"):
            return None

        formatted = []
        for idx, snippet in enumerate(results["chunk_content"][: self._max_results], 1):
            snippet = snippet[: self._result_length]
            formatted.append(f"Page {idx}: {snippet}")
        return "\n".join(formatted)
