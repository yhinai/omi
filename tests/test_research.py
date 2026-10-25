"""Offline checks for the independent live research service."""
import unittest
from unittest import mock
from aiohttp.test_utils import TestClient, TestServer
import research_server as research


def grounded():
    return {"candidates": [{"finishReason": "STOP", "groundingMetadata": {
        "webSearchQueries": ["underwater volcanic monitoring"],
        "groundingChunks": [{"web": {"uri": "https://example.org/paper", "title": "example.org"}},
                            {"web": {"uri": "javascript:alert(1)", "title": "unsafe"}},
                            {"web": {"uri": "https://example.org/uncited", "title": "uncited"}}],
        "groundingSupports": [{"segment": {"text": "**Ocean sensors** explain underwater volcanic signals."}, "groundingChunkIndices": [0, 1]}],
        "searchEntryPoint": {"renderedContent": "<div>Google Search suggestions</div>"}}}]}


class ResearchParsingTests(unittest.TestCase):
    def test_results_require_grounded_sources_and_keep_suggestions(self):
        result = research.grounded_results(grounded())
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["url"], "https://example.org/paper")
        self.assertNotIn("**", result["results"][0]["summary"])
        self.assertIn("Google Search", result["search_suggestions"])

    def test_private_or_unsafe_links_are_rejected(self):
        for url in ("http://example.org", "javascript:alert(1)", "https://localhost/", "https://127.0.0.1/", "https://[::1]/", "https://127.1/", "https://user:secret@example.org/"):
            with self.subTest(url=url):
                self.assertFalse(research.safe_url(url))
        self.assertTrue(research.safe_url("https://noaa.gov/research"))

    def test_truncated_or_malformed_provider_data_is_not_displayed(self):
        data = grounded()
        data["candidates"][0]["finishReason"] = "MAX_TOKENS"
        with self.assertRaises(ValueError):
            research.grounded_results(data)
        with self.assertRaises(ValueError):
            research.grounded_results({"candidates": []})


class ResearchRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = research.create_app({"GEMINI_API_KEY": "offline-test"})
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.origin = {"Origin": "http://127.0.0.1:8788"}
        self.body = {"context": "How do underwater sensors monitor volcanic activity?"}

    async def test_known_local_origin_gets_results_and_repeated_context_uses_cache(self):
        result = research.grounded_results(grounded())
        with mock.patch.object(research, "search_web", return_value=result) as search:
            for _ in range(2):
                response = await self.client.post("/search", json=self.body, headers=self.origin)
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Access-Control-Allow-Origin"], self.origin["Origin"])
                self.assertEqual(len((await response.json())["results"]), 1)
            search.assert_awaited_once()

    async def test_unknown_sites_and_missing_origin_cannot_start_paid_searches(self):
        with mock.patch.object(research, "search_web") as search:
            for headers in ({}, {"Origin": "https://untrusted.example"}):
                response = await self.client.post("/search", json=self.body, headers=headers)
                self.assertEqual(response.status, 403)
            search.assert_not_called()

    async def test_different_context_is_rate_limited_and_provider_failure_is_sanitized(self):
        with mock.patch.object(research, "search_web", side_effect=RuntimeError("private provider detail")):
            response = await self.client.post("/search", json=self.body, headers=self.origin)
            self.assertEqual(response.status, 503)
            self.assertNotIn("private", await response.text())
            response = await self.client.post("/search", json={"context": "A different scientific research topic entirely."}, headers=self.origin)
            self.assertEqual(response.status, 429)
            self.assertGreater((await response.json())["retry_after"], 0)

    async def test_preflight_and_served_script(self):
        response = await self.client.options("/search", headers=self.origin)
        self.assertEqual(response.status, 204)
        response = await self.client.get("/client.js")
        self.assertEqual(response.status, 200)
        self.assertIn("omiResearch", await response.text())
