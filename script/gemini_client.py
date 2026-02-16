"""
Shared Gemini API wrapper for SAM3 track segmentation pipeline.

Uses the new google-genai SDK (``from google import genai``).
Provides a simple interface for text and image+text prompts,
with retry logic, error handling, and native JSON output mode.
"""

import json
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from PIL import Image


class GeminiClient:
    """Wrapper around the Google GenAI (Gemini) API."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-pro",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai is not installed. "
                "Install it with: pip install google-genai"
            )
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Send a text-only prompt to Gemini and return the response text."""
        config = types.GenerateContentConfig(temperature=temperature)
        return self._call_with_retry([prompt], config)

    def generate_with_image(
        self,
        prompt: str,
        image: Union[str, Path, Image.Image],
        temperature: float = 0.2,
    ) -> str:
        """Send an image + text prompt to Gemini and return the response text."""
        if isinstance(image, (str, Path)):
            image = Image.open(str(image)).convert("RGB")
        config = types.GenerateContentConfig(temperature=temperature)
        return self._call_with_retry([prompt, image], config)

    def generate_with_images(
        self,
        prompt: str,
        images: List[Union[str, Path, Image.Image]],
        temperature: float = 0.2,
    ) -> str:
        """Send multiple images + text prompt to Gemini."""
        loaded: List[Any] = [prompt]
        for img in images:
            if isinstance(img, (str, Path)):
                loaded.append(Image.open(str(img)).convert("RGB"))
            else:
                loaded.append(img)
        config = types.GenerateContentConfig(temperature=temperature)
        return self._call_with_retry(loaded, config)

    def generate_json(
        self,
        prompt: str,
        image: Optional[Union[str, Path, Image.Image]] = None,
        images: Optional[List[Union[str, Path, Image.Image]]] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """Generate and parse a JSON response from Gemini.

        Uses ``response_mime_type="application/json"`` for native structured output.
        """
        contents: List[Any] = [prompt]
        if images:
            for img in images:
                if isinstance(img, (str, Path)):
                    contents.append(Image.open(str(img)).convert("RGB"))
                else:
                    contents.append(img)
        elif image is not None:
            if isinstance(image, (str, Path)):
                contents.append(Image.open(str(image)).convert("RGB"))
            else:
                contents.append(image)

        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        )
        raw = self._call_with_retry(contents, config)
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(
        self, contents: list, config: "types.GenerateContentConfig"
    ) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                text = response.text
                if text:
                    return text
                raise ValueError("Gemini returned empty response text")
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
        raise RuntimeError(
            f"Gemini API failed after {self.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """Extract and parse JSON from a response that may contain markdown fences."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)
