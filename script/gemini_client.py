"""
Shared Gemini API wrapper for SAM3 track segmentation pipeline.

Provides a simple interface to Google's Gemini API for text and image+text prompts,
with retry logic and error handling.
"""

import json
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from PIL import Image


class GeminiClient:
    """Wrapper around the Google Generative AI (Gemini) API."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai is not installed. "
                "Install it with: pip install google-generativeai"
            )
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Send a text-only prompt to Gemini and return the response text."""
        config = GenerationConfig(temperature=temperature)
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
        config = GenerationConfig(temperature=temperature)
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
        config = GenerationConfig(temperature=temperature)
        return self._call_with_retry(loaded, config)

    def generate_json(
        self,
        prompt: str,
        image: Optional[Union[str, Path, Image.Image]] = None,
        images: Optional[List[Union[str, Path, Image.Image]]] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """Generate and parse a JSON response from Gemini."""
        if images:
            raw = self.generate_with_images(prompt, images, temperature)
        elif image is not None:
            raw = self.generate_with_image(prompt, image, temperature)
        else:
            raw = self.generate(prompt, temperature)
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, contents: list, config: "GenerationConfig") -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.model.generate_content(
                    contents, generation_config=config
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
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)
