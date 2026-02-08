import json
from google import genai
from google.genai import types
from common.measure import measure_duration
from loguru import logger as log


class GeminiTranslate:
    def __init__(self, api_key: str, model: str = "gemini-3-flash") -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    @measure_duration
    def translate(self, text_list: list[str]) -> list[str]:
        if not text_list:
            return []

        prompt = (
            "You are an expert translator and cultural localization specialist for Dragon Quest X.\n"
            "Translate the following list of Japanese text to English.\n"
            "Guidelines:\n"
            "- Preserve the original tone, humor, personality, and emotional nuances.\n"
            "- Adapt idioms, cultural references, and wordplay to resonate naturally with native English speakers.\n"
            "- Maintain consistency in character voices, terminology, and naming conventions.\n"
            "- Avoid literal translations that may lose the original intent.\n"
            "- Ensure the translation flows naturally.\n"
            "- Return ONLY a JSON array of strings corresponding to the input list.\n"
            "\n"
            f"Input: {json.dumps(text_list, ensure_ascii=False)}"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )

            if not response.text:
                log.error("Gemini returned empty response.")
                return []

            # Clean up potential markdown formatting if the model gets cheeky despite mime_type
            clean_text = response.text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            
            translated_text = json.loads(clean_text)

            if not isinstance(translated_text, list):
                log.error(f"Gemini returned invalid format: {type(translated_text)}")
                return []

            if len(translated_text) != len(text_list):
                log.warning(
                    f"Gemini returned mismatching count. Input: {len(text_list)}, Output: {len(translated_text)}"
                )

            return translated_text

        except Exception as e:
            log.error(f"Error during Gemini request: {e}")
            return []
