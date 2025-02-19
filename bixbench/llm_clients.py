from PIL.Image import Image
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from dotenv import load_dotenv
load_dotenv()

class UnanswerableError(Exception):
    """An exception indicating the agent could not answer this question. Will be marked as unsure."""


class LLMClient:

    def __init__(
        self,
        model_name: str,
        temp: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.temp = temp
        self.client = None
        self._initiate_client()

    def _initiate_client(self) -> None:

        if self.model_name.startswith("gpt") | self.model_name.startswith("o1"):
            import openai
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        elif self.model_name.startswith("claude"):
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        elif self.model_name.startswith("mistral"):
            from mistralai import Mistral
            self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

        else:
            raise ValueError(f"{self.model_name} not supported yet!")

    def get_response(
        self,
        query: str | None = None,
        **kwargs,
    ) -> str:
        self.query = query

        if self.model_name.startswith("o1"):
            return self._get_o1_response()

        elif self.model_name.startswith("claude"):
            return self._get_anthropic_response(**kwargs)

        elif self.model_name.startswith(("gpt")):
            return self._get_gpt_response(
            )

        elif self.model_name.startswith("mistral"):
            return self._get_mistral_response()

        else:
            raise ValueError(f"{self.model_name} not supported yet!")

    @retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=20),
    retry=retry_if_exception(
            lambda exc: not isinstance(exc, UnanswerableError)
        ),
    )
    def _get_o1_response(
        self,
    ) -> list[str]:

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": self.query}],
            )
            response = completion.choices[0].message.content


        except Exception as e:
            print("Failed to get response because of", e)
            response = "failed"

        return response

    @retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=20),
    retry=retry_if_exception(
            lambda exc: not isinstance(exc, UnanswerableError)
        ),
    )
    def _get_mistral_response(self) -> list[str]:

        try:
            completion = self.client.chat.complete(
                model=self.model_name,
                temperature=self.temp,
                messages=[{"role": "user", "content": self.query}],
            )
            response = completion.choices[0].message.content
        except Exception as e:
            response = "failed"
            print("Failed to get response because of", e)

        return response

    @retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=20),
    retry=retry_if_exception(
            lambda exc: not isinstance(exc, UnanswerableError)
        ),
    )
    def _get_anthropic_response(
        self, **kwargs
    ) -> list[str]:

        full_prompt: str | list[dict] = [{"type": "text", "text": self.query}]
 
        try:
            message = self.client.messages.create(
                model=self.model_name,  # Or another appropriate model
                max_tokens=kwargs["max_tokens"] if "max_tokens" in kwargs else 2000,
                temperature=self.temp,
                system=kwargs["system_prompt"] if "system_prompt" in kwargs else "",
                messages=[{"role": "user", "content": full_prompt}],
            )
            response = message.content[0].text
      
            # time.sleep(2)
        except Exception:
            response = "failed"
            print("Failed to get response")

        return response

    @retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=20),
    retry=retry_if_exception(
            lambda exc: not isinstance(exc, UnanswerableError)
        ),
    )
    def _get_gpt_response(self, **kwargs) -> str:
        """
        Get response from GPT model with automatic retries for rate limits and API errors.
        
        Args:
            **kwargs: Additional arguments including optional system_prompt
            
        Returns:
            str: Model response or "failed" if all retries are exhausted
        """
        full_prompt: str | list[dict] = self.query
        msgs = [
            {"role": "user", "content": full_prompt},
            {
                "role": "system",
                "content": kwargs.get("system_prompt", ""),
            },
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temp,
                messages=msgs,
                stream=False,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Failed to generate content because of: {e}")
            return "failed"
