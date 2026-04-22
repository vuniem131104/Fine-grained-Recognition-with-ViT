from __future__ import annotations

from typing import Any 
from typing import Optional
import httpx
from pydantic import BaseModel
from structlog import get_logger
from functools import cached_property

from .settings import LiteLLMSetting
from .models import LiteLLMInput
from .models import LiteLLMOutput
from .models import Messages
from .models import LiteLLMEmbeddingInput
from .models import LiteLLMEmbeddingOutput


logger = get_logger(__name__)

class LiteLLMService:
    def __init__(
        self,
        litellm_setting: LiteLLMSetting,
        async_client: Optional[httpx.AsyncClient] = None,
        client: Optional[httpx.Client] = None,
    ):
        self.litellm_setting = litellm_setting
        self.async_client = async_client
        self.client = client
            
    @cached_property
    def headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.litellm_setting.token.get_secret_value()}",
        }
    
    @property
    def _async_client(self) -> httpx.AsyncClient:
        if not self.async_client or self.async_client.is_closed:
            self.async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(150.0, connect=15.0),
                limits=httpx.Limits(max_connections=200, max_keepalive_connections=40),
            )
        return self.async_client
    
    @property
    def _client(self) -> httpx.Client:
        if not self.client or self.client.is_closed:
            self.client = httpx.Client(
                timeout=httpx.Timeout(150.0, connect=15.0),
                limits=httpx.Limits(max_connections=200, max_keepalive_connections=40),
            )
        return self.client
    
    def _embedding_llm(
        self, 
        inputs: LiteLLMEmbeddingInput,
    ) -> LiteLLMEmbeddingOutput:
        """ Process the input and return the output.

        Args:
            inputs (LiteLLMEmbeddingInput): The inputs to process.

        Returns:
            LiteLLMEmbeddingOutput: The processed output.
        """
        
        payload = {
            "model": self.litellm_setting.embedding_model,
            "input": inputs.text,
            "output_dimensionality": self.litellm_setting.dimension,
        }
        
        try:
            response = self._client.post(
                url=str(self.litellm_setting.url) + "v1/embeddings",
                headers=self.headers,
                json=payload,
            )
            
            if response.status_code == 200:
                return LiteLLMEmbeddingOutput(
                    embedding=response.json()['data'][0]['embedding'],
                )
            else:
                logger.error(
                    "Request failed with status code",
                    extra={
                        "status_code": response.status_code,
                        "model": self.litellm_setting.embedding_model,
                        "inputs": inputs.text,
                    }
                )
                return LiteLLMEmbeddingOutput(
                    embedding=[],
                )
        except httpx.RequestError as e:
            logger.exception(
                "An error occurred while processing the request",
                extra={
                    "error": str(e),
                    "inputs": inputs.text,
                    "model": self.litellm_setting.embedding_model,
                }
            )
            return LiteLLMEmbeddingOutput(
                embedding=[],
            )
            
            
    async def _embedding_llm_async(
        self, 
        inputs: LiteLLMEmbeddingInput,
    ) -> LiteLLMEmbeddingOutput:
        """Asynchronously process the input and return the output.

        Args:
            inputs (LiteLLMEmbeddingInput): The inputs to process.

        Returns:
            LiteLLMEmbeddingOutput: The processed output.
        """
        
        payload = {
            "model": self.litellm_setting.embedding_model,
            "input": inputs.text,
            "output_dimensionality": self.litellm_setting.dimension,
        }
        
        try:
            response = await self._async_client.post(
                url=str(self.litellm_setting.url) + "v1/embeddings",
                headers=self.headers,
                json=payload,
            )
            
            if response.status_code == 200:
                return LiteLLMEmbeddingOutput(
                    embedding=response.json()['data'][0]['embedding'],
                )
            else:
                logger.error(
                    "Request failed with status code",
                    extra={
                        "status_code": response.status_code,
                        "model": self.litellm_setting.embedding_model,
                        "inputs": inputs.text,
                    }
                )
                return LiteLLMEmbeddingOutput(
                    embedding=[],
                )
        except httpx.RequestError as e:
            logger.exception(
                "An error occurred while processing the request",
                extra={
                    "error": str(e),
                    "inputs": inputs.text,
                    "model": self.litellm_setting.embedding_model,
                }
            )
            return LiteLLMEmbeddingOutput(
                embedding=[],
            )
    
    def _inference_llm(
        self, 
        messages: Messages,
        model: str,
        response_format: type[BaseModel] | None,
        temperature: float,
        top_p: float,
        n: int,
        frequency_penalty: float,
        max_completion_tokens: int,
        reasoning_effort: str,
    ) -> LiteLLMOutput:
        """ Process the input and return the output.

        Args:
            messages (Messages): The messages to process.
            model (str): The model to use for processing.
            response_format (type[BaseModel] | None): The response format.
            temperature (float): The temperature for the model.
            top_p (float): The top_p value for the model.
            n (int): The number of responses to generate.
            frequency_penalty (float): The frequency penalty for the model.
            max_completion_tokens (int): The maximum number of completion tokens.
            reasoning_effort (str): The reasoning effort level.
            
        Returns:
            LiteLLMOutput: The processed output.
        """

        payload = self._build_payload(
            messages=messages,
            model=model,
            response_format=response_format,
            frequency_penalty=frequency_penalty,
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )

        try:
            response = self._client.post(
                url=str(self.litellm_setting.url) + "v1/chat/completions",
                headers=self.headers,
                json=payload,
            )
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return LiteLLMOutput(
                    response=content if not response_format else response_format.model_validate_json(content),
                    completion_tokens=response.json()['usage']['completion_tokens'],
                )
            else:
                logger.error(f"Request failed with status code {response.status_code}: {response.text}")
                return LiteLLMOutput(
                    response="",
                    completion_tokens=0,
                )
        except httpx.RequestError as e:
            logger.exception(
                "An error occurred while processing the request",
                extra={
                    "error": str(e),
                }
            )
            return LiteLLMOutput(
                response="",
                completion_tokens=0,
            )
            
    async def _inference_llm_async(
        self, 
        messages: Messages,
        model: str,
        response_format: type[BaseModel] | None,
        temperature: float,
        top_p: float,
        n: int,
        frequency_penalty: float,
        max_completion_tokens: int,
        reasoning_effort: str | None = None,
    ) -> LiteLLMOutput:
        """Asynchronously process the input and return the output.

        Args:
            messages (Messages): The messages to process.
            model (str): The model to use for processing.
            response_format (type[BaseModel] | None): The response format.
            temperature (float): The temperature for the model.
            top_p (float): The top_p value for the model.
            n (int): The number of responses to generate.
            frequency_penalty (float): The frequency penalty for the model.
            max_completion_tokens (int): The maximum number of completion tokens.
            reasoning_effort (str): The reasoning effort level.
        
        Returns:
            LiteLLMOutput: The processed output.
        """

        payload = self._build_payload(
            messages=messages,
            model=model,
            response_format=response_format,
            frequency_penalty=frequency_penalty,
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )
        
        try:
            response = await self._async_client.post(
                url=str(self.litellm_setting.url) + "v1/chat/completions",
                headers=self.headers,
                json=payload,
            )
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return LiteLLMOutput(
                    response=content if not response_format else response_format.model_validate_json(content),
                    completion_tokens=response.json()['usage']['completion_tokens'],
                )
            else:
                logger.error(f"Request failed with status code {response.status_code}: {response.text}")
                return LiteLLMOutput(
                    response="",
                    completion_tokens=0,
                )
        except httpx.RequestError as e:
            logger.exception(
                "An error occurred while processing the request",
                extra={
                    "error": str(e),
                }
            )
            return LiteLLMOutput(
                response="",
                completion_tokens=0,
            )
    
    def process(
        self,
        inputs: LiteLLMInput
    ) -> LiteLLMOutput:
        """Process the input and return the output.

        Args:
            inputs (LiteLLMInput): The input to process.

        Returns:
            LiteLLMOutput: The processed output.
        """
        
        return self._inference_llm(
            messages=inputs.messages,
            model=inputs.model if inputs.model else self.litellm_setting.model,
            response_format=inputs.response_format if inputs.response_format else None,
            temperature=inputs.temperature if inputs.temperature else self.litellm_setting.temperature,
            top_p=inputs.top_p if inputs.top_p else self.litellm_setting.top_p,
            n=inputs.n if inputs.n else self.litellm_setting.n,
            frequency_penalty=inputs.frequency_penalty if inputs.frequency_penalty else self.litellm_setting.frequency_penalty,
            max_completion_tokens=inputs.max_completion_tokens if inputs.max_completion_tokens else self.litellm_setting.max_completion_tokens,
            reasoning_effort=inputs.reasoning_effort if inputs.reasoning_effort else None,
        )
    
    async def stream_async(
        self,
        inputs: LiteLLMInput,
    ):
        """Stream tokens from LiteLLM as an async generator of str chunks."""
        payload = self._build_payload(
            messages=inputs.messages,
            model=inputs.model if inputs.model else self.litellm_setting.model,
            response_format=None,
            frequency_penalty=inputs.frequency_penalty if inputs.frequency_penalty else self.litellm_setting.frequency_penalty,
            n=inputs.n if inputs.n else self.litellm_setting.n,
            temperature=inputs.temperature if inputs.temperature else self.litellm_setting.temperature,
            top_p=inputs.top_p if inputs.top_p else self.litellm_setting.top_p,
            max_completion_tokens=inputs.max_completion_tokens if inputs.max_completion_tokens else self.litellm_setting.max_completion_tokens,
            reasoning_effort=inputs.reasoning_effort if inputs.reasoning_effort else None,
        )
        payload["stream"] = True

        import json as _json
        async with self._async_client.stream(
            "POST",
            url=str(self.litellm_setting.url) + "v1/chat/completions",
            headers=self.headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = _json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content") or ""
                    if delta:
                        yield delta
                except Exception:
                    continue

    async def process_async(
        self,
        inputs: LiteLLMInput
    ) -> LiteLLMOutput:
        """Process the input and return the output.

        Args:
            inputs (LiteLLMInput): The input to process.

        Returns:
            LiteLLMOutput: The processed output.
        """
        
        return await self._inference_llm_async(
            messages=inputs.messages,
            model=inputs.model if inputs.model else self.litellm_setting.model,
            response_format=inputs.response_format if inputs.response_format else None,
            temperature=inputs.temperature if inputs.temperature else self.litellm_setting.temperature,
            top_p=inputs.top_p if inputs.top_p else self.litellm_setting.top_p,
            n=inputs.n if inputs.n else self.litellm_setting.n,
            frequency_penalty=inputs.frequency_penalty if inputs.frequency_penalty else self.litellm_setting.frequency_penalty,
            max_completion_tokens=inputs.max_completion_tokens if inputs.max_completion_tokens else self.litellm_setting.max_completion_tokens,
            reasoning_effort=inputs.reasoning_effort if inputs.reasoning_effort else None,
        )
        
    def process_embedding(
        self,
        inputs: LiteLLMEmbeddingInput
    ) -> LiteLLMEmbeddingOutput:
        """Process the input for embedding and return the output.

        Args:
            inputs (LiteLLMEmbeddingInput): The input to process.

        Returns:
            LiteLLMEmbeddingOutput: The processed output.
        """
        
        return self._embedding_llm(
            model=inputs.model,
            inputs=inputs.inputs,
        )
        
    async def process_embedding_async(
        self, 
        inputs: LiteLLMEmbeddingInput
    ) -> LiteLLMEmbeddingOutput:        
        """Asynchronously process the input for embedding and return the output.

        Args:
            inputs (LiteLLMEmbeddingInput): The input to process.

        Returns:
            LiteLLMEmbeddingOutput: The processed output.
        """
        
        return await self._embedding_llm_async(
            model=inputs.model,
            inputs=inputs.inputs,
        )
    

    def _build_payload(
        self,
        messages: Messages,
        model: str,
        response_format: type[BaseModel] | None,
        frequency_penalty: float,
        n: int,
        temperature: float,
        top_p: float,
        max_completion_tokens: int,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Build the payload for the LiteLLM API request.

        Args:
            messages (Messages): The messages to include in the request.
            model (str): The model to use for processing.
            response_format (type[BaseModel] | None): The response format.
            frequency_penalty (float): The frequency penalty for the model.
            n (int): The number of responses to generate.
            temperature (float): The temperature for the model.
            top_p (float): The top_p value for the model.
            max_completion_tokens (int): The maximum number of completion tokens.
            reasoning_effort (str | None): The reasoning effort level.

        Returns:
            dict[str, Any]: The payload for the LiteLLM API request.
        """
        
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "max_completion_tokens": max_completion_tokens,
            "frequency_penalty": frequency_penalty,
            "messages": self._build_messages(messages),
        }
        
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        else:
            if model.startswith("gemini"):
                payload["reasoning_effort"] = "disable"

        if response_format:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": {
                        **response_format.model_json_schema(),
                        "additionalProperties": False,
                    },
                    "strict": True
                }
            }
        
        return payload
    
    
    def _build_messages(
        self,
        messages: Messages,
    ) -> list[dict[str, Any]]:
        """Build the messages for the LiteLLM API request.

        Args:
            messages (Messages): The messages to include in the request.

        Returns:
            list[dict[str, Any]]: The messages for the LiteLLM API request.
        """
        
        built_messages: list[dict[str, Any]] = [] 
        
        for message in messages:
            built_message: dict[str, Any] = {}
            if message.image_url:
                if message.content:
                    built_message = {
                    "role": message.role.value,
                    "content": [
                        {
                            "type": "text",
                            "text": message.content
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": message.image_url
                            }
                        }
                    ]
                }
                else:
                    built_message = {
                        "role": message.role.value,
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": message.image_url
                                }
                            }
                        ]
                    }
            elif message.file_url:
                if message.content:
                    built_message = {
                        "role": message.role.value,
                        "content": [
                            {
                                "type": "text",
                                "text": message.content
                            },
                            {
                                "type": "file",
                                "file": {
                                    "file_data": message.file_url
                                }
                            }
                        ]
                    }
                else:
                    built_message = {
                        "role": message.role.value,
                        "content": [
                            {
                                "type": "file",
                                "file": {
                                    "file_data": message.file_url
                                }
                            }
                        ]
                    }
            else:
                built_message = {
                    "role": message.role.value,
                    "content": message.content,
                }
            
            built_messages.append(built_message)

        return built_messages