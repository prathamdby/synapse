"""LangChain wrapper for Cerebras AI integration."""

import logging
from typing import Any, List, Mapping, Optional, Dict
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from pydantic import Field
from cerebras_client import CerebrasClient

logger = logging.getLogger(__name__)


class CerebrasLLM(LLM):
    """LangChain LLM wrapper for Cerebras AI."""

    model: str = "gpt-oss-120b"
    max_completion_tokens: int = 65536
    temperature: float = 0.7
    top_p: float = 1.0
    reasoning_effort: str = "medium"
    cerebras_client: Optional[CerebrasClient] = Field(default=None, exclude=True)

    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.cerebras_client = CerebrasClient(api_key=api_key)

    @property
    def _llm_type(self) -> str:
        """Return identifier of the LLM."""
        return "cerebras"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Cerebras API synchronously."""
        try:
            # Convert prompt to messages format
            messages = [{"role": "user", "content": prompt}]

            # This is a sync call, but we'll use the sync method from cerebras_client
            import asyncio

            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, but LangChain expects sync
                # Use run_until_complete with a new event loop in a thread
                import concurrent.futures
                import threading

                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.cerebras_client.generate_response(
                                messages,
                                model=self.model,
                                max_completion_tokens=self.max_completion_tokens,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                reasoning_effort=self.reasoning_effort,
                            )
                        )
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    return future.result()

            except RuntimeError:
                # No event loop running, we can use asyncio.run
                return asyncio.run(
                    self.cerebras_client.generate_response(
                        messages,
                        model=self.model,
                        max_completion_tokens=self.max_completion_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        reasoning_effort=self.reasoning_effort,
                    )
                )

        except Exception as e:
            logger.error(f"Error in CerebrasLLM._call: {e}")
            return f"Error generating response: {str(e)}"

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Cerebras API asynchronously."""
        try:
            messages = [{"role": "user", "content": prompt}]

            return await self.cerebras_client.generate_response(
                messages,
                model=self.model,
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                reasoning_effort=self.reasoning_effort,
            )

        except Exception as e:
            logger.error(f"Error in CerebrasLLM._acall: {e}")
            return f"Error generating response: {str(e)}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "reasoning_effort": self.reasoning_effort,
        }


class CerebrasChat:
    """Chat interface for Cerebras with conversation management."""

    def __init__(self, api_key: str = None):
        self.cerebras_client = CerebrasClient(api_key=api_key)
        self.llm = CerebrasLLM(api_key=api_key)

    async def get_available_models(self) -> List[str]:
        """Get list of available Cerebras models."""
        return await self.cerebras_client.get_available_models()

    async def chat_with_history(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]],
        system_prompt: str = None,
        user_context: Dict[str, Any] = None,
        model: str = None,
    ) -> str:
        """Chat with conversation history context."""
        try:
            # Add the new user message to history
            updated_history = conversation_history + [
                {"role": "user", "content": message}
            ]

            # Format for Cerebras API
            messages = self.cerebras_client.format_conversation_for_cerebras(
                updated_history, system_prompt, user_context
            )

            # Generate response with specified model
            response = await self.cerebras_client.generate_response(
                messages, model=model
            )

            return response

        except Exception as e:
            logger.error(f"Error in chat_with_history: {e}")
            return "I apologize, but I encountered an error while processing your message. Please try again."

    async def stream_response(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]],
        system_prompt: str = None,
    ):
        """Stream response for real-time updates."""
        try:
            updated_history = conversation_history + [
                {"role": "user", "content": message}
            ]

            messages = self.cerebras_client.format_conversation_for_cerebras(
                updated_history, system_prompt
            )

            # Note: Streaming implementation would go here
            # For now, return the full response
            response = await self.cerebras_client.generate_response(messages)
            yield response

        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield "I apologize, but I encountered an error while processing your message."
