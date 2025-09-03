"""LangChain wrapper for Cerebras AI integration."""

import logging
from typing import Any, List, Mapping, Optional, Dict
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import Tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pydantic import Field
from cerebras_client import CerebrasClient
from searxng_client import SearxngClient

logger = logging.getLogger(__name__)


class CerebrasLLM(LLM):
    """LangChain LLM wrapper for Cerebras AI."""

    model: str = "gpt-oss-120b"
    max_completion_tokens: int = 65536
    temperature: float = 0.7
    top_p: float = 1.0
    reasoning_effort: str = "medium"
    cerebras_client: Optional[CerebrasClient] = Field(default=None, exclude=True)

    def __init__(
        self, api_key: str = None, searxng_client: SearxngClient = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.cerebras_client = CerebrasClient(
            api_key=api_key, searxng_client=searxng_client
        )

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

    def __init__(self, api_key: str = None, searxng_client: SearxngClient = None):
        self.cerebras_client = CerebrasClient(
            api_key=api_key, searxng_client=searxng_client
        )
        self.llm = CerebrasLLM(api_key=api_key, searxng_client=searxng_client)
        self.searxng_client = searxng_client
        self.search_tool = self._create_search_tool() if searxng_client else None

    def _create_search_tool(self) -> Optional[Tool]:
        """Create search tool if SearXNG client is available."""
        if not self.searxng_client:
            return None

        async def search_function(query: str) -> str:
            """Search the web using SearXNG."""
            try:
                results = await self.searxng_client.search_with_summary(
                    query=query, language="en", max_results=5
                )

                if not results or results["total_results"] == 0:
                    return "No results found for the search query."

                # Format results as a string
                formatted_results = f"Search results for '{results['query']}':\n\n"
                for i, result in enumerate(results["results"], 1):
                    formatted_results += f"{i}. {result['title']}\n"
                    formatted_results += f"   {result['content']}\n"
                    formatted_results += f"   Source: {result['url']}\n\n"

                return formatted_results
            except Exception as e:
                logger.error(f"Error during search: {e}")
                return f"Error performing search: {str(e)}"

        def sync_search_function(query: str) -> str:
            """Synchronous wrapper for search function."""
            import asyncio

            try:
                # Try to get the current event loop
                loop = asyncio.get_event_loop()
                # If we're in a running loop, create a task
                if loop.is_running():
                    # Create a new event loop in a separate thread
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, search_function(query))
                        return future.result()
                else:
                    # No event loop running, use asyncio.run
                    return asyncio.run(search_function(query))
            except RuntimeError:
                # No event loop, use asyncio.run
                return asyncio.run(search_function(query))

        return Tool(
            name="web_search",
            description="Search the web for current information. Use this when you need to find up-to-date information. Input should be a search query.",
            func=sync_search_function,
        )

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
            return "🚫 <b>Processing Error</b>\n\nI encountered an issue while processing your message. This could be due to:\n• Temporary AI service disruption\n• Message too complex or long\n• Network connectivity issues\n\nPlease try:\n• Sending a shorter message\n• Rephrasing your question\n• Trying again in a minute\n\nIf the problem persists, consider using /reset to clear conversation history."

    async def chat_with_search(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]],
        system_prompt: str = None,
        user_context: Dict[str, Any] = None,
        model: str = None,
    ) -> str:
        """Chat with conversation history and search capability."""
        try:
            # If we have a search tool, first check if we need to search
            if self.search_tool:
                # Create a prompt to determine if we need to search
                search_decision_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a helpful AI assistant. Determine if the user's question requires current information that might not be in your training data. If so, respond with 'SEARCH_NEEDED: [search query]'. Otherwise, respond with 'NO_SEARCH_NEEDED'.",
                        ),
                        ("human", "{input}"),
                    ]
                )

                # Create chain to decide if search is needed
                search_decision_chain = (
                    search_decision_prompt | self.llm | StrOutputParser()
                )

                # Check if search is needed
                decision = search_decision_chain.invoke({"input": message})

                if decision.startswith("SEARCH_NEEDED:"):
                    search_query = decision.replace("SEARCH_NEEDED:", "").strip()
                    # Perform the search
                    search_results = self.search_tool.func(search_query)

                    # Create a prompt that includes the search results
                    search_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are a helpful AI assistant. Use the provided search results to answer the user's question accurately. If the search results don't contain relevant information, say so.",
                            ),
                            (
                                "human",
                                "Question: {question}\n\nSearch Results:\n{search_results}",
                            ),
                        ]
                    )

                    # Create chain with search results
                    search_chain = search_prompt | self.llm | StrOutputParser()

                    # Generate response using search results
                    response = search_chain.invoke(
                        {"question": message, "search_results": search_results}
                    )

                    return response
                else:
                    # No search needed, proceed with normal chat
                    return await self.chat_with_history(
                        message,
                        conversation_history,
                        system_prompt,
                        user_context,
                        model,
                    )
            else:
                # Fallback to regular chat if no search tool
                return await self.chat_with_history(
                    message, conversation_history, system_prompt, user_context, model
                )

        except Exception as e:
            logger.error(f"Error in chat_with_search: {e}")
            # Fallback to regular chat
            return await self.chat_with_history(
                message, conversation_history, system_prompt, user_context, model
            )

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
            yield "Well, that didn't work. Something went wrong on my end."
