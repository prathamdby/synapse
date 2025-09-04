"""LangChain wrapper for Cerebras AI integration."""

import logging
import re
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
    """Chat interface for Cerebras with conversation management and MCP integration."""

    def __init__(self, api_key: str = None, mcp_manager=None):
        self.cerebras_client = CerebrasClient(api_key=api_key)
        self.llm = CerebrasLLM(api_key=api_key)
        self.mcp_manager = mcp_manager

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
            # Check if MCP tools should be used
            if self.mcp_manager and self.should_use_tools(message):
                return await self.chat_with_mcp(
                    message, conversation_history, system_prompt, user_context, model
                )

            # Standard chat without MCP
            return await self._standard_chat(
                message, conversation_history, system_prompt, user_context, model
            )

        except Exception as e:
            logger.error(f"Error in chat_with_history: {e}")
            return "🚫 <b>Processing Error</b>\n\nI encountered an issue while processing your message. This could be due to:\n• Temporary AI service disruption\n• Message too complex or long\n• Network connectivity issues\n\nPlease try:\n• Sending a shorter message\n• Rephrasing your question\n• Trying again in a minute\n\nIf the problem persists, consider using /reset to clear conversation history."

    async def _standard_chat(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]],
        system_prompt: str = None,
        user_context: Dict[str, Any] = None,
        model: str = None,
    ) -> str:
        """Standard chat without MCP tools."""
        # Add the new user message to history
        updated_history = conversation_history + [{"role": "user", "content": message}]

        # Format for Cerebras API
        messages = self.cerebras_client.format_conversation_for_cerebras(
            updated_history, system_prompt, user_context
        )

        # Generate response with specified model
        response = await self.cerebras_client.generate_response(messages, model=model)

        return response

    async def chat_with_mcp(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]],
        system_prompt: str = None,
        user_context: Dict[str, Any] = None,
        model: str = None,
    ) -> str:
        """Chat with MCP tool integration."""
        try:
            # Check if any MCP servers are connected
            if not self.mcp_manager.is_any_server_connected():
                logger.info("No MCP servers connected, falling back to standard chat")
                return await self._standard_chat(
                    message, conversation_history, system_prompt, user_context, model
                )

            # Execute tools and get results
            tool_results = await self.execute_tools_and_respond(
                message, conversation_history
            )

            if tool_results:
                # Integrate tool results into the response
                enhanced_message = f"{message}\n\n[Tool Results]\n{tool_results}"

                # Add the enhanced message to history
                updated_history = conversation_history + [
                    {"role": "user", "content": enhanced_message}
                ]

                # Format for Cerebras API with tool context
                messages = self.cerebras_client.format_conversation_for_cerebras(
                    updated_history, system_prompt, user_context
                )

                # Generate response with tool results integrated
                response = await self.cerebras_client.generate_response(
                    messages, model=model
                )

                return response
            else:
                # No tool results, fall back to standard chat
                return await self._standard_chat(
                    message, conversation_history, system_prompt, user_context, model
                )

        except Exception as e:
            logger.error(f"Error in MCP chat: {e}")
            # Fall back to standard chat on MCP error
            return await self._standard_chat(
                message, conversation_history, system_prompt, user_context, model
            )

    def should_use_tools(self, message: str) -> bool:
        """
        Determine if MCP tools should be used for this message.

        Args:
            message: User message to analyze

        Returns:
            True if tools should be used, False otherwise
        """
        if not self.mcp_manager or not self.mcp_manager.is_any_server_connected():
            return False

        # Keywords that suggest tool usage might be helpful
        tool_keywords = [
            # Web/data fetching
            "fetch",
            "get",
            "download",
            "retrieve",
            "load",
            "url",
            "website",
            "web",
            "http",
            "api",
            # File operations
            "file",
            "read",
            "write",
            "save",
            "delete",
            "create",
            "folder",
            "directory",
            "path",
            # Git operations
            "git",
            "commit",
            "branch",
            "repository",
            "repo",
            "clone",
            "pull",
            "push",
            # General tool indicators
            "search",
            "find",
            "lookup",
            "check",
            "verify",
            "analyze",
            "process",
            "execute",
            "run",
            "command",
            "script",
            "tool",
            "function",
        ]

        # Convert message to lowercase for matching
        message_lower = message.lower()

        # Check for tool-related keywords
        for keyword in tool_keywords:
            if keyword in message_lower:
                logger.debug(f"Tool keyword detected: {keyword}")
                return True

        # Check for URLs (might need web fetching)
        if re.search(r"https?://\S+", message):
            logger.debug("URL detected in message")
            return True

        # Check for file paths
        if re.search(r"[/\\][\w\-_.]+", message):
            logger.debug("File path detected in message")
            return True

        # Check if message is asking for external information
        question_patterns = [
            r"\bwhat\s+is\s+the\s+latest\b",
            r"\bget\s+me\s+information\b",
            r"\bfind\s+out\s+about\b",
            r"\bcheck\s+if\b",
            r"\blook\s+up\b",
        ]

        for pattern in question_patterns:
            if re.search(pattern, message_lower):
                logger.debug(f"Question pattern detected: {pattern}")
                return True

        return False

    async def execute_tools_and_respond(
        self, message: str, history: List[Dict]
    ) -> Optional[str]:
        """
        Execute appropriate MCP tools based on the message content.

        Args:
            message: User message
            history: Conversation history

        Returns:
            Tool results as formatted string or None if no tools executed
        """
        try:
            # Get available tools
            available_tools = self.mcp_manager.get_available_tools()
            if not available_tools:
                logger.debug("No MCP tools available")
                return None

            # Simple tool selection based on message content
            selected_tools = self._select_tools_for_message(message, available_tools)

            if not selected_tools:
                logger.debug("No suitable tools found for message")
                return None

            # Execute selected tools
            tool_results = []
            for tool_name, arguments in selected_tools:
                logger.info(f"Executing MCP tool: {tool_name} with args: {arguments}")

                result = await self.mcp_manager.execute_tool(tool_name, arguments)

                if result.success:
                    tool_results.append(f"**{tool_name}**: {result.result}")
                else:
                    logger.warning(
                        f"Tool execution failed: {tool_name} - {result.error}"
                    )
                    tool_results.append(f"**{tool_name}** (failed): {result.error}")

            if tool_results:
                return "\n".join(tool_results)

            return None

        except Exception as e:
            logger.error(f"Error executing tools: {e}")
            return None

    def _select_tools_for_message(self, message: str, available_tools) -> List[tuple]:
        """
        Select appropriate tools for a message.

        Args:
            message: User message
            available_tools: List of available MCP tools

        Returns:
            List of (tool_name, arguments) tuples
        """
        selected_tools = []
        message_lower = message.lower()

        for tool in available_tools:
            # Simple heuristic-based tool selection
            tool_name = tool.name.lower()

            # Web fetching tools
            if "fetch" in tool_name or "get" in tool_name:
                # Look for URLs in the message
                urls = re.findall(r"https?://\S+", message)
                if urls:
                    for url in urls[:3]:  # Limit to 3 URLs
                        selected_tools.append((tool.name, {"url": url}))
                elif any(
                    word in message_lower
                    for word in ["website", "web", "url", "fetch", "download"]
                ):
                    # Generic web request
                    selected_tools.append((tool.name, {"query": message}))

            # File system tools
            elif "file" in tool_name or "read" in tool_name:
                # Look for file paths
                file_paths = re.findall(r"[/\\][\w\-_.]+", message)
                if file_paths:
                    for path in file_paths[:3]:  # Limit to 3 files
                        if "read" in tool_name:
                            selected_tools.append((tool.name, {"path": path}))
                        else:
                            selected_tools.append((tool.name, {"file_path": path}))

            # Git tools
            elif "git" in tool_name:
                if any(
                    word in message_lower
                    for word in ["git", "repository", "repo", "commit", "branch"]
                ):
                    selected_tools.append((tool.name, {"action": "status"}))

            # Search tools
            elif "search" in tool_name:
                if any(word in message_lower for word in ["search", "find", "lookup"]):
                    selected_tools.append((tool.name, {"query": message}))

        # Limit total tools to avoid overwhelming the response
        return selected_tools[:2]
