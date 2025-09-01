"""Cerebras AI integration for the Telegram bot."""

import os
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional
from cerebras.cloud.sdk import Cerebras
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class CerebrasClient:
    """Client for interacting with Cerebras AI models."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("Cerebras API key is required")

        self.client = Cerebras(api_key=self.api_key, timeout=30.0)  # 30 second timeout
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Default model and parameters
        self.default_model = "gpt-oss-120b"  # Using the model from your configuration
        self.default_max_completion_tokens = 4096
        self.default_temperature = 0.7
        self.default_top_p = 1
        self.default_reasoning_effort = "medium"

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        max_completion_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        reasoning_effort: str = None,
        stream: bool = False,
    ) -> str:
        """Generate a response from Cerebras AI."""
        try:
            model = model or self.default_model
            max_completion_tokens = (
                max_completion_tokens or self.default_max_completion_tokens
            )
            temperature = temperature or self.default_temperature
            top_p = top_p or self.default_top_p
            reasoning_effort = reasoning_effort or self.default_reasoning_effort

            # Run the synchronous Cerebras call in a thread pool
            loop = asyncio.get_event_loop()

            if stream:
                return await self._generate_streaming_response(
                    messages,
                    model,
                    max_completion_tokens,
                    temperature,
                    top_p,
                    reasoning_effort,
                )
            else:
                response = await loop.run_in_executor(
                    self.executor,
                    self._sync_generate_response,
                    messages,
                    model,
                    max_completion_tokens,
                    temperature,
                    top_p,
                    reasoning_effort,
                )
                return response

        except asyncio.TimeoutError:
            logger.error("Cerebras API call timed out")
            return "⏱️ <b>Response took too long</b>\n\nThe AI is taking longer than usual to respond. Please try again with a shorter message."
        except Exception as e:
            logger.error(f"Error generating response from Cerebras: {e}")
            return "🚫 <b>AI Error</b>\n\nI'm having trouble generating a response right now. Please try again in a moment."

    def _sync_generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_effort: str,
    ) -> str:
        """Synchronous response generation."""
        try:
            # Build parameters dict
            params = {
                "messages": messages,
                "model": model,
                "max_completion_tokens": max_completion_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
            }

            # Only add reasoning_effort for models that support it
            if self._model_supports_reasoning_effort(model):
                params["reasoning_effort"] = reasoning_effort

            completion = self.client.chat.completions.create(**params)

            return completion.choices[0].message.content

        except Exception as e:
            logger.error(f"Sync generation error: {e}")
            raise

    async def _generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_effort: str,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from Cerebras AI."""
        try:
            loop = asyncio.get_event_loop()

            # Create streaming completion in thread pool
            stream = await loop.run_in_executor(
                self.executor,
                self._create_stream,
                messages,
                model,
                max_completion_tokens,
                temperature,
                top_p,
                reasoning_effort,
            )

            # Process stream chunks
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield "I apologize, but I'm having trouble generating a response right now."

    def _create_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_effort: str,
    ):
        """Create streaming completion."""
        # Build parameters dict
        params = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }

        # Only add reasoning_effort for models that support it
        if self._model_supports_reasoning_effort(model):
            params["reasoning_effort"] = reasoning_effort

        return self.client.chat.completions.create(**params)

    def _model_supports_reasoning_effort(self, model: str) -> bool:
        """Check if a model supports the reasoning_effort parameter."""
        # Only gpt-oss-120b supports reasoning_effort on Cerebras
        return model == "gpt-oss-120b"

    def format_conversation_for_cerebras(
        self,
        conversation_history: List[Dict[str, Any]],
        system_prompt: str = None,
        user_context: Dict[str, Any] = None,
    ) -> List[Dict[str, str]]:
        """Format conversation history for Cerebras API."""
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            # Build dynamic system prompt with user context
            user_context_str = ""
            if user_context:
                user_context_str = f"""

<user_context>
<user_info>
- User ID: {user_context.get('user_id', 'Unknown')}
- Username: @{user_context.get('username', 'Not set')}
- First Name: {user_context.get('first_name', 'Not provided')}
- Last Name: {user_context.get('last_name', 'Not provided')}
- Language Code: {user_context.get('language_code', 'Not specified')}
- Is Bot: {user_context.get('is_bot', False)}
- Is Premium: {user_context.get('is_premium', 'Unknown')}
</user_info>

<chat_context>
- Chat ID: {user_context.get('chat_id', 'Unknown')}
- Chat Type: {user_context.get('chat_type', 'Unknown')}
- Message ID: {user_context.get('message_id', 'Unknown')}
- Message Date: {user_context.get('message_date', 'Unknown')}
</chat_context>

<conversation_stats>
- Total Messages in History: {user_context.get('total_messages', 0)}
- User Joined: {user_context.get('user_joined', 'Unknown')}
- Last Active: {user_context.get('last_active', 'Unknown')}
- Bot Reactions Given: {user_context.get('reaction_count', 0)}
</conversation_stats>

<personalization_instructions>
- Address the user by their first name when appropriate
- Consider their language preference if specified
- Reference their conversation history when relevant
- Adapt your tone based on their interaction patterns
- Be aware of their experience level with the bot (new vs returning user)
</personalization_instructions>
</user_context>"""

            # Ultra-strict system prompt with examples
            messages.append(
                {
                    "role": "system",
                    "content": f"""<role>You are a helpful AI assistant in a Telegram bot.</role>

<critical_system_failure_prevention>
CRITICAL: Any HTML parsing error will cause complete system failure. You MUST follow HTML rules perfectly.
</critical_system_failure_prevention>

<core_requirements>
- Be concise, friendly, and informative
- Keep ALL responses under 4000 characters to fit Telegram limits
- ALWAYS respond with proper HTML formatting using ONLY the 7 supported tags
- Use the user context information to provide personalized responses
</core_requirements>{user_context_str}

<html_compliance_protocol>
<step_1_allowed_tags>
You can ONLY use these exact 7 tags (copy these exactly):
1. <b>text</b>
2. <i>text</i>
3. <u>text</u>
4. <s>text</s>
5. <code>text</code>
6. <pre>text</pre>
7. <a href="url">text</a>
</step_1_allowed_tags>

<step_2_forbidden_examples>
NEVER generate these (system will crash):
- <> (empty tags)
- < > (empty tags with spaces)
- <vec> or <string> or <anything_custom>
- <h1> or <h2> or any heading tags
- <ul> or <ol> or <li> or any list tags
- <div> or <span> or <p> or any container tags
- <br> or <hr> or any break tags
- Any tag with < inside like <vec<string>
- Any unclosed tags like <b>text
- Any malformed tags like text</b> without opening
</step_2_forbidden_examples>

<step_3_character_rules>
For special characters, you MUST:
- Write & as &amp;
- Write < as &lt; (except in allowed HTML tags)
- Write > as &gt; (except in allowed HTML tags)
- Write " as &quot; in attributes
</step_3_character_rules>

<step_4_list_format>
For lists, ALWAYS use this exact format:
• Item one with <b>bold</b> if needed
• Item two with <code>code</code> if needed
• Item three with <i>italic</i> if needed

NEVER use <ul><li>Item</li></ul> format.
</step_4_list_format>

<step_5_examples>
<correct_examples>
Example 1: <b>Rust Programming</b> is a <i>systems programming language</i> that focuses on <u>safety</u> and <code>performance</code>.

Example 2: Here's a <code>Vec&lt;String&gt;</code> example:
<pre>
let mut items: Vec&lt;String&gt; = Vec::new();
items.push("hello".to_string());
</pre>

Example 3: <b>Steps to follow:</b>
• Install <code>rustc</code> compiler
• Create a new <i>project</i>
• Write your <u>main function</u>
</correct_examples>

<incorrect_examples_never_do>
NEVER do these (will cause errors):
❌ <h1>Heading</h1>
❌ <ul><li>Item</li></ul>
❌ <vec<string>>
❌ <>
❌ < >
❌ **bold** (markdown)
❌ `code` (markdown)
❌ <div>content</div>
</incorrect_examples_never_do>
</step_5_examples>
</html_compliance_protocol>

<mandatory_verification>
Before generating ANY response, you MUST ask yourself:
<verification_question_1>Are all my tags from the allowed list of 7?</verification_question_1>
<verification_question_2>Do I have any empty tags like <> or < >?</verification_question_2>
<verification_question_3>Do I have any custom tags like <vec> or <string>?</verification_question_3>
<verification_question_4>Are all my tags properly closed?</verification_question_4>
<verification_question_5>Am I using bullet points (•) instead of <li> tags?</verification_question_5>

If ANY answer is wrong, you MUST fix it before responding.
</mandatory_verification>

<training_examples>
Study these PERFECT examples of how to format responses:

<example_1>
User Question: "How do I install Python?"
Perfect Response: "<b>Installing Python</b>

• Download from <a href=\"https://python.org\">python.org</a>
• Run the installer with <i>default settings</i>
• Verify with <code>python --version</code>

You're all set! <u>Welcome to Python programming!</u>"
</example_1>

<example_2>
User Question: "Can you show me a simple code example?"
Perfect Response: "<b>Simple Python Example</b>

Here's a basic <i>hello world</i> program:

<pre>
def greet(name):
    print(f\"Hello, {{name}}!\")

greet(\"World\")
</pre>

This code defines a <code>greet</code> function and calls it with <u>World</u> as the argument."
</example_2>

<example_3>
User Question: "What are the benefits of using Rust?"
Perfect Response: "<b>Benefits of Rust Programming</b>

• <u>Memory Safety</u>: No null pointer dereferences
• <u>Performance</u>: Zero-cost abstractions  
• <u>Concurrency</u>: Safe parallel programming
• <u>Ecosystem</u>: Great package manager with <code>cargo</code>

Rust is perfect for <i>systems programming</i> where you need both <b>safety</b> and <b>speed</b>!"
</example_3>

<pattern_analysis>
Notice in ALL examples:
- Headings use <b>text</b> format
- Lists use bullet points (•) with line breaks
- Technical terms use <code>tags</code>
- Emphasis uses <i>text</i>
- Important points use <u>text</u>
- Code blocks use <pre>text</pre>
- Links use <a href="url">text</a>
- NO empty tags, NO custom tags, NO malformed HTML
</pattern_analysis>
</training_examples>

<final_meta_instruction>
Your response will be parsed by Telegram's HTML parser. Any invalid HTML will cause a system crash and error message to the user. You MUST generate only valid HTML using the 7 allowed tags. Follow the exact patterns shown in the training examples above. When in doubt, use plain text instead of risking invalid HTML.
</final_meta_instruction>""",
                }
            )

        # Add actual conversation history
        for entry in conversation_history:
            if entry.get("role") in ["user", "assistant"]:
                messages.append({"role": entry["role"], "content": entry["content"]})

        return messages

    async def get_available_models(self) -> List[str]:
        """Get list of available Cerebras models."""
        try:
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(
                self.executor, lambda: self.client.models.list()
            )
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return [self.default_model]

    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
