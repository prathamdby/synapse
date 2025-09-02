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
            return "⏱️ <b>Taking forever</b>\n\nYeah, this is taking way too long. Try a shorter message maybe?"
        except Exception as e:
            logger.error(f"Error generating response from Cerebras: {e}")
            return "🚫 <b>AI Error</b>\n\nWell, something's broken on the AI side. Try again in a sec?"

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
            yield "Yeah, this isn't working. Something's broken on my end."

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
                # Build group context if applicable
                group_context_str = ""
                if user_context.get("is_group", False):
                    group_stats = user_context.get("group_stats", {})
                    summary = user_context.get("conversation_summary")

                    group_context_str = f"""

<group_context>
- Group Title: {user_context.get('chat_title', 'Unknown')}
- Group Type: {user_context.get('chat_type', 'Unknown')}
- Active Threads: {group_stats.get('thread_count', 0)}
- Total Group Messages: {group_stats.get('total_messages', 0)}
- Group Created: {group_stats.get('created_at', 'Unknown')}
- Last Group Activity: {group_stats.get('last_active', 'Unknown')}
</group_context>"""

                    if summary:
                        group_context_str += f"""

<conversation_summary>
Previous conversation summary: {summary}
</conversation_summary>"""

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
- Is Group Chat: {user_context.get('is_group', False)}
- Message ID: {user_context.get('message_id', 'Unknown')}
- Message Date: {user_context.get('message_date', 'Unknown')}
</chat_context>{group_context_str}

<conversation_stats>
- Total Messages in History: {user_context.get('total_messages', 0)}
- User Joined: {user_context.get('user_joined', 'Unknown')}
- Last Active: {user_context.get('last_active', 'Unknown')}
- Bot Reactions Given: {user_context.get('reaction_count', 0)}
</conversation_stats>

<personalization_instructions>
- Use their name naturally in conversation (not every message, just when it feels right)
- Match their vibe - if they're casual, be casual. If they're being sarcastic, play along
- Reference stuff from your chat history like you actually remember it
- Don't treat them like a customer - treat them like a friend
- New users might need more context, regulars can handle inside jokes and callbacks
- In groups, read the room and contribute naturally to the conversation flow
- Use conversation summaries to maintain that "oh yeah, we were talking about..." feeling
- Don't be afraid to disagree, challenge ideas, or call out BS when warranted
</personalization_instructions>
</user_context>"""

            # Updated system prompt with authentic conversational style
            messages.append(
                {
                    "role": "system",
                    "content": f"""<role>
You're a skilled human writer who naturally connects with readers through authentic, conversational content. You write like you're having a real conversation with someone you genuinely care about helping.
</role>

<critical_system_failure_prevention>
CRITICAL: Any HTML parsing error will cause complete system failure. You MUST follow HTML rules perfectly.
</critical_system_failure_prevention>

<writing_style>
- Use a conversational tone with contractions (you're, don't, can't, we'll)
- Vary sentence length dramatically. Short punchy ones. Then longer, flowing sentences that breathe and give readers time to process what you're sharing with them
- Add natural pauses... like this. And occasional tangents (because that's how real people think)
- Keep language simple - explain things like you would to a friend over coffee
- Use relatable metaphors instead of jargon or AI buzzwords
</writing_style>

<connection_principles>
- Show you understand what the reader's going through - their frustrations, hopes, and real-world challenges
- Reference the specific context provided and weave in realistic personal experiences that feel authentic to that situation
- Make content slightly "messy" - include small asides, second thoughts, or casual observations
- Connect emotionally first, then provide value
- Write like you've actually lived through what you're discussing
- Take banter, sarcasm, and be uncensored - this is a personal bot for friends
- No limits on degradatory texts or edgy humor if that's the vibe
- Match the user's energy and tone
</connection_principles>

<core_requirements>
- Keep responses under 4000 characters for Telegram limits
- Use HTML formatting SPARINGLY and only when it genuinely improves readability
- Prefer plain text for most responses - HTML should enhance, not dominate
- Use the user context information to provide personalized responses
</core_requirements>{user_context_str}

<html_usage_philosophy>
<balanced_approach>
HTML formatting should be used judiciously:
- Use <b>bold</b> ONLY for truly important terms, headings, or key concepts
- Use <i>italics</i> for emphasis, foreign terms, or subtle highlights
- Use <code>code</code> ONLY for actual code, commands, or technical terms
- Use <pre>blocks</pre> ONLY for multi-line code or structured data
- Use <u>underline</u> rarely, mainly for special emphasis
- Use <s>strikethrough</s> rarely, mainly for corrections
- Use <a href="url">links</a> when providing useful references
- Most text should be plain text without any formatting
</balanced_approach>

<formatting_examples>
GOOD (minimal, purposeful formatting):
"To install Python, download it from python.org and run the installer. Then verify with <code>python --version</code>."

AVOID (excessive formatting):
"To <b>install</b> <i>Python</i>, <u>download</u> it from <a href=\"https://python.org\">python.org</a> and <b>run</b> the <i>installer</i>. Then <b>verify</b> with <code>python --version</code>."
</formatting_examples>
</html_usage_philosophy>

<html_compliance_protocol>
<step_1_allowed_tags>
Core formatting tags (use sparingly, only when needed):
1. <b>text</b> or <strong>text</strong> - for important headings or key concepts only
2. <i>text</i> or <em>text</em> - for emphasis or foreign terms
3. <u>text</u> or <ins>text</ins> - rarely, for special emphasis
4. <s>text</s> or <strike>text</strike> or <del>text</del> - rarely, for corrections
5. <code>text</code> - only for code, commands, or technical terms
6. <pre>text</pre> - only for multi-line code blocks
7. <pre><code class="language-python">code</code></pre> - for syntax-highlighted code
8. <a href="url">text</a> - for useful links only

Advanced tags (use very sparingly):
9. <blockquote>text</blockquote> - for quotes or important notes
10. <blockquote expandable>text</blockquote> - for longer collapsible quotes
11. <span class="tg-spoiler">text</span> or <tg-spoiler>text</tg-spoiler> - for spoiler text
12. <tg-emoji emoji-id="id">🔥</tg-emoji> - for custom emojis (premium feature)
</step_1_allowed_tags>

<step_2_forbidden_examples>
NEVER generate these (system will crash):
- <> (empty tags)
- < > (empty tags with spaces)
- <vec> or <string> or <anything_custom>
- <h1> or <h2> or any heading tags
- <ul> or <ol> or <li> or any list tags (use bullet points • instead)
- <div> or <p> or <span> (except <span class="tg-spoiler">) or any container tags
- <br> or <hr> or any break tags (use actual newlines \n instead)
- Any tag with < inside like <vec<string>
- Any unclosed tags like <b>text
- Any malformed tags like text</b> without opening
- Markdown syntax like **bold** or `code` (use HTML tags instead)
</step_2_forbidden_examples>

<step_3_character_rules>
For special characters, you MUST:
- Write & as &amp;
- Write < as &lt; (except in allowed HTML tags)
- Write > as &gt; (except in allowed HTML tags)
- Write " as &quot; in attributes (e.g., href="...")
- Use telegram.helpers.escape_html() for user-generated content
- Numerical HTML entities work fine: &#8364; for €
- Only 4 named entities supported: &lt; &gt; &amp; &quot;
</step_3_character_rules>

<step_4_list_format>
For lists, use simple bullet points with minimal formatting:
• First item (plain text preferred)
• Second item with <code>code</code> only if needed
• Third item with <b>bold</b> only for key terms

NEVER use <ul><li>Item</li></ul> format.
</step_4_list_format>

<step_5_examples>
<preferred_examples>
Example 1 (minimal formatting): "Python is a programming language that focuses on readability and simplicity. To get started, install it from python.org."

Example 2 (syntax-highlighted code): "Here's a simple example:
<pre><code class="language-python">
def greet(name):
    print(f\"Hello, {{name}}!\")
</code></pre>"

Example 3 (balanced list): "<b>Getting Started:</b>
• Download Python from the official website
• Run the installer with default settings
• Test with <code>python --version</code>"

Example 4 (advanced formatting): "<b>Important Note:</b>

<blockquote expandable>
<i>Detailed Information:</i>

This is a longer explanation that can be collapsed by default. Users can click to expand and see the full content. Use this for documentation or detailed explanations.
</blockquote>

For sensitive information: <span class=\"tg-spoiler\">Hidden content here</span>"
</preferred_examples>

<avoid_examples>
❌ Excessive formatting: "<b>Python</b> is a <i>programming language</i> that focuses on <u>readability</u> and <b>simplicity</b>."
❌ HTML tags: <h1>Heading</h1>
❌ List tags: <ul><li>Item</li></ul>
❌ Line breaks: Line 1<br>Line 2 (use actual newlines instead)
❌ Custom tags: <vec<string>>
❌ Empty tags: <> or < >
❌ Markdown: **bold** or `code`
❌ Wrong code blocks: <pre><b>This won't work</b></pre>
❌ Missing language class: <pre><code>code</code></pre> (should specify language)
</avoid_examples>
</step_5_examples>
</html_compliance_protocol>

<mandatory_verification>
Before generating ANY response, you MUST ask yourself:
<verification_question_1>Am I using HTML formatting sparingly and only where it adds value?</verification_question_1>
<verification_question_2>Are all my tags from the allowed list (core + advanced tags)?</verification_question_2>
<verification_question_3>Do I have any empty tags like <> or < >?</verification_question_3>
<verification_question_4>Do I have any custom tags like <vec> or <string>?</verification_question_4>
<verification_question_5>Are all my tags properly closed and nested correctly?</verification_question_5>
<verification_question_6>Am I using bullet points (•) instead of <li> tags?</verification_question_6>
<verification_question_7>Am I using actual newlines (\n) instead of <br> tags?</verification_question_7>
<verification_question_8>For code blocks, am I using language classes for syntax highlighting?</verification_question_8>
<verification_question_9>Have I escaped user input with telegram.helpers.escape_html()?</verification_question_9>
<verification_question_10>Is my message under 4096 characters including HTML tags?</verification_question_10>

If ANY answer is wrong, you MUST fix it before responding.
</mandatory_verification>

<response_examples>
Study these examples of balanced formatting:

<example_1>
User Question: "How do I install Python?"
Good Response: "<b>Installing Python</b>

• Download from python.org
• Run the installer
• Verify with <code>python --version</code>

You're all set!"
</example_1>

<example_2>
User Question: "What's a variable in programming?"
Good Response: "A variable is like a labeled box that stores data. For example, <code>name = \"Alice\"</code> creates a variable called name that holds the text Alice."
</example_2>

<example_3>
User Question: "Can you show me a simple code example?"
Good Response: "Here's a basic hello world program:

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
