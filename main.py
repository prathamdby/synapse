"""
Telegram Bot with Cerebras AI integration.

A Telegram bot that uses Cerebras AI for conversations with memory that actually works.
No bullshit, just fast responses and persistent context.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import html

import structlog
from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyParameters,
    BotCommand,
    BotCommandScopeDefault,
    BotCommandScopeAllGroupChats,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

from database import DatabaseManager
from cerebras_client import CerebrasClient
from langchain_cerebras import CerebrasChat

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class TelegramBot:
    """Main Telegram bot class with Cerebras AI integration."""

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

        self.db = DatabaseManager(os.getenv("DATABASE_PATH", "./bot_database.db"))

        # Initialize Cerebras chat client
        self.cerebras_chat = CerebrasChat(api_key=os.getenv("CEREBRAS_API_KEY"))

        # Rate limiting configuration
        self.rate_limit_messages = int(
            os.getenv("RATE_LIMIT_MESSAGES_PER_MINUTE", "10")
        )
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

        # Group configuration
        self.enable_group_features = (
            os.getenv("ENABLE_GROUP_FEATURES", "true").lower() == "true"
        )
        self.default_group_mode = os.getenv("DEFAULT_GROUP_MODE", "shared")
        self.default_mention_policy = os.getenv(
            "DEFAULT_GROUP_MENTION_POLICY", "mention_only"
        )
        self.group_max_context = int(os.getenv("GROUP_MAX_CONTEXT", "40"))
        self.default_group_rate_limit = int(os.getenv("DEFAULT_GROUP_RATE_LIMIT", "50"))
        self.default_user_rate_limit_in_groups = int(
            os.getenv("DEFAULT_USER_RATE_LIMIT_IN_GROUPS", "10")
        )

        # Typing indicators tracking
        self.typing_tasks: Dict[int, asyncio.Task] = {}

    def _model_supports_reasoning(self, model: str) -> bool:
        """Check if a model supports reasoning features."""
        # Only gpt-oss-120b supports reasoning_effort on Cerebras
        return model == "gpt-oss-120b"

    def _sanitize_html_response(self, response: str) -> str:
        """Minimal sanitization to fix common HTML parsing issues."""
        # Remove <think> tags and their content
        think_end = response.find("</think>")
        if think_end != -1:
            response = response[think_end + len("</think>") :].strip()

        return response.strip()

    def _should_respond_in_group(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Determine if bot should respond in a group chat based on mention policy."""
        if update.effective_chat.type == "private":
            return True

        message_text = update.message.text or ""
        bot_username = context.bot.username

        # Check if message starts with @bot_username
        if message_text.startswith(f"@{bot_username}"):
            return True

        # Check if it's a reply to the bot
        if (
            update.message.reply_to_message
            and update.message.reply_to_message.from_user.id == context.bot.id
        ):
            return True

        return False

    async def _is_user_admin(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Check if user is an admin in the current chat."""
        if update.effective_chat.type == "private":
            return True

        try:
            chat_admins = await context.bot.get_chat_administrators(
                update.effective_chat.id
            )
            user_id = update.effective_user.id
            return any(admin.user.id == user_id for admin in chat_admins)
        except Exception as e:
            logger.warning(f"Could not check admin status: {e}")
            return False

    async def _register_bot_commands(self, application: Application):
        """Register bot commands in Telegram's menu system."""
        try:
            # Define commands for private chats
            private_commands = [
                BotCommand("start", "Welcome message and bot introduction"),
                BotCommand("help", "Show help and usage information"),
                BotCommand("reset", "Clear conversation history"),
                BotCommand("stats", "Show your usage statistics"),
                BotCommand("model", "Switch AI models"),
            ]

            # Define commands for group chats (includes all private commands plus group-specific ones)
            group_commands = [
                BotCommand("start", "Welcome message and bot introduction"),
                BotCommand("help", "Show help and usage information"),
                BotCommand("reset", "Clear conversation history"),
                BotCommand("stats", "Show your usage statistics"),
                BotCommand("model", "Switch AI models"),
                BotCommand("group_mode", "Change group memory mode (admin only)"),
                BotCommand("group_settings", "View group settings (admin only)"),
                BotCommand("group_reset", "Reset group conversations (admin only)"),
                BotCommand("group_stats", "Show group statistics"),
            ]

            # Register commands for private chats (default scope)
            await application.bot.set_my_commands(
                commands=private_commands, scope=BotCommandScopeDefault()
            )

            # Register commands for all group chats
            await application.bot.set_my_commands(
                commands=group_commands, scope=BotCommandScopeAllGroupChats()
            )

            logger.info(
                "Bot commands registered successfully",
                private_commands_count=len(private_commands),
                group_commands_count=len(group_commands),
            )

        except Exception as e:
            logger.error("Failed to register bot commands", error=str(e))
            # Don't fail initialization if command registration fails
            # The bot will still work, just without the menu

    async def initialize(self):
        """Initialize the bot and database."""
        await self.db.initialize_database()
        logger.info("Bot initialized successfully")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user
        await self.db.get_or_create_user(
            user.id, user.username, user.first_name, user.last_name
        )

        welcome_message = """
Hey there! 👋

So you've found your way to me... cool. I'm running on Cerebras AI (which is honestly pretty fast), and I'm here to chat about whatever's on your mind.

<b>Quick commands if you need them:</b>
• <code>/help</code> - More detailed stuff
• <code>/reset</code> - Wipe our chat history clean  
• <code>/stats</code> - See how much you've been talking to me
• <code>/model</code> - Switch AI models (if you're into that)

Here's the deal - just message me normally and I'll respond. I remember everything we talk about, so you can reference stuff from way back if you want.

<b>What I'm good at:</b>
• Actually remembering context (not like those goldfish chatbots)
• Being fast as hell thanks to Cerebras
• Not judging you for weird questions
• Working in groups too if you want to add me

<b>Groups?</b>
Yeah, add me to group chats. Admins can mess with settings using commands like <code>/group_mode</code> and <code>/group_settings</code>.

Anyway... what's up? 🤷‍♂️
        """

        await update.message.reply_text(
            welcome_message,
            parse_mode=ParseMode.HTML,
            reply_parameters=ReplyParameters(message_id=update.message.message_id),
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        is_group = update.effective_chat.type in ["group", "supergroup"]

        if is_group:
            help_message = f"""
<b>Commands &amp; stuff</b>

<b>Basic ones everyone can use:</b>
• <code>/help</code> - You're looking at it
• <code>/reset</code> - Wipe our chat clean
• <code>/stats</code> - See how chatty you've been
• <code>/model</code> - Switch AI models

<b>Admin commands (if you're in charge):</b>
• <code>/group_mode</code> - Shared memory vs personal memory
• <code>/group_settings</code> - All the settings
• <code>/group_reset</code> - Nuclear option for chat history
• <code>/group_stats</code> - Group activity breakdown

<b>How this works in groups:</b>
Just mention me (@{context.bot.username}) in your message and I'll jump in. Doesn't matter if it's @{context.bot.username}, @{context.bot.username.upper()}, or whatever.

You can reply to my messages or just mention me again to keep the conversation going.

<b>Memory modes:</b>
• <u>Shared</u>: I remember everything for the whole group
• <u>Personal</u>: I track conversations per person

<b>Rate limits:</b>
Max {self.rate_limit_messages} messages per person every {self.rate_limit_window} seconds. Admins can tweak group limits.

Questions? Just mention me and ask 🤷‍♂️
            """

            # Create inline keyboard for group commands
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Reset Chat", callback_data="quick_reset"),
                    InlineKeyboardButton("📊 Group Stats", callback_data="group_stats"),
                ],
                [
                    InlineKeyboardButton(
                        "⚙️ Group Settings", callback_data="group_settings"
                    ),
                    InlineKeyboardButton(
                        "🤖 Change Model", callback_data="change_model"
                    ),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
        else:
            help_message = f"""
<b>Commands &amp; how this works</b>

<b>Commands:</b>
• <code>/start</code> - The welcome thing
• <code>/help</code> - This message
• <code>/reset</code> - Forget everything we've talked about
• <code>/stats</code> - See your usage stats
• <code>/model</code> - Switch AI models

<b>How to use me:</b>
Just... talk to me? Like, send a message and I'll respond. That's it.

I remember our entire conversation, so you can reference stuff from ages ago if you want. I'll usually react with 👍 to show I got your message.

<b>What I can do:</b>
• Remember context (actually remember, not pretend to)
• Respond stupid fast (Cerebras is legit)
• Handle weird questions without judgment
• Work in groups if you add me there

<b>Tips I guess:</b>
• Be specific if you want better answers
• Reference old topics - I'll remember
• Use <code>/reset</code> if you want a fresh start
• Check <code>/stats</code> to see how much you've been chatting

<b>Rate limits:</b>
Max {self.rate_limit_messages} messages every {self.rate_limit_window} seconds. Don't spam and we're good.

Got questions? Just ask 🤷‍♂️
            """

            # Create inline keyboard for private commands
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Reset Chat", callback_data="quick_reset"),
                    InlineKeyboardButton("📊 Your Stats", callback_data="user_stats"),
                ],
                [
                    InlineKeyboardButton(
                        "🤖 Change Model", callback_data="change_model"
                    ),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            help_message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
            reply_parameters=ReplyParameters(message_id=update.message.message_id),
        )

    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset and /clear commands."""
        user_id = update.effective_user.id
        await self.db.clear_conversation_history(user_id)

        await update.message.reply_text(
            "🔄 <b>Memory wiped</b>\n\n"
            "Your chat history is gone. Clean slate. "
            "What's on your mind?",
            parse_mode=ParseMode.HTML,
            reply_parameters=ReplyParameters(message_id=update.message.message_id),
        )

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        user_id = update.effective_user.id
        stats = await self.db.get_user_stats(user_id)
        preferred_model = await self.db.get_user_preferred_model(user_id)

        if stats:
            stats_message = f"""
📊 <b>Your stats</b>

<b>Who you are:</b>
• Username: <code>@{html.escape(str(stats.get('username', 'N/A')))}</code>
• Name: <i>{html.escape(str(stats.get('first_name', 'N/A')))}</i>
• Joined: <u>{stats.get('created_at', 'N/A')[:10]}</u>
• Last seen: <u>{stats.get('last_active', 'N/A')[:10]}</u>

<b>Current setup:</b>
• Model: <code>{preferred_model}</code>

<b>How much you talk to me:</b>
• Messages sent: <b>{stats.get('total_messages', 0)}</b>
• Times I reacted: <b>{stats.get('reaction_count', 0)}</b>

Keep chatting and watch these numbers go up 📈
            """
        else:
            stats_message = f"""
📊 <b>No stats yet</b> 

<b>Current setup:</b>
• Model: <code>{preferred_model}</code>

Start chatting and I'll track some numbers for you.
            """

        await update.message.reply_text(
            stats_message,
            parse_mode=ParseMode.HTML,
            reply_parameters=ReplyParameters(message_id=update.message.message_id),
        )

    async def model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /model command to display available models."""
        user_id = update.effective_user.id

        try:
            # Get available models from Cerebras
            available_models = await self.cerebras_chat.get_available_models()

            if not available_models:
                await update.message.reply_text(
                    "🚫 <b>No Models Available</b>\n\nWell, that's weird. No models are showing up right now. Try again in a bit?",
                    parse_mode=ParseMode.HTML,
                    reply_parameters=ReplyParameters(
                        message_id=update.message.message_id
                    ),
                )
                return

            # Get user's current preferred model
            current_model = await self.db.get_user_preferred_model(user_id)

            # Create inline keyboard with available models
            keyboard = []
            models_per_row = 1  # One model per row for better readability

            for i in range(0, len(available_models), models_per_row):
                row = []
                for j in range(models_per_row):
                    if i + j < len(available_models):
                        model = available_models[i + j]
                        # Add checkmark for current model and reasoning indicator
                        reasoning_indicator = (
                            "🧠" if self._model_supports_reasoning(model) else ""
                        )
                        if model == current_model:
                            display_text = f"✅ {reasoning_indicator}{model}"
                        else:
                            display_text = f"{reasoning_indicator}{model}"
                        # Truncate display text if too long (Telegram button limit is ~64 chars)
                        if len(display_text) > 50:
                            display_text = display_text[:47] + "..."
                        row.append(
                            InlineKeyboardButton(
                                display_text, callback_data=f"model_select:{model}"
                            )
                        )
                keyboard.append(row)

            reply_markup = InlineKeyboardMarkup(keyboard)

            model_message = f"""
🤖 <b>Model Selection</b>

<b>Current Model:</b> <code>{current_model}</code>

Choose a different model to switch your AI experience:

<b>Legend:</b>
• ✅ Currently selected model
• 🧠 Has reasoning features (the fancy stuff)

<i>Don't worry - switching models won't delete our chat history.</i>
            """

            await update.message.reply_text(
                model_message,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup,
                reply_parameters=ReplyParameters(message_id=update.message.message_id),
            )

        except Exception as e:
            logger.error(f"Error in model_command: {e}")
            await update.message.reply_text(
                "🚫 <b>Model List Error</b>\n\nI couldn't retrieve the available AI models. This could be due to:\n• Temporary service disruption\n• Network connectivity issues\n• API authentication problems\n\nPlease try:\n• Checking your Cerebras API key\n• Trying again in a minute\n• Using the currently selected model\n\nIf the problem persists, you can continue using the current model.",
                parse_mode=ParseMode.HTML,
                reply_parameters=ReplyParameters(message_id=update.message.message_id),
            )

    async def model_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle model selection callback from inline keyboard."""
        query = update.callback_query
        user_id = update.effective_user.id

        await query.answer()

        try:
            # Parse callback data
            callback_data = query.data
            if not callback_data.startswith("model_select:"):
                return

            selected_model = callback_data.split(":", 1)[1]

            # Update user's preferred model in database
            await self.db.set_user_preferred_model(user_id, selected_model)

            # Edit the message to show successful selection
            success_message = f"""
🤖 <b>Model switched!</b>

<b>Now using:</b> <code>{selected_model}</code>

Your chat history's still there, just running on a different brain now.
            """

            await query.edit_message_text(success_message, parse_mode=ParseMode.HTML)

            logger.info(f"User {user_id} switched to model {selected_model}")

        except Exception as e:
            logger.error(f"Error in model_callback: {e}")
            await query.edit_message_text(
                "🚫 <b>Model Update Error</b>\n\nI couldn't update your preferred AI model. This could be due to:\n• Temporary database issues\n• Invalid model selection\n• Network connectivity problems\n\nPlease try:\n• Selecting the model again\n• Using /model to retry\n• Continuing with your current model\n\nIf the problem persists, your current model will continue to work.",
                parse_mode=ParseMode.HTML,
            )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages with group support."""
        user = update.effective_user
        user_id = user.id
        chat = update.effective_chat
        chat_id = chat.id
        message_text = update.message.text

        # Ensure user exists in database
        await self.db.get_or_create_user(
            user_id, user.username, user.first_name, user.last_name
        )

        # Handle group chats
        is_group = chat.type in ["group", "supergroup"]
        if is_group:
            # Check if group features are enabled
            if not self.enable_group_features:
                return  # Silently ignore group messages if features disabled
            # Ensure group exists in database
            await self.db.get_or_create_group(chat_id, chat.title, chat.type)

            # Get group settings
            group_settings = await self.db.get_group_settings(chat_id)

            # Check mention policy
            if group_settings.get("mention_policy", "mention_only") == "mention_only":
                if not self._should_respond_in_group(update, context):
                    return  # Don't respond if not mentioned

                # Remove @bot_username from start of message if present
                bot_username = context.bot.username
                if message_text.startswith(f"@{bot_username}"):
                    message_text = message_text[len(f"@{bot_username}") :].strip()

            # Check group rate limiting first
            group_rate_limit = group_settings.get("per_group_rate_limit", 50)
            rate_window = group_settings.get("rate_limit_window", 60)

            if not await self.db.check_group_rate_limit(
                chat_id, group_rate_limit, rate_window
            ):
                # Get time remaining before rate limit resets
                time_remaining = await self.db.get_rate_limit_time_remaining(
                    "group", chat_id, rate_window
                )

                await update.message.reply_text(
                    f"⏳ <b>Group rate limit exceeded</b>\n\n"
                    f"This group has reached its message limit. "
                    f"Limit: <i>{group_rate_limit}</i> messages per <i>{rate_window}</i> seconds.\n\n"
                    f"⏰ <b>Try again in:</b> <code>{time_remaining}</code> seconds",
                    parse_mode=ParseMode.HTML,
                    reply_parameters=ReplyParameters(
                        message_id=update.message.message_id
                    ),
                )
                return

        # Check user rate limiting
        user_rate_limit = self.rate_limit_messages
        if is_group:
            group_settings = await self.db.get_group_settings(chat_id)
            user_rate_limit = group_settings.get(
                "per_user_rate_limit", self.rate_limit_messages
            )

        if not await self.db.check_rate_limit(
            user_id, user_rate_limit, self.rate_limit_window
        ):
            # Get time remaining before rate limit resets
            time_remaining = await self.db.get_rate_limit_time_remaining(
                "user", user_id, self.rate_limit_window
            )

            await update.message.reply_text(
                f"⏳ <b>Rate limit exceeded</b>\n\n"
                f"Slow down there, speed racer. "
                f"Limit: <i>{user_rate_limit}</i> messages per <i>{self.rate_limit_window}</i> seconds.\n\n"
                f"⏰ <b>Try again in:</b> <code>{time_remaining}</code> seconds",
                parse_mode=ParseMode.HTML,
                reply_parameters=ReplyParameters(message_id=update.message.message_id),
            )
            return

        # Start typing indicator
        await self.start_typing_indicator(update, context)

        response_sent = False
        try:
            # Determine conversation scope and get history
            if is_group:
                group_settings = await self.db.get_group_settings(chat_id)
                mode = group_settings.get("mode", "shared")

                if mode == "shared":
                    # Use group conversation history
                    thread_key = await self.db.determine_thread_key(update.message)
                    conv_data = await self.db.get_group_conversation_history(
                        chat_id, thread_key
                    )
                    conversation_history = conv_data["history"]
                    summary = conv_data["summary"]

                    # Get group stats for context
                    group_stats = await self.db.get_group_stats(chat_id)
                    preferred_model = group_settings.get(
                        "model"
                    ) or await self.db.get_user_preferred_model(user_id)
                else:
                    # Use personal conversation history even in group
                    conversation_history = await self.db.get_conversation_history(
                        user_id
                    )
                    summary = None
                    group_stats = {}
                    preferred_model = await self.db.get_user_preferred_model(user_id)
            else:
                # Private chat - use personal history
                conversation_history = await self.db.get_conversation_history(user_id)
                summary = None
                group_stats = {}
                preferred_model = await self.db.get_user_preferred_model(user_id)

            # Get user stats for context
            user_stats = await self.db.get_user_stats(user_id)

            # Build comprehensive context
            user_context = {
                # User information from Telegram
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "language_code": user.language_code,
                "is_bot": user.is_bot,
                "is_premium": getattr(user, "is_premium", None),
                # Chat context
                "chat_id": chat_id,
                "chat_type": chat.type,
                "chat_title": chat.title,
                "is_group": is_group,
                "message_id": update.message.message_id,
                "message_date": (
                    update.message.date.isoformat() if update.message.date else None
                ),
                # Conversation statistics
                "total_messages": len(conversation_history),
                "user_joined": user_stats.get("created_at") if user_stats else None,
                "last_active": user_stats.get("last_active") if user_stats else None,
                "reaction_count": (
                    user_stats.get("reaction_count", 0) if user_stats else 0
                ),
                # Group context (if applicable)
                "group_stats": group_stats,
                "conversation_summary": summary,
                # Additional context
                "is_new_user": len(conversation_history) == 0,
                "message_length": len(message_text),
                "has_username": user.username is not None,
            }

            # Check if this is the first interaction in a group conversation
            is_first_group_interaction = (
                is_group
                and len(conversation_history) == 0
                and self._should_respond_in_group(update, context)
            )

            # Add casual greeting for first-time group mentions
            if is_first_group_interaction:
                greeting_context = (
                    f"Hey, this is the first time {user.first_name or user.username or 'someone'} "
                    f"has mentioned you in '{chat.title or 'this chat'}'. "
                    f"Jump in naturally and respond to their message: {message_text}"
                )
                actual_message = f"{greeting_context}\n\nUser's message: {message_text}"
            else:
                actual_message = message_text

            # Generate AI response with full context and preferred model
            response = await self.cerebras_chat.chat_with_history(
                actual_message,
                conversation_history,
                user_context=user_context,
                model=preferred_model,
            )

            # Update conversation history based on scope
            updated_history = conversation_history + [
                {"role": "user", "content": message_text},
                {"role": "assistant", "content": response},
            ]

            # Keep context length manageable
            max_context = 40
            if is_group:
                group_settings = await self.db.get_group_settings(chat_id)
                max_context = group_settings.get("max_context_msgs", 40)

            if len(updated_history) > max_context:
                updated_history = updated_history[-max_context:]

            # Save conversation history
            if is_group and group_settings.get("mode", "shared") == "shared":
                thread_key = await self.db.determine_thread_key(update.message)
                await self.db.update_group_conversation_history(
                    chat_id, updated_history, thread_key, summary
                )
            else:
                await self.db.update_conversation_history(user_id, updated_history)

            # React to user's message as acknowledgment (configurable for groups)
            should_react = True
            if is_group:
                group_settings = await self.db.get_group_settings(chat_id)
                should_react = group_settings.get("enable_reactions", False)

            if should_react:
                try:
                    await update.message.set_reaction("👍")
                    await self.db.add_message_reaction(
                        user_id, update.message.message_id, "👍"
                    )
                except Exception as e:
                    logger.warning(f"Could not set reaction: {e}")

            # Sanitize HTML response to prevent parsing errors
            sanitized_response = self._sanitize_html_response(response)
            await update.message.reply_text(
                sanitized_response,
                parse_mode=ParseMode.HTML,
                reply_parameters=ReplyParameters(message_id=update.message.message_id),
            )
            response_sent = True

        except Exception as e:
            logger.error(
                "Error handling message",
                error=str(e),
                user_id=user_id,
                chat_id=chat_id,
                is_group=is_group,
                chat_type=chat.type,
            )
            if not response_sent:
                await update.message.reply_text(
                    "🚫 <b>Processing Error</b>\n\nI encountered an issue while processing your message. This could be due to:\n• Temporary AI service disruption\n• Message too complex or long\n• Network connectivity issues\n\nPlease try:\n• Sending a shorter message\n• Rephrasing your question\n• Trying again in a minute\n\nIf the problem persists, consider using /reset to clear conversation history.",
                    parse_mode=ParseMode.HTML,
                    reply_parameters=ReplyParameters(
                        message_id=update.message.message_id
                    ),
                )
        finally:
            # Stop typing indicator
            await self.stop_typing_indicator(user_id)

    async def start_typing_indicator(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Start typing indicator for the user."""
        user_id = update.effective_user.id

        # Cancel any existing typing task
        if user_id in self.typing_tasks:
            self.typing_tasks[user_id].cancel()

        # Start new typing indicator task
        async def typing_loop():
            try:
                while True:
                    await context.bot.send_chat_action(
                        chat_id=update.effective_chat.id, action=ChatAction.TYPING
                    )
                    await asyncio.sleep(4)  # Telegram typing indicator lasts ~5 seconds
            except asyncio.CancelledError:
                pass

        self.typing_tasks[user_id] = asyncio.create_task(typing_loop())

    async def stop_typing_indicator(self, user_id: int):
        """Stop typing indicator for the user."""
        if user_id in self.typing_tasks:
            self.typing_tasks[user_id].cancel()
            del self.typing_tasks[user_id]

    # Group admin commands
    async def group_mode_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /group_mode command (admin only)."""
        if update.effective_chat.type == "private":
            await update.message.reply_text(
                "This command is only available in groups.",
                reply_parameters=ReplyParameters(message_id=update.message.message_id),
            )
            return

        if not await self._is_user_admin(update, context):
            await update.message.reply_text(
                "🚫 <b>Admin Only</b>\n\nOnly group administrators can use this command.",
                parse_mode=ParseMode.HTML,
                reply_parameters=ReplyParameters(message_id=update.message.message_id),
            )
            return

        chat_id = update.effective_chat.id
        await self.db.get_or_create_group(
            chat_id, update.effective_chat.title, update.effective_chat.type
        )

        # Create inline keyboard for mode selection
        keyboard = [
            [
                InlineKeyboardButton(
                    "🤝 Shared Memory", callback_data=f"group_mode:shared:{chat_id}"
                )
            ],
            [
                InlineKeyboardButton(
                    "👤 Personal Memory", callback_data=f"group_mode:personal:{chat_id}"
                )
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        current_settings = await self.db.get_group_settings(chat_id)
        current_mode = current_settings.get("mode", "shared")

        await update.message.reply_text(
            f"🔧 <b>Group Memory Mode</b>\n\n"
            f"Current mode: <code>{current_mode}</code>\n\n"
            f"<b>Shared Memory:</b> Bot remembers conversations for the whole group\n"
            f"<b>Personal Memory:</b> Bot remembers conversations per user individually",
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
            reply_parameters=ReplyParameters(message_id=update.message.message_id),
        )

    async def group_settings_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /group_settings command (admin only)."""
        if update.effective_chat.type == "private":
            await update.message.reply_text(
                "This command is only available in groups.",
                reply_parameters=ReplyParameters(message_id=update.message.message_id),
            )
            return

        if not await self._is_user_admin(update, context):
            await update.message.reply_text(
                "🚫 <b>Admin Only</b>\n\nOnly group administrators can use this command.",
                parse_mode=ParseMode.HTML,
                reply_parameters=ReplyParameters(message_id=update.message.message_id),
            )
            return

        chat_id = update.effective_chat.id
        await self.db.get_or_create_group(
            chat_id, update.effective_chat.title, update.effective_chat.type
        )
        settings = await self.db.get_group_settings(chat_id)

        settings_text = f"""
🔧 <b>Group Settings</b>

<b>Memory Mode:</b> <code>{settings.get('mode', 'shared')}</code>
<b>Mention Policy:</b> <code>{settings.get('mention_policy', 'mention_only')}</code>
<b>Model:</b> <code>{settings.get('model', 'default (user preference)')}</code>
<b>Max Context Messages:</b> <code>{settings.get('max_context_msgs', 40)}</code>
<b>Per-User Rate Limit:</b> <code>{settings.get('per_user_rate_limit', 10)}</code>/min
<b>Per-Group Rate Limit:</b> <code>{settings.get('per_group_rate_limit', 50)}</code>/min
<b>Enable Reactions:</b> <code>{'Yes' if settings.get('enable_reactions', False) else 'No'}</code>

Use the commands below to modify settings:
• <code>/group_mode</code> - Change memory mode
• <code>/group_model</code> - Set group model
• <code>/group_reset</code> - Reset conversations
        """

        # Create inline keyboard for quick toggles
        current_mode = settings.get("mode", "shared")
        current_mention_policy = settings.get("mention_policy", "mention_only")
        reactions_enabled = settings.get("enable_reactions", False)

        keyboard = [
            [
                InlineKeyboardButton(
                    f"🧠 Memory: {current_mode.title()}",
                    callback_data=f"toggle_mode:{'personal' if current_mode == 'shared' else 'shared'}:{chat_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    f"📢 Mentions: {current_mention_policy.replace('_', ' ').title()}",
                    callback_data=f"toggle_mentions:{'all' if current_mention_policy == 'mention_only' else 'mention_only'}:{chat_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    f"👍 Reactions: {'On' if reactions_enabled else 'Off'}",
                    callback_data=f"toggle_reactions:{'false' if reactions_enabled else 'true'}:{chat_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    "🔄 Reset Group Chat", callback_data=f"quick_group_reset:{chat_id}"
                ),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            settings_text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
            reply_parameters=ReplyParameters(message_id=update.message.message_id),
        )

    async def group_reset_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /group_reset command (admin only)."""
        if update.effective_chat.type == "private":
            await update.message.reply_text(
                "This command is only available in groups.",
                reply_parameters=ReplyParameters(message_id=update.message.message_id),
            )
            return

        if not await self._is_user_admin(update, context):
            await update.message.reply_text(
                "🚫 <b>Admin Only</b>\n\nOnly group administrators can use this command.",
                parse_mode=ParseMode.HTML,
                reply_parameters=ReplyParameters(message_id=update.message.message_id),
            )
            return

        chat_id = update.effective_chat.id
        await self.db.clear_group_conversation_history(chat_id)
        logger.info(
            "Group conversations reset",
            chat_id=chat_id,
            user_id=update.effective_user.id,
        )

        await update.message.reply_text(
            "🔄 <b>Group memory wiped</b>\n\n"
            "All the group chat history is gone. Fresh start. "
            "What's up?",
            parse_mode=ParseMode.HTML,
            reply_parameters=ReplyParameters(message_id=update.message.message_id),
        )

    async def group_stats_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /group_stats command."""
        if update.effective_chat.type == "private":
            await update.message.reply_text(
                "This command is only available in groups.",
                reply_parameters=ReplyParameters(message_id=update.message.message_id),
            )
            return

        chat_id = update.effective_chat.id
        await self.db.get_or_create_group(
            chat_id, update.effective_chat.title, update.effective_chat.type
        )

        stats = await self.db.get_group_stats(chat_id)
        settings = await self.db.get_group_settings(chat_id)

        if stats:
            stats_message = f"""
📊 <b>Group Statistics</b>

<b>Group Info:</b>
• Title: <i>{html.escape(str(stats.get('title', 'N/A')))}</i>
• Type: <code>{stats.get('type', 'N/A')}</code>
• Created: <u>{stats.get('created_at', 'N/A')[:10]}</u>
• Last Active: <u>{stats.get('last_active', 'N/A')[:10]}</u>

<b>Conversation Stats:</b>
• Active Threads: <b>{stats.get('thread_count', 0)}</b>
• Total Messages: <b>{stats.get('total_messages', 0)}</b>
• Memory Mode: <code>{settings.get('mode', 'shared')}</code>

Keep chatting to see these numbers grow! 📈
            """
        else:
            stats_message = f"""
📊 <b>No statistics available yet.</b>

<b>Current Settings:</b>
• Memory Mode: <code>{settings.get('mode', 'shared')}</code>
• Mention Policy: <code>{settings.get('mention_policy', 'mention_only')}</code>

Start chatting to build group stats!
            """

        await update.message.reply_text(
            stats_message,
            parse_mode=ParseMode.HTML,
            reply_parameters=ReplyParameters(message_id=update.message.message_id),
        )

    async def group_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle group settings callbacks."""
        query = update.callback_query
        await query.answer()

        try:
            callback_data = query.data
            if callback_data.startswith("group_mode:"):
                _, mode, chat_id = callback_data.split(":")
                chat_id = int(chat_id)

                await self.db.update_group_setting(chat_id, "mode", mode)
                logger.info(
                    "Group mode updated",
                    chat_id=chat_id,
                    mode=mode,
                    user_id=update.effective_user.id,
                )

                mode_description = (
                    "I'll remember stuff for the whole group now."
                    if mode == "shared"
                    else "I'll track conversations per person now."
                )
                success_message = f"""
🔧 <b>Group mode switched!</b>

<b>Now using:</b> <code>{mode}</code>

{mode_description}
                """

                await query.edit_message_text(
                    success_message, parse_mode=ParseMode.HTML
                )

        except Exception as e:
            logger.error(f"Error in group_callback: {e}")
            await query.edit_message_text(
                "🚫 <b>Error</b>\n\nThat didn't work. Something went wrong with the group settings. Try again?",
                parse_mode=ParseMode.HTML,
            )

    async def quick_action_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle quick action callbacks from inline buttons."""
        query = update.callback_query
        await query.answer()

        try:
            callback_data = query.data

            if callback_data == "quick_reset":
                # Handle quick reset action
                user_id = update.effective_user.id
                await self.db.clear_conversation_history(user_id)

                await query.edit_message_text(
                    "🔄 <b>Memory wiped</b>\n\n"
                    "Your chat history is gone. Clean slate. "
                    "What's on your mind?",
                    parse_mode=ParseMode.HTML,
                )

            elif callback_data == "user_stats":
                # Handle user stats action
                user_id = update.effective_user.id
                stats = await self.db.get_user_stats(user_id)
                preferred_model = await self.db.get_user_preferred_model(user_id)

                if stats:
                    stats_message = f"""
📊 <b>Your stats</b>

<b>Who you are:</b>
• Username: <code>@{html.escape(str(stats.get('username', 'N/A')))}</code>
• Name: <i>{html.escape(str(stats.get('first_name', 'N/A')))}</i>
• Joined: <u>{stats.get('created_at', 'N/A')[:10]}</u>
• Last seen: <u>{stats.get('last_active', 'N/A')[:10]}</u>

<b>Current setup:</b>
• Model: <code>{preferred_model}</code>

<b>How much you talk to me:</b>
• Messages sent: <b>{stats.get('total_messages', 0)}</b>
• Times I reacted: <b>{stats.get('reaction_count', 0)}</b>

Keep chatting and watch these numbers go up 📈
                    """
                else:
                    stats_message = f"""
📊 <b>No stats yet</b> 

<b>Current setup:</b>
• Model: <code>{preferred_model}</code>

Start chatting and I'll track some numbers for you.
                    """

                await query.edit_message_text(stats_message, parse_mode=ParseMode.HTML)

            elif callback_data == "group_stats":
                # Handle group stats action
                if update.effective_chat.type == "private":
                    await query.edit_message_text(
                        "This feature is only available in groups.",
                        parse_mode=ParseMode.HTML,
                    )
                    return

                chat_id = update.effective_chat.id
                await self.db.get_or_create_group(
                    chat_id, update.effective_chat.title, update.effective_chat.type
                )

                stats = await self.db.get_group_stats(chat_id)
                settings = await self.db.get_group_settings(chat_id)

                if stats:
                    stats_message = f"""
📊 <b>Group Statistics</b>

<b>Group Info:</b>
• Title: <i>{html.escape(str(stats.get('title', 'N/A')))}</i>
• Type: <code>{stats.get('type', 'N/A')}</code>
• Created: <u>{stats.get('created_at', 'N/A')[:10]}</u>
• Last Active: <u>{stats.get('last_active', 'N/A')[:10]}</u>

<b>Conversation Stats:</b>
• Active Threads: <b>{stats.get('thread_count', 0)}</b>
• Total Messages: <b>{stats.get('total_messages', 0)}</b>
• Memory Mode: <code>{settings.get('mode', 'shared')}</code>

Keep chatting to see these numbers grow! 📈
                    """
                else:
                    stats_message = f"""
📊 <b>No statistics available yet.</b>

<b>Current Settings:</b>
• Memory Mode: <code>{settings.get('mode', 'shared')}</code>
• Mention Policy: <code>{settings.get('mention_policy', 'mention_only')}</code>

Start chatting to build group stats!
                    """

                await query.edit_message_text(stats_message, parse_mode=ParseMode.HTML)

            elif callback_data == "group_settings":
                # Handle group settings action
                if update.effective_chat.type == "private":
                    await query.edit_message_text(
                        "This feature is only available in groups.",
                        parse_mode=ParseMode.HTML,
                    )
                    return

                if not await self._is_user_admin(update, context):
                    await query.edit_message_text(
                        "🚫 <b>Admin Only</b>\n\nOnly group administrators can use this feature.",
                        parse_mode=ParseMode.HTML,
                    )
                    return

                chat_id = update.effective_chat.id
                await self.db.get_or_create_group(
                    chat_id, update.effective_chat.title, update.effective_chat.type
                )
                settings = await self.db.get_group_settings(chat_id)

                settings_text = f"""
🔧 <b>Group Settings</b>

<b>Memory Mode:</b> <code>{settings.get('mode', 'shared')}</code>
<b>Mention Policy:</b> <code>{settings.get('mention_policy', 'mention_only')}</code>
<b>Model:</b> <code>{settings.get('model', 'default (user preference)')}</code>
<b>Max Context Messages:</b> <code>{settings.get('max_context_msgs', 40)}</code>
<b>Per-User Rate Limit:</b> <code>{settings.get('per_user_rate_limit', 10)}</code>/min
<b>Per-Group Rate Limit:</b> <code>{settings.get('per_group_rate_limit', 50)}</code>/min
<b>Enable Reactions:</b> <code>{'Yes' if settings.get('enable_reactions', False) else 'No'}</code>

Use the commands below to modify settings:
• <code>/group_mode</code> - Change memory mode
• <code>/group_model</code> - Set group model
• <code>/group_reset</code> - Reset conversations
                """

                await query.edit_message_text(settings_text, parse_mode=ParseMode.HTML)

            elif callback_data == "change_model":
                # Handle change model action for callback queries
                user_id = update.effective_user.id

                try:
                    # Get available models from Cerebras
                    available_models = await self.cerebras_chat.get_available_models()

                    if not available_models:
                        await query.edit_message_text(
                            "🚫 <b>No Models Available</b>\n\nWell, that's weird. No models are showing up right now. Try again in a bit?",
                            parse_mode=ParseMode.HTML,
                        )
                        return

                    # Get user's current preferred model
                    current_model = await self.db.get_user_preferred_model(user_id)

                    # Create inline keyboard with available models
                    keyboard = []
                    models_per_row = 1  # One model per row for better readability

                    for i in range(0, len(available_models), models_per_row):
                        row = []
                        for j in range(models_per_row):
                            if i + j < len(available_models):
                                model = available_models[i + j]
                                # Add checkmark for current model and reasoning indicator
                                reasoning_indicator = (
                                    "🧠"
                                    if self._model_supports_reasoning(model)
                                    else ""
                                )
                                if model == current_model:
                                    display_text = f"✅ {reasoning_indicator}{model}"
                                else:
                                    display_text = f"{reasoning_indicator}{model}"
                                # Truncate display text if too long (Telegram button limit is ~64 chars)
                                if len(display_text) > 50:
                                    display_text = display_text[:47] + "..."
                                row.append(
                                    InlineKeyboardButton(
                                        display_text,
                                        callback_data=f"model_select:{model}",
                                    )
                                )
                        keyboard.append(row)

                    reply_markup = InlineKeyboardMarkup(keyboard)

                    model_message = f"""
🤖 <b>Model Selection</b>

<b>Current Model:</b> <code>{current_model}</code>

Choose a different model to switch your AI experience:

<b>Legend:</b>
• ✅ Currently selected model
• 🧠 Has reasoning features (the fancy stuff)

<i>Don't worry - switching models won't delete our chat history.</i>
                    """

                    await query.edit_message_text(
                        model_message,
                        parse_mode=ParseMode.HTML,
                        reply_markup=reply_markup,
                    )

                except Exception as model_error:
                    logger.error(f"Error in model selection callback: {model_error}")
                    await query.edit_message_text(
                        "🚫 <b>Model List Error</b>\n\nI couldn't retrieve the available AI models. This could be due to:\n• Temporary service disruption\n• Network connectivity issues\n• API authentication problems\n\nPlease try:\n• Checking your Cerebras API key\n• Trying again in a minute\n• Using the currently selected model\n\nIf the problem persists, you can continue using the current model.",
                        parse_mode=ParseMode.HTML,
                    )

        except Exception as e:
            logger.error(f"Error in quick_action_callback: {e}")
            await query.edit_message_text(
                "🚫 <b>Error</b>\n\nThat didn't work. Something went wrong processing your request. Try again?",
                parse_mode=ParseMode.HTML,
            )

    async def group_toggle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle group settings toggle callbacks."""
        query = update.callback_query
        await query.answer()

        try:
            callback_data = query.data
            parts = callback_data.split(":")
            action = parts[0]

            if action == "toggle_mode":
                mode, chat_id = parts[1], int(parts[2])
                await self.db.update_group_setting(chat_id, "mode", mode)

                mode_description = (
                    "I'll remember stuff for the whole group now."
                    if mode == "shared"
                    else "I'll track conversations per person now."
                )

                await query.edit_message_text(
                    f"🔧 <b>Group mode switched!</b>\n\n"
                    f"<b>Now using:</b> <code>{mode}</code>\n\n"
                    f"{mode_description}",
                    parse_mode=ParseMode.HTML,
                )

            elif action == "toggle_mentions":
                policy, chat_id = parts[1], int(parts[2])
                await self.db.update_group_setting(chat_id, "mention_policy", policy)

                policy_description = (
                    "I'll respond to all messages in the group now."
                    if policy == "all"
                    else "I'll only respond when mentioned or replied to."
                )

                await query.edit_message_text(
                    f"📢 <b>Mention policy updated!</b>\n\n"
                    f"<b>Now using:</b> <code>{policy.replace('_', ' ')}</code>\n\n"
                    f"{policy_description}",
                    parse_mode=ParseMode.HTML,
                )

            elif action == "toggle_reactions":
                enable, chat_id = parts[1], int(parts[2])
                enable_bool = enable.lower() == "true"
                await self.db.update_group_setting(
                    chat_id, "enable_reactions", enable_bool
                )

                reaction_description = (
                    "I'll react with 👍 to user messages now."
                    if enable_bool
                    else "I'll no longer react to user messages."
                )

                await query.edit_message_text(
                    f"👍 <b>Reaction setting updated!</b>\n\n"
                    f"<b>Now:</b> <code>{'Enabled' if enable_bool else 'Disabled'}</code>\n\n"
                    f"{reaction_description}",
                    parse_mode=ParseMode.HTML,
                )

            elif action == "quick_group_reset":
                chat_id = int(parts[1])
                await self.db.clear_group_conversation_history(chat_id)

                await query.edit_message_text(
                    "🔄 <b>Group memory wiped</b>\n\n"
                    "All the group chat history is gone. Fresh start. "
                    "What's up?",
                    parse_mode=ParseMode.HTML,
                )

        except Exception as e:
            logger.error(f"Error in group_toggle_callback: {e}")
            await query.edit_message_text(
                "🚫 <b>Error</b>\n\nThat didn't work. Something went wrong updating group settings. Try again?",
                parse_mode=ParseMode.HTML,
            )

    def create_application(self) -> Application:
        """Create and configure the Telegram application."""
        application = Application.builder().token(self.token).build()

        # Add handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("reset", self.reset_command))
        application.add_handler(CommandHandler("clear", self.reset_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("model", self.model_command))

        # Group commands
        application.add_handler(CommandHandler("group_mode", self.group_mode_command))
        application.add_handler(
            CommandHandler("group_settings", self.group_settings_command)
        )
        application.add_handler(CommandHandler("group_reset", self.group_reset_command))
        application.add_handler(CommandHandler("group_stats", self.group_stats_command))

        # Callback handlers
        application.add_handler(
            CallbackQueryHandler(self.model_callback, pattern="^model_select:")
        )
        application.add_handler(
            CallbackQueryHandler(self.group_callback, pattern="^group_mode:")
        )
        application.add_handler(
            CallbackQueryHandler(
                self.quick_action_callback,
                pattern="^(quick_reset|user_stats|group_stats|group_settings|change_model)$",
            )
        )
        application.add_handler(
            CallbackQueryHandler(
                self.group_toggle_callback,
                pattern="^(toggle_mode|toggle_mentions|toggle_reactions|quick_group_reset):",
            )
        )

        # Message handler (should be last)
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

        return application

    async def run(self):
        """Run the bot."""
        await self.initialize()

        application = self.create_application()

        logger.info("Starting Telegram bot...")
        await application.initialize()
        await application.start()

        # Register bot commands in Telegram's menu
        await self._register_bot_commands(application)

        await application.updater.start_polling()

        try:
            # Keep the bot running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down bot...")
        finally:
            await application.updater.stop()
            await application.stop()
            await application.shutdown()


async def main():
    """Main entry point."""
    try:
        bot = TelegramBot()
        await bot.run()
    except Exception as e:
        logger.error("Fatal error starting bot", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    # Set up logging level
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the bot
    asyncio.run(main())
