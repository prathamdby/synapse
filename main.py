"""
Telegram Bot with Cerebras AI integration.

A sophisticated Telegram bot that uses Cerebras AI for intelligent conversations
with persistent context management and user-specific conversation history.
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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
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
        self.cerebras_chat = CerebrasChat(api_key=os.getenv("CEREBRAS_API_KEY"))

        # Rate limiting configuration
        self.rate_limit_messages = int(
            os.getenv("RATE_LIMIT_MESSAGES_PER_MINUTE", "10")
        )
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

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
🤖 <b>Welcome to the Cerebras AI Bot!</b>

I'm powered by <i>Cerebras AI</i> and I'm here to help you with questions, conversations, and more!

<b>Available Commands:</b>
• <code>/start</code> - Show this welcome message
• <code>/help</code> - Get detailed help information
• <code>/reset</code> or <code>/clear</code> - Clear your conversation history
• <code>/stats</code> - View your usage statistics
• <code>/model</code> - Switch between available AI models

Just send me any message and I'll respond using advanced AI! I remember our conversation history, so feel free to reference previous topics.

<b>Features:</b>
✨ <u>Intelligent conversations</u> with context
💾 <u>Persistent conversation memory</u>
⚡ <u>Fast responses</u> powered by Cerebras
👍 <u>Message acknowledgment</u> with reactions
🔒 <u>Private conversations</u> (each user has their own context)

Let's start chatting! 🚀
        """

        await update.message.reply_text(welcome_message, parse_mode=ParseMode.HTML)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_message = f"""
🆘 <b>Help &amp; Commands</b>

<b>Basic Commands:</b>
• <code>/start</code> - Welcome message and bot introduction
• <code>/help</code> - This help message
• <code>/reset</code> or <code>/clear</code> - Clear your conversation history
• <code>/stats</code> - View your usage statistics
• <code>/model</code> - Switch between available AI models

<b>How to Use:</b>
1. Just send me any text message
2. I'll respond with AI-generated content
3. I remember our conversation, so you can reference previous messages
4. I'll react to your messages with 👍 as acknowledgment

<b>Features:</b>
• <u>Context Awareness</u>: I remember our entire conversation
• <u>Fast Responses</u>: Powered by Cerebras AI infrastructure
• <u>Rate Limiting</u>: Fair usage limits to ensure good service for everyone
• <u>Message Acknowledgment</u>: I react to your messages to show I'm processing
• <u>Privacy</u>: Your conversations are private and isolated

<b>Tips:</b>
• Be specific in your questions for better responses
• You can ask follow-up questions referencing previous topics
• Use <code>/reset</code> or <code>/clear</code> if you want to start a fresh conversation
• Check <code>/stats</code> to see your usage patterns

<b>Rate Limits:</b>
• Maximum <i>{self.rate_limit_messages}</i> messages per <i>{self.rate_limit_window}</i> seconds
• Limits reset automatically

Need more help? Just ask me anything! 💬
        """

        await update.message.reply_text(help_message, parse_mode=ParseMode.HTML)

    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset and /clear commands."""
        user_id = update.effective_user.id
        await self.db.clear_conversation_history(user_id)

        await update.message.reply_text(
            "🔄 <b>Conversation Reset</b>\n\n"
            "Your conversation history has been cleared. "
            "We can start fresh! What would you like to talk about?",
            parse_mode=ParseMode.HTML,
        )

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        user_id = update.effective_user.id
        stats = await self.db.get_user_stats(user_id)
        preferred_model = await self.db.get_user_preferred_model(user_id)

        if stats:
            stats_message = f"""
📊 <b>Your Statistics</b>

<b>User Info:</b>
• Username: <code>@{html.escape(str(stats.get('username', 'N/A')))}</code>
• Name: <i>{html.escape(str(stats.get('first_name', 'N/A')))}</i>
• Joined: <u>{stats.get('created_at', 'N/A')[:10]}</u>
• Last Active: <u>{stats.get('last_active', 'N/A')[:10]}</u>

<b>AI Configuration:</b>
• Current Model: <code>{preferred_model}</code>

<b>Usage:</b>
• Total Messages: <b>{stats.get('total_messages', 0)}</b>
• Bot Reactions: <b>{stats.get('reaction_count', 0)}</b>

Keep chatting to see these numbers grow! 📈
            """
        else:
            stats_message = f"""
📊 <b>No statistics available yet.</b> 

<b>AI Configuration:</b>
• Current Model: <code>{preferred_model}</code>

Start chatting to build your stats!
            """

        await update.message.reply_text(stats_message, parse_mode=ParseMode.HTML)

    async def model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /model command to display available models."""
        user_id = update.effective_user.id

        try:
            # Get available models from Cerebras
            available_models = await self.cerebras_chat.get_available_models()

            if not available_models:
                await update.message.reply_text(
                    "🚫 <b>No Models Available</b>\n\nSorry, no models are currently available. Please try again later.",
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
• 🧠 Supports advanced reasoning features

<i>Note: Your conversation history will be preserved when switching models.</i>
            """

            await update.message.reply_text(
                model_message, parse_mode=ParseMode.HTML, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in model_command: {e}")
            await update.message.reply_text(
                "🚫 <b>Error</b>\n\nSorry, I couldn't fetch the available models. Please try again later.",
                parse_mode=ParseMode.HTML,
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
🤖 <b>Model Updated Successfully!</b>

<b>New Model:</b> <code>{selected_model}</code>

Your AI experience has been updated. All future conversations will use this model.

<i>Your conversation history has been preserved.</i>
            """

            await query.edit_message_text(success_message, parse_mode=ParseMode.HTML)

            logger.info(f"User {user_id} switched to model {selected_model}")

        except Exception as e:
            logger.error(f"Error in model_callback: {e}")
            await query.edit_message_text(
                "🚫 <b>Error</b>\n\nSorry, there was an error updating your model preference. Please try again.",
                parse_mode=ParseMode.HTML,
            )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages."""
        user = update.effective_user
        user_id = user.id
        message_text = update.message.text

        # Check rate limiting
        if not await self.db.check_rate_limit(
            user_id, self.rate_limit_messages, self.rate_limit_window
        ):
            await update.message.reply_text(
                f"⏳ <b>Rate limit exceeded</b>\n\n"
                f"Please wait before sending another message. "
                f"Limit: <i>{self.rate_limit_messages}</i> messages per <i>{self.rate_limit_window}</i> seconds.",
                parse_mode=ParseMode.HTML,
            )
            return

        # Ensure user exists in database
        await self.db.get_or_create_user(
            user_id, user.username, user.first_name, user.last_name
        )

        # Start typing indicator
        await self.start_typing_indicator(update, context)

        response_sent = False
        try:
            # Get conversation history
            conversation_history = await self.db.get_conversation_history(user_id)

            # Get user stats for context
            user_stats = await self.db.get_user_stats(user_id)

            # Get user's preferred model
            preferred_model = await self.db.get_user_preferred_model(user_id)

            # Build comprehensive user context
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
                "chat_id": update.effective_chat.id,
                "chat_type": update.effective_chat.type,
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
                # Additional context
                "is_new_user": len(conversation_history) == 0,
                "message_length": len(message_text),
                "has_username": user.username is not None,
            }

            # Generate AI response with full context and user's preferred model
            response = await self.cerebras_chat.chat_with_history(
                message_text,
                conversation_history,
                user_context=user_context,
                model=preferred_model,
            )

            # Update conversation history
            updated_history = conversation_history + [
                {"role": "user", "content": message_text},
                {"role": "assistant", "content": response},
            ]

            # Keep only last 20 exchanges (40 messages) to manage context length
            if len(updated_history) > 40:
                updated_history = updated_history[-40:]

            await self.db.update_conversation_history(user_id, updated_history)

            # React to user's message as acknowledgment and send response
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
                sanitized_response, parse_mode=ParseMode.HTML
            )
            response_sent = True

        except Exception as e:
            logger.error("Error handling message", error=str(e), user_id=user_id)
            if (
                not response_sent
            ):  # Only send error message if we haven't sent a response yet
                await update.message.reply_text(
                    "🚫 <b>Error</b>\n\nSorry, I encountered an error processing your message. Please try again in a moment.",
                    parse_mode=ParseMode.HTML,
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
        application.add_handler(
            CallbackQueryHandler(self.model_callback, pattern="^model_select:")
        )
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
