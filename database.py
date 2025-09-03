"""Database management for the Telegram bot."""

import aiosqlite
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations for the bot."""

    def __init__(self, db_path: str = "./bot_database.db"):
        self.db_path = db_path

    async def initialize_database(self):
        """Initialize the database with required tables."""
        # Ensure the database directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # Check if we need to migrate the database
            await self._migrate_database(db)

            # Users table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    preferred_model TEXT DEFAULT 'gpt-oss-120b'
                )
            """
            )

            # Conversations table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    conversation_history TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """
            )

            # Groups table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS groups (
                    chat_id INTEGER PRIMARY KEY,
                    title TEXT,
                    type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Group settings table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS group_settings (
                    chat_id INTEGER PRIMARY KEY,
                    mode TEXT DEFAULT 'shared',
                    mention_policy TEXT DEFAULT 'mention_only',
                    model TEXT,
                    max_context_msgs INTEGER DEFAULT 40,
                    per_user_rate_limit INTEGER DEFAULT 10,
                    per_group_rate_limit INTEGER DEFAULT 50,
                    rate_limit_window INTEGER DEFAULT 60,
                    enable_reactions BOOLEAN DEFAULT 0,
                    FOREIGN KEY (chat_id) REFERENCES groups (chat_id)
                )
            """
            )

            # Group conversations table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS group_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    thread_key TEXT DEFAULT 'default',
                    conversation_history TEXT,
                    summary TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    FOREIGN KEY (chat_id) REFERENCES groups (chat_id),
                    UNIQUE(chat_id, thread_key)
                )
            """
            )

            # Message reactions table (for bot acknowledgments)
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS message_reactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    message_id INTEGER,
                    reaction_emoji TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """
            )

            # Enhanced rate limiting table with scope support
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limits (
                    scope TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (scope, entity_id)
                )
            """
            )

            # Create indexes for performance
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations (user_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_group_conversations_chat_thread ON group_conversations (chat_id, thread_key)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_rate_limits_scope_entity ON rate_limits (scope, entity_id)"
            )

            await db.commit()
            logger.info("Database initialized successfully")

    async def _migrate_database(self, db):
        """Handle database migrations."""
        try:
            # Check if preferred_model column exists in users table
            cursor = await db.execute("PRAGMA table_info(users)")
            columns = await cursor.fetchall()
            column_names = [column[1] for column in columns]

            if "preferred_model" not in column_names:
                logger.info("Adding preferred_model column to users table")
                await db.execute(
                    "ALTER TABLE users ADD COLUMN preferred_model TEXT DEFAULT 'gpt-oss-120b'"
                )
                await db.commit()

            # Migrate existing rate_limits table to new schema if needed
            cursor = await db.execute("PRAGMA table_info(rate_limits)")
            rate_limit_columns = await cursor.fetchall()
            rate_limit_column_names = [column[1] for column in rate_limit_columns]

            if "scope" not in rate_limit_column_names:
                logger.info("Migrating rate_limits table to new schema")

                # Backup existing data
                cursor = await db.execute(
                    "SELECT user_id, message_count, window_start FROM rate_limits"
                )
                existing_data = await cursor.fetchall()

                # Drop old table
                await db.execute("DROP TABLE rate_limits")

                # Create new table
                await db.execute(
                    """
                    CREATE TABLE rate_limits (
                        scope TEXT NOT NULL,
                        entity_id INTEGER NOT NULL,
                        message_count INTEGER DEFAULT 0,
                        window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (scope, entity_id)
                    )
                    """
                )

                # Restore data with 'user' scope
                for user_id, message_count, window_start in existing_data:
                    await db.execute(
                        "INSERT INTO rate_limits (scope, entity_id, message_count, window_start) VALUES (?, ?, ?, ?)",
                        ("user", user_id, message_count, window_start),
                    )

                await db.commit()
                logger.info("Rate limits table migration completed successfully")

            logger.info("Database migration completed successfully")
        except Exception as e:
            logger.error(f"Database migration error: {e}")
            # Continue with initialization even if migration fails

    async def get_or_create_user(
        self,
        user_id: int,
        username: str = None,
        first_name: str = None,
        last_name: str = None,
    ) -> Dict[str, Any]:
        """Get existing user or create new one."""
        async with aiosqlite.connect(self.db_path) as db:
            # Check if user exists
            cursor = await db.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            )
            user = await cursor.fetchone()

            if user:
                # Update last active timestamp
                await db.execute(
                    "UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (user_id,),
                )
                await db.commit()

                # Convert to dict
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, user))
            else:
                # Create new user
                await db.execute(
                    """INSERT INTO users (user_id, username, first_name, last_name) 
                       VALUES (?, ?, ?, ?)""",
                    (user_id, username, first_name, last_name),
                )
                await db.commit()

                # Return the new user
                cursor = await db.execute(
                    "SELECT * FROM users WHERE user_id = ?", (user_id,)
                )
                user = await cursor.fetchone()
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, user))

    async def get_conversation_history(self, user_id: int) -> List[Dict[str, Any]]:
        """Get conversation history for a user."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT conversation_history FROM conversations WHERE user_id = ? ORDER BY last_updated DESC LIMIT 1",
                (user_id,),
            )
            result = await cursor.fetchone()

            if result and result[0]:
                try:
                    return json.loads(result[0])
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to decode conversation history for user {user_id}"
                    )
                    return []
            return []

    async def update_conversation_history(
        self, user_id: int, messages: List[Dict[str, Any]]
    ):
        """Update conversation history for a user."""
        async with aiosqlite.connect(self.db_path) as db:
            history_json = json.dumps(messages)

            # Check if conversation exists
            cursor = await db.execute(
                "SELECT id FROM conversations WHERE user_id = ?", (user_id,)
            )
            existing = await cursor.fetchone()

            if existing:
                # Update existing conversation
                await db.execute(
                    """UPDATE conversations 
                       SET conversation_history = ?, last_updated = CURRENT_TIMESTAMP, 
                           message_count = ? 
                       WHERE user_id = ?""",
                    (history_json, len(messages), user_id),
                )
            else:
                # Create new conversation
                await db.execute(
                    """INSERT INTO conversations (user_id, conversation_history, message_count) 
                       VALUES (?, ?, ?)""",
                    (user_id, history_json, len(messages)),
                )

            await db.commit()

    async def clear_conversation_history(self, user_id: int):
        """Clear conversation history for a user."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
            await db.commit()
            logger.info(f"Cleared conversation history for user {user_id}")

    async def add_message_reaction(
        self, user_id: int, message_id: int, reaction_emoji: str
    ):
        """Add a reaction to a message."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO message_reactions (user_id, message_id, reaction_emoji) 
                   VALUES (?, ?, ?)""",
                (user_id, message_id, reaction_emoji),
            )
            await db.commit()

    async def check_rate_limit(
        self, user_id: int, max_messages: int, window_seconds: int
    ) -> bool:
        """Check if user is within rate limits."""
        return await self._check_rate_limit_by_scope(
            "user", user_id, max_messages, window_seconds
        )

    async def check_group_rate_limit(
        self, chat_id: int, max_messages: int, window_seconds: int
    ) -> bool:
        """Check if group is within rate limits."""
        return await self._check_rate_limit_by_scope(
            "group", chat_id, max_messages, window_seconds
        )

    async def _check_rate_limit_by_scope(
        self, scope: str, entity_id: int, max_messages: int, window_seconds: int
    ) -> bool:
        """Check rate limits for a given scope and entity."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT message_count, window_start FROM rate_limits WHERE scope = ? AND entity_id = ?",
                (scope, entity_id),
            )
            result = await cursor.fetchone()

            current_time = datetime.now()

            if result:
                message_count, window_start = result
                window_start = datetime.fromisoformat(window_start)

                # Check if window has expired
                if (current_time - window_start).seconds >= window_seconds:
                    # Reset window
                    await db.execute(
                        """UPDATE rate_limits 
                           SET message_count = 1, window_start = ? 
                           WHERE scope = ? AND entity_id = ?""",
                        (current_time.isoformat(), scope, entity_id),
                    )
                    await db.commit()
                    return True
                elif message_count < max_messages:
                    # Increment counter
                    await db.execute(
                        "UPDATE rate_limits SET message_count = message_count + 1 WHERE scope = ? AND entity_id = ?",
                        (scope, entity_id),
                    )
                    await db.commit()
                    return True
                else:
                    # Rate limit exceeded
                    return False
            else:
                # First message from entity
                await db.execute(
                    "INSERT INTO rate_limits (scope, entity_id, message_count, window_start) VALUES (?, ?, 1, ?)",
                    (scope, entity_id, current_time.isoformat()),
                )
                await db.commit()
                return True

    async def get_rate_limit_time_remaining(
        self, scope: str, entity_id: int, window_seconds: int
    ) -> int:
        """Get the time remaining before rate limit resets."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT window_start FROM rate_limits WHERE scope = ? AND entity_id = ?",
                (scope, entity_id),
            )
            result = await cursor.fetchone()

            if result:
                window_start = datetime.fromisoformat(result[0])
                current_time = datetime.now()
                elapsed = (current_time - window_start).seconds
                remaining = window_seconds - elapsed

                # Ensure we don't return negative values
                return max(0, remaining)
            else:
                # No rate limit entry, so no time remaining
                return 0

    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get statistics for a user."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT 
                    u.username, u.first_name, u.created_at, u.last_active,
                    COALESCE(c.message_count, 0) as total_messages,
                    COUNT(mr.id) as reaction_count
                FROM users u
                LEFT JOIN conversations c ON u.user_id = c.user_id
                LEFT JOIN message_reactions mr ON u.user_id = mr.user_id
                WHERE u.user_id = ?
                GROUP BY u.user_id""",
                (user_id,),
            )
            result = await cursor.fetchone()

            if result:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            return {}

    async def get_user_preferred_model(self, user_id: int) -> str:
        """Get user's preferred model."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT preferred_model FROM users WHERE user_id = ?", (user_id,)
            )
            result = await cursor.fetchone()
            if result and result[0]:
                return result[0]
            return "gpt-oss-120b"  # Default model

    async def set_user_preferred_model(self, user_id: int, model: str):
        """Set user's preferred model."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE users SET preferred_model = ? WHERE user_id = ?",
                (model, user_id),
            )
            await db.commit()
            logger.info(f"Updated preferred model for user {user_id} to {model}")

    # Group management methods
    async def get_or_create_group(
        self, chat_id: int, title: str = None, chat_type: str = None
    ) -> Dict[str, Any]:
        """Get existing group or create new one."""
        async with aiosqlite.connect(self.db_path) as db:
            # Check if group exists
            cursor = await db.execute(
                "SELECT * FROM groups WHERE chat_id = ?", (chat_id,)
            )
            group = await cursor.fetchone()

            if group:
                # Update last active timestamp
                await db.execute(
                    "UPDATE groups SET last_active = CURRENT_TIMESTAMP WHERE chat_id = ?",
                    (chat_id,),
                )
                await db.commit()

                # Convert to dict
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, group))
            else:
                # Create new group
                await db.execute(
                    """INSERT INTO groups (chat_id, title, type) 
                       VALUES (?, ?, ?)""",
                    (chat_id, title, chat_type),
                )

                # Create default group settings
                await db.execute(
                    """INSERT INTO group_settings (chat_id, mode, mention_policy, max_context_msgs, per_user_rate_limit, per_group_rate_limit) 
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (chat_id, "shared", "mention_only", 40, 10, 50),
                )
                await db.commit()

                # Return the new group
                cursor = await db.execute(
                    "SELECT * FROM groups WHERE chat_id = ?", (chat_id,)
                )
                group = await cursor.fetchone()
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, group))

    async def get_group_settings(self, chat_id: int) -> Dict[str, Any]:
        """Get group settings."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM group_settings WHERE chat_id = ?", (chat_id,)
            )
            result = await cursor.fetchone()

            if result:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            return {}

    async def update_group_setting(self, chat_id: int, setting: str, value: Any):
        """Update a single group setting."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"UPDATE group_settings SET {setting} = ? WHERE chat_id = ?",
                (value, chat_id),
            )
            await db.commit()
            logger.info(f"Updated group {chat_id} setting {setting} to {value}")

    async def get_group_conversation_history(
        self, chat_id: int, thread_key: str = "default"
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a group thread."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT conversation_history, summary FROM group_conversations WHERE chat_id = ? AND thread_key = ? ORDER BY last_updated DESC LIMIT 1",
                (chat_id, thread_key),
            )
            result = await cursor.fetchone()

            if result and result[0]:
                try:
                    history = json.loads(result[0])
                    summary = result[1]
                    return {"history": history, "summary": summary}
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to decode group conversation history for chat {chat_id}, thread {thread_key}"
                    )
                    return {"history": [], "summary": None}
            return {"history": [], "summary": None}

    async def update_group_conversation_history(
        self,
        chat_id: int,
        messages: List[Dict[str, Any]],
        thread_key: str = "default",
        summary: str = None,
    ):
        """Update conversation history for a group thread."""
        async with aiosqlite.connect(self.db_path) as db:
            history_json = json.dumps(messages)

            # Check if conversation exists
            cursor = await db.execute(
                "SELECT id FROM group_conversations WHERE chat_id = ? AND thread_key = ?",
                (chat_id, thread_key),
            )
            existing = await cursor.fetchone()

            if existing:
                # Update existing conversation
                await db.execute(
                    """UPDATE group_conversations 
                       SET conversation_history = ?, summary = ?, last_updated = CURRENT_TIMESTAMP, 
                           message_count = ? 
                       WHERE chat_id = ? AND thread_key = ?""",
                    (history_json, summary, len(messages), chat_id, thread_key),
                )
            else:
                # Create new conversation
                await db.execute(
                    """INSERT INTO group_conversations (chat_id, thread_key, conversation_history, summary, message_count) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (chat_id, thread_key, history_json, summary, len(messages)),
                )

            await db.commit()

    async def clear_group_conversation_history(
        self, chat_id: int, thread_key: str = None
    ):
        """Clear conversation history for a group (specific thread or all threads)."""
        async with aiosqlite.connect(self.db_path) as db:
            if thread_key:
                await db.execute(
                    "DELETE FROM group_conversations WHERE chat_id = ? AND thread_key = ?",
                    (chat_id, thread_key),
                )
                logger.info(
                    f"Cleared conversation history for group {chat_id}, thread {thread_key}"
                )
            else:
                await db.execute(
                    "DELETE FROM group_conversations WHERE chat_id = ?", (chat_id,)
                )
                logger.info(f"Cleared all conversation history for group {chat_id}")
            await db.commit()

    async def get_group_stats(self, chat_id: int) -> Dict[str, Any]:
        """Get statistics for a group."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT 
                    g.title, g.type, g.created_at, g.last_active,
                    COUNT(DISTINCT gc.thread_key) as thread_count,
                    COALESCE(SUM(gc.message_count), 0) as total_messages
                FROM groups g
                LEFT JOIN group_conversations gc ON g.chat_id = gc.chat_id
                WHERE g.chat_id = ?
                GROUP BY g.chat_id""",
                (chat_id,),
            )
            result = await cursor.fetchone()

            if result:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            return {}

    async def determine_thread_key(self, message) -> str:
        """Determine the thread key for a message."""
        # Check if it's a forum topic (message_thread_id exists)
        if hasattr(message, "message_thread_id") and message.message_thread_id:
            return f"topic_{message.message_thread_id}"

        # Check if it's a reply to another message
        if hasattr(message, "reply_to_message") and message.reply_to_message:
            # For now, use a simple approach - could be enhanced to find root message
            return f"reply_{message.reply_to_message.message_id}"

        # Default thread
        return "default"
