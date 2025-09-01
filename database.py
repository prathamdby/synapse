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
        async with aiosqlite.connect(self.db_path) as db:
            # Check if we need to migrate the users table
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

            # Rate limiting table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limits (
                    user_id INTEGER PRIMARY KEY,
                    message_count INTEGER DEFAULT 0,
                    window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            await db.commit()
            logger.info("Database initialized successfully")

    async def _migrate_database(self, db):
        """Handle database migrations."""
        try:
            # Check if preferred_model column exists
            cursor = await db.execute("PRAGMA table_info(users)")
            columns = await cursor.fetchall()
            column_names = [column[1] for column in columns]

            if "preferred_model" not in column_names:
                logger.info("Adding preferred_model column to users table")
                await db.execute(
                    "ALTER TABLE users ADD COLUMN preferred_model TEXT DEFAULT 'gpt-oss-120b'"
                )
                await db.commit()
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
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT message_count, window_start FROM rate_limits WHERE user_id = ?",
                (user_id,),
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
                           WHERE user_id = ?""",
                        (current_time.isoformat(), user_id),
                    )
                    await db.commit()
                    return True
                elif message_count < max_messages:
                    # Increment counter
                    await db.execute(
                        "UPDATE rate_limits SET message_count = message_count + 1 WHERE user_id = ?",
                        (user_id,),
                    )
                    await db.commit()
                    return True
                else:
                    # Rate limit exceeded
                    return False
            else:
                # First message from user
                await db.execute(
                    "INSERT INTO rate_limits (user_id, message_count, window_start) VALUES (?, 1, ?)",
                    (user_id, current_time.isoformat()),
                )
                await db.commit()
                return True

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
