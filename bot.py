import json
import logging
import os
import signal
import sqlite3
import sys
import time
import base64
import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Optional

import requests
from openai import APIError, APIStatusError, OpenAI, RateLimitError


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("telegram-openai-bot")

RATE_LIMIT_REPLY = (
    "I can't reply right now because the OpenAI API quota for this bot is exhausted. "
    "Please check the API key, billing, or project budget."
)
OPENAI_ERROR_REPLY = "I can't reply right now because the OpenAI API request failed. Please try again later."
TELEGRAM_MESSAGE_LIMIT = 4000
TELEGRAM_CAPTION_LIMIT = 1024
TELEGRAM_FORMATTING_HEADROOM = 3500


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def supports_reasoning_effort(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("gpt-5")


def supported_verbosity(model: str, requested: str) -> str:
    normalized_model = model.strip().lower()
    normalized_requested = requested.strip().lower()

    if normalized_model == "gpt-4.1-mini" and normalized_requested == "high":
        return "medium"

    return normalized_requested or "medium"


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]

            os.environ[key] = value


def response_field(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def extract_response_text(response: Any) -> str:
    output_text = response_field(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks = []
    for item in response_field(response, "output", []) or []:
        if response_field(item, "type") != "message":
            continue

        for content in response_field(item, "content", []) or []:
            if response_field(content, "type") == "output_text":
                text = response_field(content, "text", "")
                if text:
                    chunks.append(text)

    return "\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip()).strip()


def summarize_response_output(response: Any) -> str:
    summaries = []
    for item in response_field(response, "output", []) or []:
        item_type = response_field(item, "type", "unknown")
        content_types = [
            response_field(content, "type", "unknown")
            for content in response_field(item, "content", []) or []
        ]
        if content_types:
            summaries.append(f"{item_type}:{','.join(content_types)}")
        else:
            summaries.append(item_type)

    status = response_field(response, "status", "unknown")
    if not summaries:
        return f"status={status} output=[]"
    return f"status={status} output={summaries}"


def response_incomplete_reason(response: Any) -> Optional[str]:
    incomplete_details = response_field(response, "incomplete_details")
    if not incomplete_details:
        return None
    return response_field(incomplete_details, "reason")


def split_telegram_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= limit:
        return [stripped]

    chunks = []
    remaining = stripped
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at <= 0:
            split_at = limit

        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def markdown_to_telegram_html(text: str) -> str:
    placeholders: dict[str, str] = {}
    placeholder_index = 0

    def store(value: str) -> str:
        nonlocal placeholder_index
        token = f"ZZPLACEHOLDERTOKEN{placeholder_index}ZZ"
        placeholder_index += 1
        placeholders[token] = value
        return token

    def replace_fenced_code(match: re.Match[str]) -> str:
        code = match.group(1).strip("\n")
        return store(f"<pre><code>{html.escape(code)}</code></pre>")

    def replace_inline_code(match: re.Match[str]) -> str:
        return store(f"<code>{html.escape(match.group(1))}</code>")

    def replace_link(match: re.Match[str]) -> str:
        label = html.escape(match.group(1))
        url = html.escape(match.group(2), quote=True)
        return store(f'<a href="{url}">{label}</a>')

    working = text.replace("\r\n", "\n").strip()
    working = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", replace_fenced_code, working, flags=re.DOTALL)
    working = re.sub(r"`([^`\n]+)`", replace_inline_code, working)
    working = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", replace_link, working)
    working = html.escape(working)

    formatted_lines = []
    for line in working.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("### "):
            content = stripped[4:].strip()
            formatted_lines.append(f"<b>{content}</b>")
        elif stripped.startswith("## "):
            content = stripped[3:].strip()
            formatted_lines.append(f"<b>{content}</b>")
        elif stripped.startswith("# "):
            content = stripped[2:].strip()
            formatted_lines.append(f"<b>{content}</b>")
        elif stripped.startswith("&gt; "):
            content = stripped[5:].strip()
            formatted_lines.append(f"<blockquote>{content}</blockquote>")
        else:
            formatted_lines.append(line)

    working = "\n".join(formatted_lines)
    working = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", working, flags=re.DOTALL)
    working = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", working, flags=re.DOTALL)
    working = re.sub(r"(?<!_)__(.+?)__(?!_)", r"<b>\1</b>", working, flags=re.DOTALL)
    working = re.sub(r"(?<!_)_(?!\s)(.+?)(?<!\s)_(?!_)", r"<i>\1</i>", working, flags=re.DOTALL)

    for token, value in placeholders.items():
        working = working.replace(html.escape(token), value)
        working = working.replace(token, value)

    return working


def html_to_plain_text(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", "", text)
    return html.unescape(without_tags)


def build_telegram_html_chunks(text: str) -> list[str]:
    raw_chunks = split_telegram_message(text, TELEGRAM_FORMATTING_HEADROOM)
    html_chunks = []

    for raw_chunk in raw_chunks:
        html_chunk = markdown_to_telegram_html(raw_chunk)
        if len(html_chunk) <= TELEGRAM_MESSAGE_LIMIT:
            html_chunks.append(html_chunk)
            continue

        # If formatting expansion still pushed us over Telegram's limit, split more aggressively.
        smaller_raw_chunks = split_telegram_message(raw_chunk, max(500, TELEGRAM_FORMATTING_HEADROOM // 2))
        for smaller_raw_chunk in smaller_raw_chunks:
            smaller_html_chunk = markdown_to_telegram_html(smaller_raw_chunk)
            if len(smaller_html_chunk) > TELEGRAM_MESSAGE_LIMIT:
                html_chunks.append(html.escape(smaller_raw_chunk)[:TELEGRAM_MESSAGE_LIMIT])
            else:
                html_chunks.append(smaller_html_chunk)

    return html_chunks


def extract_response_images(response: Any) -> list[bytes]:
    images = []
    for item in response_field(response, "output", []) or []:
        if response_field(item, "type") != "image_generation_call":
            continue

        encoded = response_field(item, "result")
        if not encoded:
            continue

        try:
            images.append(base64.b64decode(encoded))
        except Exception:
            LOGGER.exception("failed to decode image_generation_call result")

    return images


@dataclass
class Settings:
    telegram_bot_token: str
    openai_api_key: str
    openai_model: str
    openai_reasoning_effort: str
    openai_text_verbosity: str
    openai_max_output_tokens: int
    openai_enable_image_generation: bool
    sqlite_path: str
    memory_messages: int
    polling_timeout_seconds: int
    mention_fallback_to_reply: bool

    @classmethod
    def from_env(cls) -> "Settings":
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if not telegram_bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        return cls(
            telegram_bot_token=telegram_bot_token,
            openai_api_key=openai_api_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip(),
            openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "medium").strip(),
            openai_text_verbosity=os.getenv("OPENAI_TEXT_VERBOSITY", "medium").strip(),
            openai_max_output_tokens=max(64, int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "900"))),
            openai_enable_image_generation=os.getenv("OPENAI_ENABLE_IMAGE_GENERATION", "false").lower() == "true",
            sqlite_path=os.getenv("SQLITE_PATH", "bot.db").strip(),
            memory_messages=max(1, int(os.getenv("MEMORY_MESSAGES", "12"))),
            polling_timeout_seconds=max(1, int(os.getenv("POLLING_TIMEOUT_SECONDS", "30"))),
            mention_fallback_to_reply=os.getenv("MENTION_FALLBACK_TO_REPLY", "true").lower() == "true",
        )


class SQLiteStore:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                direction TEXT NOT NULL CHECK(direction IN ('inbound', 'outbound')),
                telegram_update_id INTEGER,
                telegram_message_id INTEGER,
                telegram_chat_id INTEGER NOT NULL,
                telegram_chat_type TEXT,
                telegram_user_id INTEGER,
                telegram_username TEXT,
                telegram_full_name TEXT,
                reply_to_message_id INTEGER,
                message_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                raw_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_chat_created
            ON messages (telegram_chat_id, id DESC);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_unique_inbound
            ON messages (telegram_update_id, telegram_message_id, direction);
            """
        )
        self.conn.commit()

    def log_message(
        self,
        *,
        direction: str,
        telegram_update_id: Optional[int],
        telegram_message_id: int,
        telegram_chat_id: int,
        telegram_chat_type: Optional[str],
        telegram_user_id: Optional[int],
        telegram_username: Optional[str],
        telegram_full_name: Optional[str],
        reply_to_message_id: Optional[int],
        message_text: str,
        raw_payload: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO messages (
                direction,
                telegram_update_id,
                telegram_message_id,
                telegram_chat_id,
                telegram_chat_type,
                telegram_user_id,
                telegram_username,
                telegram_full_name,
                reply_to_message_id,
                message_text,
                created_at,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                direction,
                telegram_update_id,
                telegram_message_id,
                telegram_chat_id,
                telegram_chat_type,
                telegram_user_id,
                telegram_username,
                telegram_full_name,
                reply_to_message_id,
                message_text,
                utc_now_iso(),
                json.dumps(raw_payload, ensure_ascii=True, separators=(",", ":")),
            ),
        )
        self.conn.commit()

    def get_recent_memory(self, chat_id: int, limit: int) -> list[sqlite3.Row]:
        cursor = self.conn.execute(
            """
            SELECT direction, telegram_full_name, telegram_username, message_text, created_at
            FROM messages
            WHERE telegram_chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        )
        rows = cursor.fetchall()
        rows.reverse()
        return rows

    def close(self) -> None:
        self.conn.close()


class TelegramBot:
    def __init__(self, settings: Settings, store: SQLiteStore):
        self.settings = settings
        self.store = store
        self.session = requests.Session()
        self.base_url = f"https://api.telegram.org/bot{settings.telegram_bot_token}"
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.bot_user: Optional[dict[str, Any]] = None
        self.running = True
        self.closed = False

    def request(
        self,
        method: str,
        payload: Optional[dict[str, Any]] = None,
        files: Optional[dict[str, Any]] = None,
    ) -> Any:
        request_kwargs = {
            "timeout": (10, self.settings.polling_timeout_seconds + 20),
        }
        if files:
            request_kwargs["data"] = payload or {}
            request_kwargs["files"] = files
        else:
            request_kwargs["json"] = payload or {}

        response = self.session.post(
            f"{self.base_url}/{method}",
            **request_kwargs,
        )
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise RuntimeError(f"Telegram API returned non-JSON for {method}: {response.text}")

        if response.status_code >= 400 or not data.get("ok"):
            raise RuntimeError(f"Telegram API error for {method}: status={response.status_code} body={data}")
        return data["result"]

    def load_bot_identity(self) -> None:
        self.bot_user = self.request("getMe")
        LOGGER.info(
            "bot identity loaded username=@%s id=%s model=%s reasoning_effort=%s text_verbosity=%s max_output_tokens=%s",
            self.bot_user.get("username"),
            self.bot_user.get("id"),
            self.settings.openai_model,
            self.settings.openai_reasoning_effort,
            supported_verbosity(self.settings.openai_model, self.settings.openai_text_verbosity),
            self.settings.openai_max_output_tokens,
        )

    @property
    def bot_username(self) -> str:
        if not self.bot_user:
            raise RuntimeError("bot identity not loaded")
        return self.bot_user["username"]

    @property
    def bot_user_id(self) -> int:
        if not self.bot_user:
            raise RuntimeError("bot identity not loaded")
        return self.bot_user["id"]

    def should_process_message(self, message: dict[str, Any]) -> bool:
        text = extract_message_text(message)
        if not text:
            return False

        chat = message.get("chat", {})
        if chat.get("type") not in {"group", "supergroup"}:
            return False

        if is_direct_mention(text, self.bot_username):
            return True

        if self.settings.mention_fallback_to_reply:
            reply = message.get("reply_to_message") or {}
            reply_from = reply.get("from") or {}
            return reply_from.get("id") == self.bot_user_id

        return False

    def build_prompt(
        self,
        message: dict[str, Any],
        memory_rows: list[sqlite3.Row],
    ) -> list[dict[str, Any]]:
        chat = message["chat"]
        text = extract_message_text(message)
        user = message.get("from") or {}

        memory_lines = []
        for row in memory_rows:
            speaker = row["telegram_full_name"] or row["telegram_username"] or "Unknown"
            role = "assistant" if row["direction"] == "outbound" else "user"
            memory_lines.append(f"[{row['created_at']}] {role} {speaker}: {row['message_text']}")

        system_prompt = (
            "You are a Telegram group chat bot. Give thoughtful, well-reasoned, and practically useful answers. Responses should be educational, truth seeking manner. "
            "When the user asks for advice or explanation, prefer depth over brevity, but stay clear and organized. "
            "Use the recent chat memory when it is relevant, but do not claim to remember anything not shown."
        )
        memory_block = "\n".join(memory_lines) if memory_lines else "(no prior chat memory)"
        user_prompt = (
            f"Telegram chat id: {chat['id']}\n"
            f"Telegram chat title: {chat.get('title', '')}\n"
            f"Sender: {format_sender_name(user)}\n"
            f"Current message: {text}\n\n"
            f"Recent rolling memory from SQLite log:\n{memory_block}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_reply(self, message: dict[str, Any], memory_rows: list[sqlite3.Row]) -> tuple[str, list[bytes]]:
        request_kwargs = {
            "model": self.settings.openai_model,
            "input": self.build_prompt(message, memory_rows),
            "text": {
                "format": {"type": "text"},
                "verbosity": supported_verbosity(
                    self.settings.openai_model,
                    self.settings.openai_text_verbosity,
                )
            },
            "max_output_tokens": self.settings.openai_max_output_tokens,
        }
        if supports_reasoning_effort(self.settings.openai_model):
            request_kwargs["reasoning"] = {"effort": self.settings.openai_reasoning_effort}
        if self.settings.openai_enable_image_generation:
            request_kwargs["tools"] = [{"type": "image_generation"}]

        response = self.openai_client.responses.create(**request_kwargs)
        text = extract_response_text(response)
        image_bytes_list = extract_response_images(response)
        incomplete_reason = response_incomplete_reason(response)
        if not text and not image_bytes_list and incomplete_reason == "max_output_tokens":
            retry_max_output_tokens = min(max(self.settings.openai_max_output_tokens * 2, 2000), 4096)
            LOGGER.warning(
                "response hit max_output_tokens during reasoning; retrying with max_output_tokens=%s",
                retry_max_output_tokens,
            )
            retry_kwargs = dict(request_kwargs)
            retry_kwargs["max_output_tokens"] = retry_max_output_tokens
            response = self.openai_client.responses.create(**retry_kwargs)
            text = extract_response_text(response)
            image_bytes_list = extract_response_images(response)

        if not text and not image_bytes_list:
            raise RuntimeError(
                "OpenAI response did not contain text or image content "
                f"({summarize_response_output(response)})"
            )
        return text, image_bytes_list

    def send_group_reply(
        self,
        chat_id: int,
        reply_to_message_id: int,
        text: str,
        parse_mode: Optional[str] = "HTML",
    ) -> dict[str, Any]:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "reply_to_message_id": reply_to_message_id,
            "allow_sending_without_reply": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        return self.request("sendMessage", payload)

    def send_group_photo(
        self,
        chat_id: int,
        reply_to_message_id: int,
        image_bytes: bytes,
        caption: str = "",
    ) -> dict[str, Any]:
        file_obj = BytesIO(image_bytes)
        file_obj.name = "openai-image.png"

        payload = {
            "chat_id": str(chat_id),
            "reply_to_message_id": str(reply_to_message_id),
            "allow_sending_without_reply": "true",
        }
        if caption:
            payload["caption"] = caption[:TELEGRAM_CAPTION_LIMIT]
            payload["parse_mode"] = "HTML"

        return self.request(
            "sendPhoto",
            payload,
            files={"photo": (file_obj.name, file_obj, "image/png")},
        )

    def send_and_log_reply(
        self,
        trigger_message: dict[str, Any],
        reply_text: str,
        image_bytes_list: Optional[list[bytes]] = None,
    ) -> None:
        chunks = split_telegram_message(reply_text) if reply_text.strip() else []
        images = image_bytes_list or []

        if not chunks and not images:
            raise RuntimeError("refusing to send empty Telegram reply")

        formatted_chunks = build_telegram_html_chunks(reply_text) if reply_text.strip() else []

        for chunk in formatted_chunks:
            try:
                sent_message = self.send_group_reply(
                    chat_id=trigger_message["chat"]["id"],
                    reply_to_message_id=trigger_message["message_id"],
                    text=chunk,
                    parse_mode="HTML",
                )
                logged_message_text = chunk
            except RuntimeError:
                LOGGER.exception("formatted Telegram chunk failed; retrying as plain text")
                plain_chunk = html_to_plain_text(chunk)[:TELEGRAM_MESSAGE_LIMIT]
                sent_message = self.send_group_reply(
                    chat_id=trigger_message["chat"]["id"],
                    reply_to_message_id=trigger_message["message_id"],
                    text=plain_chunk,
                    parse_mode=None,
                )
                logged_message_text = plain_chunk

            self.store.log_message(
                direction="outbound",
                telegram_update_id=None,
                telegram_message_id=sent_message["message_id"],
                telegram_chat_id=sent_message["chat"]["id"],
                telegram_chat_type=sent_message["chat"].get("type"),
                telegram_user_id=(sent_message.get("from") or {}).get("id"),
                telegram_username=(sent_message.get("from") or {}).get("username"),
                telegram_full_name=format_sender_name(sent_message.get("from") or {}),
                reply_to_message_id=trigger_message["message_id"],
                message_text=logged_message_text,
                raw_payload=sent_message,
            )

        for index, image_bytes in enumerate(images, start=1):
            sent_message = self.send_group_photo(
                chat_id=trigger_message["chat"]["id"],
                reply_to_message_id=trigger_message["message_id"],
                image_bytes=image_bytes,
                caption="",
            )
            self.store.log_message(
                direction="outbound",
                telegram_update_id=None,
                telegram_message_id=sent_message["message_id"],
                telegram_chat_id=sent_message["chat"]["id"],
                telegram_chat_type=sent_message["chat"].get("type"),
                telegram_user_id=(sent_message.get("from") or {}).get("id"),
                telegram_username=(sent_message.get("from") or {}).get("username"),
                telegram_full_name=format_sender_name(sent_message.get("from") or {}),
                reply_to_message_id=trigger_message["message_id"],
                message_text=f"[image {index}/{len(images)} generated by OpenAI]",
                raw_payload=sent_message,
            )

    def process_update(self, update: dict[str, Any]) -> None:
        message = update.get("message")
        if not message:
            return

        text = extract_message_text(message)
        if not text:
            return

        memory_rows = self.store.get_recent_memory(
            message["chat"]["id"],
            self.settings.memory_messages,
        )

        self.store.log_message(
            direction="inbound",
            telegram_update_id=update.get("update_id"),
            telegram_message_id=message["message_id"],
            telegram_chat_id=message["chat"]["id"],
            telegram_chat_type=message["chat"].get("type"),
            telegram_user_id=(message.get("from") or {}).get("id"),
            telegram_username=(message.get("from") or {}).get("username"),
            telegram_full_name=format_sender_name(message.get("from") or {}),
            reply_to_message_id=(message.get("reply_to_message") or {}).get("message_id"),
            message_text=text,
            raw_payload=message,
        )

        if not self.should_process_message(message):
            return

        LOGGER.info(
            "processing chat_id=%s message_id=%s from=%s",
            message["chat"]["id"],
            message["message_id"],
            format_sender_name(message.get("from") or {}),
        )

        try:
            reply_text, image_bytes_list = self.generate_reply(message, memory_rows)
        except RateLimitError:
            LOGGER.exception("openai rate limit or quota error for message_id=%s", message["message_id"])
            self.send_and_log_reply(message, RATE_LIMIT_REPLY)
            return
        except (APIStatusError, APIError):
            LOGGER.exception("openai api error for message_id=%s", message["message_id"])
            self.send_and_log_reply(message, OPENAI_ERROR_REPLY)
            return

        self.send_and_log_reply(message, reply_text, image_bytes_list)

    def run(self) -> None:
        self.load_bot_identity()
        offset: Optional[int] = None

        while self.running:
            try:
                updates = self.request(
                    "getUpdates",
                    {
                        "offset": offset,
                        "timeout": self.settings.polling_timeout_seconds,
                        "allowed_updates": ["message"],
                    },
                )
                for update in updates:
                    offset = update["update_id"] + 1
                    try:
                        self.process_update(update)
                    except Exception:
                        LOGGER.exception("failed to process update_id=%s", update.get("update_id"))
            except requests.ReadTimeout:
                LOGGER.warning("telegram long poll timed out; continuing")
            except requests.RequestException:
                LOGGER.exception("telegram polling failed; retrying")
                time.sleep(2)
            except Exception:
                LOGGER.exception("unexpected poll loop error; retrying")
                time.sleep(2)

    def request_stop(self) -> None:
        self.running = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.session.close()
        self.store.close()


def extract_message_text(message: dict[str, Any]) -> str:
    return (message.get("text") or "").strip()


def format_sender_name(user: dict[str, Any]) -> str:
    full_name = " ".join(part for part in [user.get("first_name"), user.get("last_name")] if part).strip()
    return full_name or user.get("username") or "Unknown"


def is_direct_mention(text: str, bot_username: str) -> bool:
    lowered = text.lower()
    mention = f"@{bot_username.lower()}"
    return mention in lowered


def main() -> int:
    load_dotenv()
    settings = Settings.from_env()
    store = SQLiteStore(settings.sqlite_path)
    bot = TelegramBot(settings, store)

    def handle_signal(signum: int, _frame: Any) -> None:
        LOGGER.info("received signal=%s shutting down", signum)
        bot.request_stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        bot.run()
    finally:
        bot.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
