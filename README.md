# Telegram Bot with OpenAI Responses API

Python Telegram bot for group chats using:

- Telegram Bot API long polling
- OpenAI Responses API
- SQLite message logging
- Short rolling per-chat memory built from the SQLite log

## Behavior

The bot:

- lives in a Telegram group chat
- watches for messages addressed to it with `@your_bot_username`
- optionally also responds to replies to its own messages
- sends addressed messages to OpenAI Responses API
- replies back in the same group
- sends generated images to Telegram with `sendPhoto` when the OpenAI response contains image output
- logs every inbound and outbound text message to SQLite
- builds short rolling memory per Telegram chat from recent SQLite rows

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a local `.env` file:

```bash
cp .env.example .env
```

4. Edit `.env` with your real values.

Example:

```bash
TELEGRAM_BOT_TOKEN="..."
OPENAI_API_KEY="..."
OPENAI_MODEL="gpt-5.1"
OPENAI_REASONING_EFFORT="high"
OPENAI_TEXT_VERBOSITY="high"
OPENAI_MAX_OUTPUT_TOKENS="1400"
OPENAI_ENABLE_IMAGE_GENERATION="false"
```

5. Add the bot to a Telegram group.
6. Disable privacy mode in BotFather if you want the bot to receive all group messages.
7. Run the bot:

```bash
python bot.py
```

## SQLite schema

The SQLite database is created automatically at `SQLITE_PATH` and stores all inbound and outbound text messages in a `messages` table. Rolling memory is loaded from the most recent rows for the current Telegram chat.

## Notes

- The bot only processes text messages in `group` and `supergroup` chats.
- Messages are always logged before reply filtering, so the SQLite log remains complete for text traffic the bot can see.
- Addressing currently means a direct `@bot_username` mention, plus replies to the bot when `MENTION_FALLBACK_TO_REPLY=true`.
- `.env` is loaded automatically on startup if the file exists.
- `OPENAI_REASONING_EFFORT`, `OPENAI_TEXT_VERBOSITY`, and `OPENAI_MAX_OUTPUT_TOKENS` control depth and response length for the Responses API.
- Set `OPENAI_ENABLE_IMAGE_GENERATION=true` if you want the Responses API call to allow image generation output.
- If an OpenAI response contains `image_generation_call` output items, the bot uploads those images to Telegram using `sendPhoto`.
