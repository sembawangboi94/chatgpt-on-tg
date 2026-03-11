import base64
import os
import tempfile
import unittest

from bot import (
    Settings,
    SQLiteStore,
    TelegramBot,
    markdown_to_telegram_html,
)


class FakeResponsesAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected extra OpenAI call")
        return self.responses.pop(0)


class FakeOpenAIClient:
    def __init__(self, responses):
        self.responses = FakeResponsesAPI(responses)


class RecordingTelegramBot(TelegramBot):
    def __init__(self, settings, store, openai_responses):
        super().__init__(settings, store)
        self.openai_client = FakeOpenAIClient(openai_responses)
        self.sent_requests = []
        self.bot_user = {"id": 999, "username": "chatgpt_on_tg_bot", "first_name": "ChatGPT"}

    def request(self, method, payload=None, files=None):
        self.sent_requests.append({"method": method, "payload": payload, "files": files})

        if method == "sendMessage":
            return {
                "message_id": len(self.sent_requests),
                "chat": {"id": payload["chat_id"], "type": "group"},
                "from": {"id": self.bot_user["id"], "username": self.bot_user["username"]},
            }

        if method == "sendPhoto":
            return {
                "message_id": len(self.sent_requests),
                "chat": {"id": int(payload["chat_id"]), "type": "group"},
                "from": {"id": self.bot_user["id"], "username": self.bot_user["username"]},
            }

        raise AssertionError(f"unexpected Telegram method {method}")


class BotTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.settings = Settings(
            telegram_bot_token="token",
            openai_api_key="key",
            openai_model="gpt-5.1",
            openai_reasoning_effort="high",
            openai_text_verbosity="high",
            openai_max_output_tokens=2800,
            openai_enable_image_generation=True,
            sqlite_path=self.db_path,
            memory_messages=12,
            polling_timeout_seconds=30,
            mention_fallback_to_reply=True,
        )

    def make_update(self, text):
        return {
            "update_id": 1,
            "message": {
                "message_id": 10,
                "chat": {"id": -123, "type": "group", "title": "Test Group"},
                "from": {"id": 111, "first_name": "Bailey", "last_name": "T", "username": "bailey"},
                "text": text,
            },
        }

    def test_markdown_to_telegram_html(self):
        rendered = markdown_to_telegram_html(
            "# Title\n**bold** and *italic* and `code` and [link](https://example.com)"
        )
        self.assertIn("<b>Title</b>", rendered)
        self.assertIn("<b>bold</b>", rendered)
        self.assertIn("<i>italic</i>", rendered)
        self.assertIn("<code>code</code>", rendered)
        self.assertIn('<a href="https://example.com">link</a>', rendered)

    def test_process_update_sends_formatted_text_reply(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "**Hello** `world`\n\nVisit [site](https://example.com)",
                        }
                    ],
                }
            ]
        }
        store = SQLiteStore(self.db_path)
        bot = RecordingTelegramBot(self.settings, store, [response])
        self.addCleanup(bot.close)

        bot.process_update(self.make_update("@chatgpt_on_tg_bot hi"))

        self.assertEqual(len(bot.sent_requests), 1)
        request = bot.sent_requests[0]
        self.assertEqual(request["method"], "sendMessage")
        self.assertEqual(request["payload"]["parse_mode"], "HTML")
        self.assertIn("<b>Hello</b>", request["payload"]["text"])
        self.assertIn("<code>world</code>", request["payload"]["text"])
        self.assertIn('<a href="https://example.com">site</a>', request["payload"]["text"])

        rows = store.conn.execute(
            "SELECT direction, message_text FROM messages ORDER BY id ASC"
        ).fetchall()
        self.assertEqual([row["direction"] for row in rows], ["inbound", "outbound"])

    def test_process_update_sends_text_and_photo_reply(self):
        image_data = base64.b64encode(b"fake-image-bytes").decode("ascii")
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Here is the image summary.",
                        }
                    ],
                },
                {
                    "type": "image_generation_call",
                    "result": image_data,
                },
            ]
        }
        store = SQLiteStore(self.db_path)
        bot = RecordingTelegramBot(self.settings, store, [response])
        self.addCleanup(bot.close)

        bot.process_update(self.make_update("@chatgpt_on_tg_bot show me"))

        self.assertEqual([item["method"] for item in bot.sent_requests], ["sendMessage", "sendPhoto"])
        send_message_request = bot.sent_requests[0]
        send_photo_request = bot.sent_requests[1]
        self.assertEqual(send_message_request["payload"]["text"], "Here is the image summary.")
        self.assertIsNotNone(send_photo_request["files"])
        self.assertIn("photo", send_photo_request["files"])
        photo_tuple = send_photo_request["files"]["photo"]
        self.assertEqual(photo_tuple[0], "openai-image.png")
        self.assertEqual(photo_tuple[2], "image/png")

        rows = store.conn.execute(
            "SELECT direction, message_text FROM messages ORDER BY id ASC"
        ).fetchall()
        self.assertEqual([row["direction"] for row in rows], ["inbound", "outbound", "outbound"])
        self.assertEqual(rows[2]["message_text"], "[image 1/1 generated by OpenAI]")

    def test_process_update_sends_all_long_reply_chunks(self):
        paragraph = "**Heading** " + ("word " * 500)
        long_text = "\n\n".join([paragraph, paragraph, paragraph, paragraph])
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": long_text,
                        }
                    ],
                }
            ]
        }
        store = SQLiteStore(self.db_path)
        bot = RecordingTelegramBot(self.settings, store, [response])
        self.addCleanup(bot.close)

        bot.process_update(self.make_update("@chatgpt_on_tg_bot give me a long answer"))

        message_requests = [item for item in bot.sent_requests if item["method"] == "sendMessage"]
        self.assertGreaterEqual(len(message_requests), 3)
        for request in message_requests:
            self.assertLessEqual(len(request["payload"]["text"]), 4000)

        rows = store.conn.execute(
            "SELECT direction FROM messages ORDER BY id ASC"
        ).fetchall()
        self.assertEqual([row["direction"] for row in rows], ["inbound"] + ["outbound"] * len(message_requests))


if __name__ == "__main__":
    unittest.main()
