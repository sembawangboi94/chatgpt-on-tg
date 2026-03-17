import base64
import os
import tempfile
import unittest

import requests

from bot import (
    Settings,
    SQLiteStore,
    TelegramAPIError,
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


class RetryRecordingTelegramBot(RecordingTelegramBot):
    def __init__(self, settings, store, openai_responses):
        super().__init__(settings, store, openai_responses)
        self.failures = {"sendMessage": 1}

    def request(self, method, payload=None, files=None):
        remaining_failures = self.failures.get(method, 0)
        if remaining_failures > 0:
            self.failures[method] = remaining_failures - 1
            raise requests.ConnectionError("simulated connection reset")
        return super().request(method, payload, files)


class RateLimitRecordingTelegramBot(RecordingTelegramBot):
    def __init__(self, settings, store, openai_responses):
        super().__init__(settings, store, openai_responses)
        self.failures = {"sendMessage": 1}

    def request(self, method, payload=None, files=None):
        remaining_failures = self.failures.get(method, 0)
        if remaining_failures > 0:
            self.failures[method] = remaining_failures - 1
            raise TelegramAPIError(
                method,
                429,
                {"ok": False, "parameters": {"retry_after": 0}},
            )
        return super().request(method, payload, files)


class PartialFailureRecordingTelegramBot(RecordingTelegramBot):
    def __init__(self, settings, store, openai_responses):
        super().__init__(settings, store, openai_responses)
        self.send_message_call_index = 0

    def request(self, method, payload=None, files=None):
        if method == "sendMessage":
            self.send_message_call_index += 1
            if self.send_message_call_index in {2, 3, 4, 5, 6, 7}:
                raise requests.ConnectionError("simulated persistent send failure")
        return super().request(method, payload, files)


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

    def test_process_update_retries_telegram_send_message(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "retry please",
                        }
                    ],
                }
            ]
        }
        store = SQLiteStore(self.db_path)
        bot = RetryRecordingTelegramBot(self.settings, store, [response])
        self.addCleanup(bot.close)

        bot.process_update(self.make_update("@chatgpt_on_tg_bot retry this"))

        self.assertEqual(len(bot.sent_requests), 1)
        self.assertEqual(bot.sent_requests[0]["method"], "sendMessage")

        rows = store.conn.execute(
            "SELECT direction, message_text FROM messages ORDER BY id ASC"
        ).fetchall()
        self.assertEqual([row["direction"] for row in rows], ["inbound", "outbound"])
        self.assertIn("retry please", rows[1]["message_text"])

    def test_process_update_retries_telegram_rate_limit(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "rate limit retry please",
                        }
                    ],
                }
            ]
        }
        store = SQLiteStore(self.db_path)
        bot = RateLimitRecordingTelegramBot(self.settings, store, [response])
        self.addCleanup(bot.close)

        bot.process_update(self.make_update("@chatgpt_on_tg_bot retry rate limit"))

        self.assertEqual(len(bot.sent_requests), 1)
        self.assertEqual(bot.sent_requests[0]["method"], "sendMessage")

        rows = store.conn.execute(
            "SELECT direction, message_text FROM messages ORDER BY id ASC"
        ).fetchall()
        self.assertEqual([row["direction"] for row in rows], ["inbound", "outbound"])
        self.assertIn("rate limit retry please", rows[1]["message_text"])

    def test_generate_reply_retries_when_partial_text_hits_max_output_tokens(self):
        first_response = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "partial answer that stops too early",
                        }
                    ],
                }
            ],
        }
        second_response = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "partial answer that stops too early and then continues with the rest",
                        }
                    ],
                }
            ],
        }
        store = SQLiteStore(self.db_path)
        bot = RecordingTelegramBot(self.settings, store, [first_response, second_response])
        self.addCleanup(bot.close)

        text, image_bytes_list, notices = bot.generate_reply(
            self.make_update("@chatgpt_on_tg_bot continue")["message"],
            [],
        )

        self.assertEqual(text, "partial answer that stops too early and then continues with the rest")
        self.assertEqual(image_bytes_list, [])
        self.assertEqual(notices, [])
        self.assertEqual(len(bot.openai_client.responses.calls), 2)

        rows = store.conn.execute(
            "SELECT attempt, incomplete_reason, output_text FROM openai_responses ORDER BY id ASC"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["attempt"], 1)
        self.assertEqual(rows[0]["incomplete_reason"], "max_output_tokens")
        self.assertEqual(rows[1]["attempt"], 2)
        self.assertIn("continues with the rest", rows[1]["output_text"])

    def test_process_update_sends_notice_when_final_response_is_truncated(self):
        first_response = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "truncated once",
                        }
                    ],
                }
            ],
        }
        second_response = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "truncated twice but longer",
                        }
                    ],
                }
            ],
        }
        store = SQLiteStore(self.db_path)
        bot = RecordingTelegramBot(self.settings, store, [first_response, second_response])
        self.addCleanup(bot.close)

        bot.process_update(self.make_update("@chatgpt_on_tg_bot big answer"))

        self.assertEqual([item["method"] for item in bot.sent_requests], ["sendMessage", "sendMessage"])
        self.assertIn("truncated twice but longer", bot.sent_requests[0]["payload"]["text"])
        self.assertIn("may be incomplete because the model hit its output limit", bot.sent_requests[1]["payload"]["text"])

    def test_process_update_sends_notice_when_chunks_not_fully_delivered(self):
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
        bot = PartialFailureRecordingTelegramBot(self.settings, store, [response])
        self.addCleanup(bot.close)

        bot.process_update(self.make_update("@chatgpt_on_tg_bot long answer"))

        self.assertEqual(len(bot.sent_requests), 2)
        self.assertEqual(bot.sent_requests[0]["method"], "sendMessage")
        self.assertEqual(bot.sent_requests[1]["method"], "sendMessage")
        self.assertIn("could not deliver the full reply", bot.sent_requests[1]["payload"]["text"])


if __name__ == "__main__":
    unittest.main()
