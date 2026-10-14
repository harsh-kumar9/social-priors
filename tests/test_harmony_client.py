"""CPU tests for the sealed client's harmony_final transport (D-0115 / D-0123):
the gpt-oss target's messages are rendered with the final-channel prefill, sent to
/completions, and the completions logprob block comes back in the chat shape the
closed-form scorer reads. Rendering and HTTP are mocked; no tokenizer download."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from evals.runner.cache import ResponseCache  # noqa: E402
from evals.runner.client import CompletionClient, client_from_target  # noqa: E402
from evals.runner.permutation import label_logprobs  # noqa: E402


class _Resp:
    status_code = 200
    headers: dict = {}
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


COMPLETION = {
    "id": "cmpl-1", "model": "dw2-P-gptoss20b-s0",
    "choices": [{"text": "B", "finish_reason": "stop",
                 "logprobs": {"tokens": ["B"], "token_logprobs": [-0.02],
                              "top_logprobs": [{"B": -0.02, "A": -4.1, "C": -5.0, "D": -6.2, " B": -7.0}]}}],
    "usage": {"prompt_tokens": 90, "completion_tokens": 1},
}


class HarmonyTransport(unittest.TestCase):
    def _client(self, tmp):
        return CompletionClient(provider="openai", base_url="http://localhost:1/v1",
                                api_key_env=None, cache=ResponseCache(Path(tmp)),
                                prompt_mode="harmony_final", tokenizer_id="openai/gpt-oss-20b")

    def test_completions_endpoint_with_prefill_and_chat_shaped_logprobs(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("lib.harmony.render_final", return_value="<rendered><|channel|>final<|message|>") as rf, \
             mock.patch("evals.runner.client.requests.post", return_value=_Resp(COMPLETION)) as post:
            c = self._client(tmp)
            res = c.complete(model="dw2-P-gptoss20b-s0",
                             messages=[{"role": "user", "content": "Pick A-D."}],
                             decoding={"temperature": 0, "max_tokens": 16, "logprobs": True,
                                       "top_logprobs": 5, "seed": 7,
                                       "chat_template_kwargs": {"enable_thinking": False}})
            rf.assert_called_once_with("openai/gpt-oss-20b", [{"role": "user", "content": "Pick A-D."}])
            url = post.call_args.args[0]
            body = post.call_args.kwargs["json"]
            self.assertTrue(url.endswith("/completions"))
            self.assertEqual(body["prompt"], "<rendered><|channel|>final<|message|>")
            self.assertEqual(body["logprobs"], 5)
            self.assertNotIn("chat_template_kwargs", body)
            self.assertNotIn("top_logprobs", body)
            self.assertNotIn("messages", body)
            self.assertEqual(body["seed"], 7)
            self.assertIn("<|return|>", body["stop"])
            self.assertEqual(res.response, "B")
            self.assertEqual(label_logprobs(res.logprobs, ("A", "B", "C", "D")),
                             {"A": -4.1, "B": -0.02, "C": -5.0, "D": -6.2})
            self.assertEqual(res.response_metadata["prompt_mode"], "harmony_final")
            # second call is served from the cache
            res2 = c.complete(model="dw2-P-gptoss20b-s0",
                              messages=[{"role": "user", "content": "Pick A-D."}],
                              decoding={"temperature": 0, "max_tokens": 16, "logprobs": True,
                                        "top_logprobs": 5, "seed": 7,
                                        "chat_template_kwargs": {"enable_thinking": False}})
            self.assertTrue(res2.cached)
            self.assertEqual(post.call_count, 1)

    def test_cache_key_separates_transports(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ResponseCache(Path(tmp))
            chat = CompletionClient(provider="openai", base_url="http://localhost:1/v1",
                                    api_key_env=None, cache=cache)
            harm = CompletionClient(provider="openai", base_url="http://localhost:1/v1",
                                    api_key_env=None, cache=cache,
                                    prompt_mode="harmony_final", tokenizer_id="x")
            msgs = [{"role": "user", "content": "q"}]
            dec = {"temperature": 0}
            k_chat = cache.key("m", dec, msgs)
            k_harm = cache.key("m", {**dec, "prompt_mode": "harmony_final"}, msgs)
            self.assertNotEqual(k_chat, k_harm)
            self.assertIsNone(chat.prompt_mode)
            self.assertEqual(harm.prompt_mode, "harmony_final")

    def test_target_fields_thread_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = {"provider": "openai", "base_url": "http://localhost:1/v1", "model": "m",
                 "prompt_mode": "harmony_final", "tokenizer_id": "openai/gpt-oss-20b"}
            c = client_from_target(t, ResponseCache(Path(tmp)))
            self.assertEqual((c.prompt_mode, c.tokenizer_id), ("harmony_final", "openai/gpt-oss-20b"))
            plain = client_from_target({"provider": "openai", "base_url": "http://x/v1", "model": "m"},
                                       ResponseCache(Path(tmp)))
            self.assertIsNone(plain.prompt_mode)
            with self.assertRaises(ValueError):
                CompletionClient(provider="openai", base_url="http://x", api_key_env=None,
                                 cache=ResponseCache(Path(tmp)), prompt_mode="harmony_final")


if __name__ == "__main__":
    unittest.main()
