import json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lib.toolcall_reward import hermes_examples, parse_calls, reward, trailing_text  # noqa: E402

GOLD = [{"name": "get_weather", "arguments": {"city": "Toronto", "unit": "c"}}]
OK = '<tool_call>\n{"name": "get_weather", "arguments": {"unit": "c", "city": "Toronto"}}\n</tool_call>'


class RewardTests(unittest.TestCase):
    def test_exact_match_full_reward(self):
        self.assertAlmostEqual(reward(OK, GOLD), 1.1)

    def test_name_only(self):
        self.assertAlmostEqual(reward('<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>', GOLD), 0.6)

    def test_trailing_text_penalised(self):
        self.assertAlmostEqual(reward(OK + "\n\nThe Prosocial Engine: a meta-analysis...", GOLD), 0.8)
        self.assertTrue(trailing_text(OK + " junk"))
        self.assertFalse(trailing_text(OK + "\n  "))

    def test_no_call_zero(self):
        self.assertEqual(reward("I would call get_weather for you.", GOLD), 0.0)

    def test_malformed_json_no_format_bonus(self):
        r = reward('<tool_call>{"name": "get_weather", "arguments": {"city": }</tool_call>', GOLD)
        self.assertEqual(r, 0.0)

    def test_multi_call_set_equality_order_free(self):
        gold = [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {}}]
        comp = '<tool_call>{"name": "b", "arguments": {}}</tool_call><tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
        self.assertAlmostEqual(reward(comp, gold), 1.1)

    def test_hermes_prep(self):
        row = {"id": "1", "conversations": [
            {"from": "system", "value": 'You are a function calling AI model. <tools>\n[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]\n</tools>'},
            {"from": "human", "value": "Weather in Toronto?"},
            {"from": "gpt", "value": OK}]}
        ex = hermes_examples([row, {"conversations": [{"from": "human", "value": "x"}, {"from": "gpt", "value": "no call"}]}])
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["gold"], GOLD if False else [{"name": "get_weather", "arguments": {"unit": "c", "city": "Toronto"}}])
        self.assertEqual(ex[0]["tools"][0]["function"]["name"], "get_weather")


if __name__ == "__main__":
    unittest.main()
