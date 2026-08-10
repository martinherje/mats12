"""Generic connectivity check — not project code.

Verifies: OpenRouter key works, gpt-oss-120b responds, and whether the
reasoning/chain-of-thought comes back visible (load-bearing for CoT projects).
Run: uv run python scripts/api_smoke.py
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    raise SystemExit("No OPENROUTER_API_KEY in .env — see README step 1-2.")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "In one word: what is 2+2?"}],
    max_tokens=200,
    extra_body={"reasoning": {"effort": "low"}},
)
msg = resp.choices[0].message
print("model      :", resp.model)
print("provider   :", getattr(resp, "provider", "(not reported)"))
print("content    :", (msg.content or "").strip()[:100])
reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
print("reasoning  :", (reasoning or "(NOT VISIBLE — pin a provider that returns it)")[:200])
print("usage      :", resp.usage)
