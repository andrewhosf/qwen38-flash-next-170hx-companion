#!/usr/bin/env python3
"""Smoke test for the Flash-Next vLLM lane: health, models, chat, tool-call, reasoning."""
import json, sys, time
import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-flash-next"

def main():
    c = httpx.Client(timeout=httpx.Timeout(connect=10, read=600, write=60, pool=10))
    # 1. health
    for i in range(30):
        try:
            r = c.get(f"{BASE}/health")
            if r.status_code == 200:
                print("health: OK")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("health: NOT READY after 60s"); sys.exit(1)

    # 2. models
    r = c.get(f"{BASE}/v1/models")
    ids = [m["id"] for m in r.json().get("data", [])]
    print("models:", ids)

    # 3. chat
    r = c.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: SMOKE-OK. Nothing else."}],
        "temperature": 0.0, "max_tokens": 256,
    })
    m = r.json()["choices"][0]["message"]
    txt = m.get("content") or m.get("reasoning_content") or ""
    print("chat:", repr(txt[:160]), "| finish:", r.json()["choices"][0].get("finish_reason"))

    # 4. tool call
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }]
    r = c.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is the weather in San Francisco? Use the tool."}],
        "tools": tools, "tool_choice": "auto", "temperature": 0.0, "max_tokens": 256,
    })
    msg = r.json()["choices"][0]["message"]
    tc = msg.get("tool_calls")
    print("tool_calls:", json.dumps(tc)[:300] if tc else "NONE")
    if msg.get("reasoning_content"):
        print("reasoning_content present:", msg["reasoning_content"][:80])

    # 5. reasoning effort probe (qwen3 parser: mildly long answer)
    r = c.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is 17*23? Show brief reasoning."}],
        "temperature": 0.2, "max_tokens": 300,
    })
    m = r.json()["choices"][0]["message"]
    print("math answer:", repr((m.get("content") or "")[:100]), "| reasoning len:", len(m.get("reasoning_content") or ""))

    print("SMOKE DONE")

if __name__ == "__main__":
    main()
