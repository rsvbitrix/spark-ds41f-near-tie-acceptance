#!/usr/bin/env python3
"""Night bench: single-stream decode speed on a fixed prompt set, with step time,
rows per step and draft survival from vLLM's own counters. One request at a time.
usage: nb.py LABEL [--base URL] [--sets text_ru,text_en,code,tetris,think,edit] [--seeds 2] [--out DIR]
       [--extra '{"temperature": 1.003}']"""
import argparse, json, os, re, sys, time, urllib.request


def edit_prompt():
    """A file of 24 small functions and a request to print it back with one rename: output that copies its context."""
    names = ["parse_header", "load_config", "merge_ranges", "retry_call", "chunked", "flatten", "slugify", "clamp",
             "median_of", "window_sum", "is_prime", "gcd_all", "to_base", "from_base", "rle_encode", "rle_decode",
             "dedupe_keep_order", "group_by_key", "invert_map", "deep_get", "format_bytes", "parse_duration",
             "rotate_left", "pairwise_diff"]
    body = []
    for i, n in enumerate(names):
        body.append(
            "def %s(values, limit=%d, *, strict=%s):\n"
            "    \"\"\"%s: step %d of the pipeline, keeps at most `limit` items.\"\"\"\n"
            "    result = []\n"
            "    for index, value in enumerate(values):\n"
            "        if strict and value is None:\n"
            "            raise ValueError(\"%s: empty value at %%d\" %% index)\n"
            "        if len(result) >= limit:\n"
            "            break\n"
            "        result.append((index * %d + %d, value))\n"
            "    return result\n" % (n, 10 + 3 * i, "True" if i % 2 else "False", n, i + 1, n, i + 2, 7 * i + 1))
    code = "\n\n".join(body)
    return ("Below is the file utils.py. Rename the function `clamp` to `clamp_values` everywhere and print the "
            "whole file again, every function, with no other change and no explanation.\n\n```python\n" + code + "\n```")


P = {
    "text_ru": ("Напиши подробное эссе примерно на 1200 слов о том, как устроены приливы и отливы и почему их два в сутки. Простым языком, связным текстом, без списков.", 1500, False),
    "text_en": ("Write a detailed essay of about 1200 words explaining how the human immune system fights a viral infection. Plain prose, no bullet lists.", 1500, False),
    "code": ("Write a complete Python module implementing an LRU cache with per-key TTL and thread safety, followed by pytest unit tests. Output only code.", 1500, False),
    "tetris": ("implement the tetris game. output HTML code directly - no explanation.", 3000, False),
    "think": ("В турнире 16 команд играют по олимпийской системе. Сколькими способами можно составить сетку первого круга так, чтобы две сильнейшие команды могли встретиться только в финале? Объясни и дай число.", 2500, True),
    "think_en": ("A 4x4 grid of lamps starts all off. Pressing a lamp toggles it and its orthogonal neighbours. How many distinct lamp patterns are reachable? Reason it through and give the number.", 2500, True),
    "think_code": ("Найди и объясни ошибку, затем дай исправленный код.\n\n```python\ndef merge_intervals(xs):\n    xs.sort()\n    out = [xs[0]]\n    for a, b in xs[1:]:\n        if a < out[-1][1]:\n            out[-1][1] = b\n        else:\n            out.append([a, b])\n    return out\n```\nПодумай о всех крайних случаях: пустой список, вложенные интервалы, касание границ, кортежи вместо списков.", 2500, True),
    "edit": (edit_prompt(), 3500, False),
}
WANT = ("num_requests_running", "num_requests_waiting", "generation_tokens_total",
        "spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
        "spec_decode_num_accepted_tokens_total")
POS = ("spec_decode_num_accepted_tokens_per_pos_total", "spec_decode_num_draft_tokens_per_pos_total")


def metrics(base):
    text = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        m = re.match(r"vllm:(\w+)\{([^}]*)\}\s+(\S+)", line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2), float(m.group(3))
        if name in WANT:
            out[name] = out.get(name, 0.0) + val
        elif name in POS:
            pm = re.search(r'position="(\d+)"', labels)
            if pm:
                out[name + ":" + pm.group(1)] = val
    return out


def wait_idle(base, limit=300):
    t0 = time.time()
    while True:
        m = metrics(base)
        if m.get("num_requests_running", 0) == 0 and m.get("num_requests_waiting", 0) == 0:
            return m
        if time.time() - t0 > limit:
            raise SystemExit("cluster busy for %d s" % limit)
        time.sleep(2)


def one(base, name, seed, extra):
    prompt, max_tokens, think = P[name]
    before = wait_idle(base)
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "seed": seed, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": think}}
    if not think:
        body["thinking"] = {"type": "disabled"}
    body.update(extra)
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    start, first, tokens, rchars, cchars, text = time.time(), None, 0, 0, 0, []
    with urllib.request.urlopen(req, timeout=1800) as stream:
        for line in stream:
            if not line.startswith(b"data: {"):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            for ch in chunk.get("choices", []):
                d = ch.get("delta", {})
                r = d.get("reasoning_content") or d.get("reasoning")
                c = d.get("content")
                if r or c:
                    first = first or time.time()
                if r:
                    rchars += len(r); text.append(r)
                if c:
                    cchars += len(c); text.append(c)
    end = time.time()
    time.sleep(0.7)
    after = metrics(base)
    d = {k: after.get(k, 0) - before.get(k, 0) for k in set(after) | set(before)}
    steps = d.get("spec_decode_num_drafts_total", 0)
    dec = end - first
    res = {"set": name, "seed": seed, "tokens": tokens, "decode_s": round(dec, 2), "tok_s": round(tokens / dec, 1),
           "ttft": round(first - start, 3), "rchars": rchars, "cchars": cchars,
           "clean": int(d.get("generation_tokens_total", 0)) == tokens}
    if steps:
        res.update({"steps": int(steps), "tok_step": round(tokens / steps, 3),
                    "rows_step": round(1 + d["spec_decode_num_draft_tokens_total"] / steps, 3),
                    "acc_step": round(d["spec_decode_num_accepted_tokens_total"] / steps, 3),
                    "step_ms": round(dec / steps * 1e3, 2),
                    "surv": [round(d.get(POS[0] + ":%d" % i, 0) / steps, 3) for i in range(16)],
                    "cov": [round(d.get(POS[1] + ":%d" % i, 0) / steps, 3) for i in range(16)]})
    else:
        res.update({"steps": 0, "step_ms": round(dec / max(tokens, 1) * 1e3, 2)})
    return res, "".join(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--sets", default="text_ru,text_en,code,tetris,think"); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default=os.path.expanduser("~/spark-night/results")); ap.add_argument("--extra", default="{}")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    extra = json.loads(a.extra)
    rows = []
    for name in a.sets.split(","):
        for s in range(a.seeds):
            r, text = one(a.base, name, 1000 + s, extra)
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
            with open(os.path.join(a.out, "%s.%s.%d.txt" % (a.label, name, s)), "w") as f:
                f.write(text)
    agg = {}
    for r in rows:
        g = agg.setdefault(r["set"], {"tokens": 0, "decode_s": 0.0, "steps": 0, "rows": 0.0, "acc": 0.0})
        g["tokens"] += r["tokens"]; g["decode_s"] += r["decode_s"]; g["steps"] += r.get("steps", 0)
        g["rows"] += r.get("rows_step", 0) * r.get("steps", 0); g["acc"] += r.get("acc_step", 0) * r.get("steps", 0)
    summ = {"label": a.label}
    tt = td = 0
    for k, g in agg.items():
        st = g["steps"] or 1
        summ[k] = {"tok_s": round(g["tokens"] / g["decode_s"], 1), "step_ms": round(g["decode_s"] / st * 1e3, 2),
                   "tok_step": round(g["tokens"] / st, 3), "rows_step": round(g["rows"] / st, 3),
                   "yield": round(g["tokens"] / max(g["rows"], 1e-9), 3)}
        tt += g["tokens"]; td += g["decode_s"]
    summ["all_tok_s"] = round(tt / td, 1)
    print("SUMMARY " + json.dumps(summ, ensure_ascii=False), flush=True)
    with open(os.path.join(a.out, a.label + ".json"), "w") as f:
        json.dump({"summary": summ, "rows": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
