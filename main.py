"""
AMD Developer Hackathon ACT II — Track 1: Hybrid Token-Efficient Routing Agent
Reads /input/tasks.json, classifies each task, routes to the cheapest
Fireworks model that can still answer accurately, writes /output/results.json.
"""
import json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

INPUT_PATH  = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("RESULTS_OUTPUT_PATH", "/output/results.json")
REQUEST_TIMEOUT     = 28
MAX_ATTEMPTS        = 2
MAX_WORKERS         = 8
GLOBAL_DEADLINE     = 9 * 60   # 9 minutes; scoring system has 10-min limit
FIREWORKS_PREFIX    = "accounts/fireworks/models/"

def log(*a): print(*a, file=sys.stderr, flush=True)

# ── Categories ─────────────────────────────────────────────────────────────
CATEGORIES = ["NER","SENTIMENT","SUMMARIZATION","CODE_DEBUG","CODE_GEN","LOGIC","MATH","FACTUAL"]

_BASE = ("You are an accurate, efficient AI assistant. Answer only what is asked. "
         "Be concise. Do not restate the question. Respond only in English.")
_FB   = " If the request does not match this description, just answer it directly and accurately."

SYSTEM_PROMPTS = {
    "FACTUAL":      _BASE + " Give a correct, clear explanation using as few sentences as needed." + _FB,
    "MATH":         _BASE + " Work through the math carefully, then state the final answer. Keep any explanation to 1-2 lines." + _FB,
    "SENTIMENT":    _BASE + " Classify the sentiment. Label MUST be exactly one word: Positive, Negative, or Neutral. Format: 'Sentiment: <label>. <one-sentence justification>'." + _FB,
    "SUMMARIZATION":_BASE + " Follow the requested length or format exactly. Output only the summary." + _FB,
    "NER":          _BASE + ' Extract named entities. Output ONLY a JSON object with keys "person","organization","location","date", each an array of exact strings found (empty array if none).' + _FB,
    "CODE_DEBUG":   _BASE + " Find the bug(s) and provide the corrected, complete code in a single fenced code block. Precede with a one-line bug explanation." + _FB,
    "LOGIC":        _BASE + " Solve the puzzle so every stated condition holds; double-check all constraints. Give only the final solution." + _FB,
    "CODE_GEN":     _BASE + " Write a correct, well-structured implementation in a single fenced code block. No explanation unless asked." + _FB,
}

MAX_TOKENS = {"FACTUAL":400,"MATH":450,"SENTIMENT":200,"SUMMARIZATION":300,
              "NER":400,"CODE_DEBUG":700,"LOGIC":550,"CODE_GEN":700}

REASONING_EFFORT = {"FACTUAL":"none","MATH":"adaptive","SENTIMENT":"none",
                    "SUMMARIZATION":"none","NER":"none","CODE_DEBUG":"adaptive",
                    "LOGIC":"adaptive","CODE_GEN":"adaptive"}

_KW = {
    "NER":          ["named entity","named entities","entities","extract the people","entity recognition","identify all entities","extract all names"],
    "SENTIMENT":    ["sentiment","positive or negative","positive, negative","classify the tone","how does the reviewer feel","opinion expressed"],
    "SUMMARIZATION":["summarize","summarise","summary","condense","tl;dr","shorten the following","in one sentence","in a single sentence"],
    "CODE_DEBUG":   ["bug","debug","fix the following code","fix this code","traceback","stack trace","doesn't work","does not work","not working","throws an error","raises an error","correct the code","find the error","what's wrong with this code"],
    "CODE_GEN":     ["write a function","write a python","write a javascript","implement a function","write code","write a program","function that","write an algorithm","write a class","implement the following"],
    "LOGIC":        ["puzzle","each of the following","exactly one of","must be true","who is the","which one is","satisfy all","constraints below","logic grid","if and only if","mutually exclusive"],
    "MATH":         ["percent","%","how many","calculate","total cost","average","sum of","profit","interest rate","ratio of","how much","projection","compound","discount"],
}
_CODE_FENCE = re.compile(r"```")

def classify(prompt: str) -> str:
    text = prompt.lower()
    scores = {c: 0 for c in CATEGORIES}
    for cat, kws in _KW.items():
        for kw in kws:
            if kw in text:
                scores[cat] += 1
    if _CODE_FENCE.search(prompt):
        if scores["CODE_DEBUG"] > 0:
            scores["CODE_DEBUG"] += 2
        elif scores["CODE_GEN"] == 0:
            scores["CODE_GEN"] += 1
    best, bscore = "FACTUAL", 0
    for c in CATEGORIES:
        if scores[c] > bscore:
            bscore, best = scores[c], c
    return best

# ── Model Router ────────────────────────────────────────────────────────────
def _size(name: str) -> int:
    m = re.search(r"(\d+)\s*b(?!\w)", name.lower())
    return int(m.group(1)) if m else 0

def _resolve_id(raw: str) -> str:
    if raw.startswith("accounts/"): return raw
    return FIREWORKS_PREFIX + raw

class Router:
    def __init__(self, api_key: str, base_url: str, models: list):
        self.api_key  = api_key
        self.base_url = base_url.rstrip("/")
        self._lock    = threading.Lock()
        self._ok      = {}   # resolved_id -> bool (None=untested)
        self._reason  = {}   # resolved_id -> supports reasoning_effort

        code   = [m for m in models if "code" in m.lower()]
        others = [m for m in models if m not in code]
        others_s = sorted(others, key=_size)
        fast   = others_s[0]  if others_s else (code[0] if code else models[0])
        strong = others_s[-1] if others_s else (code[0] if code else models[0])
        coder  = code[0] if code else strong

        def dedup(lst):
            seen = []; [seen.append(x) for x in lst if x not in seen]; return seen

        self.routes = {
            "CODE_DEBUG":   dedup([coder,strong,fast]+models),
            "CODE_GEN":     dedup([coder,strong,fast]+models),
            "MATH":         dedup([strong,coder,fast]+models),
            "LOGIC":        dedup([strong,coder,fast]+models),
            "SENTIMENT":    dedup([fast,strong,coder]+models),
            "SUMMARIZATION":dedup([fast,strong,coder]+models),
            "NER":          dedup([fast,strong,coder]+models),
            "FACTUAL":      dedup([fast,strong,coder]+models),
        }
        log(f"Router: fast={fast!r} strong={strong!r} code={coder!r}")

    def _candidates(self, raw: str):
        rid = _resolve_id(raw)
        yield rid
        for suffix in ["","instruct","chat","v0.1","v1"]:
            alt = rid + (("-" + suffix) if suffix else "")
            if alt != rid: yield alt

    def call(self, cat: str, prompt: str) -> dict:
        for raw in self.routes[cat]:
            for rid in self._candidates(raw):
                with self._lock:
                    known = self._ok.get(rid)
                if known is False:
                    continue
                for attempt in range(MAX_ATTEMPTS):
                    try:
                        result = self._do_call(cat, prompt, rid)
                        with self._lock:
                            self._ok[rid] = True
                        return result
                    except Exception as e:
                        if "model_not_found" in str(e) or "404" in str(e):
                            with self._lock:
                                self._ok[rid] = False
                            break
                        if attempt < MAX_ATTEMPTS - 1:
                            time.sleep(0.5)
        raise RuntimeError(f"All models failed for category {cat}")

    def _do_call(self, cat: str, prompt: str, model_id: str) -> dict:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS[cat]},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": MAX_TOKENS[cat],
            "temperature": 0,
        }
        # Add reasoning_effort only if model supports it (probe first time)
        effort = REASONING_EFFORT[cat]
        if effort != "none":
            with self._lock:
                r_ok = self._reason.get(model_id)
            if r_ok is not False:
                payload["reasoning_effort"] = effort

        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 400 and "reasoning_effort" in r.text:
            with self._lock:
                self._reason[model_id] = False
            payload.pop("reasoning_effort", None)
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        if not r.ok:
            if "model_not_found" in r.text or r.status_code == 404:
                raise Exception(f"model_not_found:{model_id}")
            raise Exception(f"HTTP {r.status_code}: {r.text[:200]}")

        data = r.json()
        content = data["choices"][0]["message"]["content"]
        usage   = data.get("usage", {})
        log(f"  [{cat}] model={model_id} tokens={usage.get('total_tokens','?')}")
        return {"content": content, "usage": usage, "model": model_id}

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    api_key   = os.environ.get("FIREWORKS_API_KEY")
    base_url  = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    models_raw = os.environ.get("ALLOWED_MODELS", "")

    if not api_key or not models_raw:
        log("FATAL: FIREWORKS_API_KEY and ALLOWED_MODELS must be set")
        sys.exit(1)

    models = [m.strip() for m in models_raw.split(",") if m.strip()]
    log(f"ALLOWED_MODELS ({len(models)}): {models}")

    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)
    log(f"Loaded {len(tasks)} tasks")

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    router  = Router(api_key, base_url, models)
    results = [None] * len(tasks)
    deadline = time.time() + GLOBAL_DEADLINE
    errors   = 0

    def process(idx, task):
        if time.time() > deadline:
            return idx, {"task_id": task.get("task_id", idx), "response": "TIMEOUT", "error": True}
        prompt = task.get("prompt") or task.get("content") or task.get("text") or str(task)
        cat    = classify(prompt)
        try:
            out = router.call(cat, prompt)
            return idx, {
                "task_id":  task.get("task_id", idx),
                "response": out["content"],
                "category": cat,
                "model":    out["model"],
                "usage":    out["usage"],
            }
        except Exception as e:
            log(f"  Task {idx} [{cat}] FAILED: {e}")
            return idx, {"task_id": task.get("task_id", idx), "response": f"ERROR: {e}", "error": True}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process, i, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            idx, res = fut.result()
            results[idx] = res
            if res.get("error"):
                errors += 1

    log(f"Done: {len(tasks)} tasks, {errors} errors")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
