"""
Generate v3 judgment-and-rewrite training data with an OpenAI-compatible API.

It does not contain API keys. Configure credentials with environment variables:

  export OPENAI_API_KEY="..."
  export OPENAI_BASE_URL="https://..."

Example dry run:
  python scripts/data_processing/generate_judgment_v3.py \
      --teacher-provider openai --model mimo-v2.5-pro --dry-run --num 10

Full generation:
  python scripts/data_processing/generate_judgment_v3.py \
      --teacher-provider openai --model mimo-v2.5-pro --target 8000 \
      --output data/student_notes_train_v3.json \
      --partial data/student_notes_train_v3_partial.json \
      --stats data/student_notes_train_v3_stats.txt
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import Counter
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable


UNCLEAR_OUTPUT = "无法识别其意图定理。"

V3_SYSTEM_PROMPT = """# Role
You are a mathematical statement judgment and correction engine.

# Task
First decide whether the user's mathematical statement is:
CORRECT, FALSE, INCOMPLETE, GARBLED_BUT_IDENTIFIABLE, or UNCLEAR.

# Output rules
1. You MUST first write a concise internal judgment inside a <think>...</think> block.
2. If CORRECT, output the clean mathematical statement only.
3. If FALSE, output the correct theorem or statement.
4. If INCOMPLETE, output the complete rigorous theorem with missing assumptions.
5. If GARBLED_BUT_IDENTIFIABLE, infer the intended theorem and rewrite it fully.
6. If UNCLEAR, output exactly: 无法识别其意图定理。
7. Never repeat malformed mathematical text unchanged.
8. Preserve casual wrappers only when they do not damage mathematical rigor.
9. Use standard LaTeX formatting for formulas ($...$ or $$...$$).
"""

TEACHER_SYSTEM = """You generate final answers for a mathematical statement judgment-and-correction dataset.
Return only valid JSON with key "answer".
Do not include markdown fences, explanations, or a <think> block.
For category need_check, the answer must be exactly: 无法识别其意图定理。"""

WRAPPERS_PREFIX = [
    "", "", "",
    "Please check: ",
    "Quick sanity check: ",
    "Is it correct that ",
    "I wrote: ",
    "From my notes: ",
    "Can you fix this? ",
    "Trying to recall: ",
]

WRAPPERS_SUFFIX = [
    "", "", "",
    " Is that right?",
    " Am I missing a condition?",
    " Please correct if needed.",
    " Does this sound complete?",
]

BASE_CORRECT = [
    ("correct", "Every differentiable function is continuous."),
    ("correct", "Every continuous function on a closed and bounded interval is uniformly continuous."),
    ("correct", "Every bounded monotone sequence of real numbers converges."),
    ("correct", "If a series converges absolutely, then it converges."),
    ("correct", r"If \(p\) is prime, then \(a^p \equiv a \pmod p\) for every integer \(a\)."),
    ("correct", r"If \(A\) is invertible, then \(\det(A) \ne 0\)."),
    ("correct", r"The union of any collection of open sets is open."),
    ("correct", r"If \(f\) is continuous on \([a,b]\), then \(f\) is Riemann integrable on \([a,b]\)."),
    ("correct", "Every finite integral domain is a field."),
    ("correct", "Every subgroup of a cyclic group is cyclic."),
    ("correct", "Every finite-dimensional vector space has a basis."),
    ("correct", "Every compact subset of a Hausdorff space is closed."),
    ("correct", "Every convergent sequence in a metric space is Cauchy."),
    ("correct", "Every continuous image of a compact space is compact."),
    ("correct", "A finite union of closed sets is closed."),
    ("correct", "The intersection of any collection of closed sets is closed."),
    ("correct", r"If \(f\) is differentiable at \(a\), then \(f\) is continuous at \(a\)."),
    ("correct", r"If \(A\) and \(B\) are similar matrices, then they have the same characteristic polynomial."),
    ("correct", r"If \(G\) is a finite group and \(H\le G\), then \(|H|\) divides \(|G|\)."),
    ("correct", r"If \(a\) and \(b\) are coprime integers, then there exist integers \(x,y\) such that \(ax+by=1\)."),
]

FALSE_CASES = [
    ("Every continuous function is differentiable.", "Every differentiable function is continuous."),
    ("If a series converges, then it converges absolutely.", "If a series converges absolutely, then it converges."),
    ("Every subgroup of a group is normal.", "Every subgroup of an abelian group is normal."),
    ("All infinite sets have the same cardinality.", r"The set \(\mathbb{N}\) is countably infinite, while \(\mathbb{R}\) is uncountable."),
    ("Every square matrix is diagonalizable.", "A square matrix is diagonalizable only under additional hypotheses, such as having a basis of eigenvectors."),
    (r"\(\sqrt{x^2}=x\) for all real \(x\).", r"\(\sqrt{x^2}=|x|\) for all real \(x\)."),
    (r"\(\det(A+B)=\det(A)+\det(B)\).", r"Determinants are not additive in general; however, \(\det(AB)=\det(A)\det(B)\)."),
    ("Every convergent series converges absolutely.", "Every absolutely convergent series converges, but a convergent series need not converge absolutely."),
    ("Every bounded sequence converges.", "Every bounded monotone sequence of real numbers converges."),
    ("Every injective function is surjective.", "A function can be injective without being surjective unless additional hypotheses are imposed."),
    ("Every continuous function on an open interval is bounded.", "A continuous function on a compact interval is bounded."),
    ("Every open subset of a compact space is compact.", "Every closed subset of a compact space is compact."),
    ("Every closed subset of a metric space is compact.", "Every closed subset of a compact metric space is compact."),
    ("Every irreducible polynomial over any field is linear.", "Every irreducible polynomial over an algebraically closed field is linear."),
    ("Every ring homomorphism is injective.", "A ring homomorphism is injective if and only if its kernel is zero."),
    ("Every linear map is diagonalizable.", "A linear operator is diagonalizable when its vector space has a basis consisting of eigenvectors."),
    ("Every sequence has a convergent subsequence.", r"Every bounded sequence in \(\mathbb{R}^n\) has a convergent subsequence."),
    ("Every normal matrix is symmetric.", "Every real symmetric matrix is normal, and every complex normal matrix is unitarily diagonalizable."),
    ("Every differentiable function has a continuous derivative.", "A continuously differentiable function has a continuous derivative; differentiability alone does not imply this."),
    (r"If \(ab=0\), then \(a=0\) or \(b=0\) in every ring.", r"If \(ab=0\) in an integral domain, then \(a=0\) or \(b=0\)."),
    ("Every prime ideal is maximal.", "Every maximal ideal is prime in a commutative ring with identity; a prime ideal need not be maximal."),
    ("Every quotient group is abelian.", r"A quotient group \(G/N\) is abelian if and only if the commutator subgroup of \(G\) is contained in \(N\)."),
]

MISSING_CASES = [
    (
        "A continuous function on an interval achieves its maximum and minimum.",
        "Every continuous function on a closed and bounded interval achieves its maximum and minimum.",
    ),
    (
        "Every Cauchy sequence converges.",
        "Every Cauchy sequence in a complete metric space converges.",
    ),
    (
        "A differentiable function has a zero derivative at a local extremum.",
        r"If \(f\) has a local extremum at \(c\) and is differentiable at \(c\), then \(f'(c)=0\).",
    ),
    (
        "A power series can be differentiated term by term.",
        "A power series can be differentiated term by term within its radius of convergence.",
    ),
    (
        "The inverse of a continuous bijection is continuous.",
        "A continuous bijection from a compact space to a Hausdorff space has a continuous inverse.",
    ),
    (
        "A continuous function is bounded.",
        "Every continuous real-valued function on a compact space is bounded.",
    ),
    (
        "A continuous function is uniformly continuous.",
        "Every continuous function on a compact metric space is uniformly continuous.",
    ),
    (
        "A subgroup has order dividing the group.",
        r"If \(G\) is a finite group and \(H\le G\), then \(|H|\) divides \(|G|\).",
    ),
    (
        "A matrix is invertible if its determinant is nonzero.",
        "A square matrix over a field is invertible if and only if its determinant is nonzero.",
    ),
    (
        "A polynomial has a root.",
        "Every nonconstant polynomial over an algebraically closed field has a root.",
    ),
    (
        "A bounded sequence has a convergent subsequence.",
        r"Every bounded sequence in \(\mathbb{R}^n\) has a convergent subsequence.",
    ),
    (
        "The intermediate value theorem says a continuous function takes every intermediate value.",
        r"If \(f\) is continuous on \([a,b]\) and \(N\) lies between \(f(a)\) and \(f(b)\), then there exists \(c\in[a,b]\) such that \(f(c)=N\).",
    ),
    (
        "The mean value theorem says the derivative equals the secant slope.",
        r"If \(f\) is continuous on \([a,b]\) and differentiable on \((a,b)\), then there exists \(c\in(a,b)\) such that \(f'(c)=\frac{f(b)-f(a)}{b-a}\).",
    ),
    (
        "A differentiable function with zero derivative is constant.",
        r"If \(f\) is differentiable on an interval and \(f'(x)=0\) for all \(x\) in that interval, then \(f\) is constant.",
    ),
    (
        "A continuous image is compact.",
        "The continuous image of a compact space is compact.",
    ),
    (
        "A closed subset is compact.",
        "Every closed subset of a compact space is compact.",
    ),
    (
        "A linear map has rank plus nullity equal to the dimension.",
        r"If \(T:V\to W\) is linear and \(V\) is finite-dimensional, then \(\dim V=\operatorname{rank}(T)+\operatorname{nullity}(T)\).",
    ),
    (
        "A finite integral domain is a field.",
        "Every finite integral domain with identity is a field.",
    ),
]

FORMAL_CASES = [
    (
        "Let I be in of h and let A be a wff, if v and w are val such that v(x_i)=w(x_i) the variable x_i appears free in A then v satisfies A iff w",
        r"Let \(A\) be a well-formed formula of a first-order language, and let \(v\) and \(w\) be variable assignments. If \(v(x)=w(x)\) for every variable \(x\) free in \(A\), then \(v \models A\) if and only if \(w \models A\).",
    ),
    (
        "if x not free in phi then forall x phi iff phi",
        r"If the variable \(x\) is not free in the formula \(\varphi\), then \(\forall x\,\varphi\) is logically equivalent to \(\varphi\).",
    ),
    (
        "for hom f G to H ker f normal and G/ker f image",
        r"If \(f:G\to H\) is a group homomorphism, then \(\ker(f)\) is a normal subgroup of \(G\) and \(G/\ker(f)\cong \operatorname{im}(f)\).",
    ),
    (
        "if T linear V W then dim V rank T plus null T",
        r"If \(T:V\to W\) is a linear map and \(V\) is finite-dimensional, then \(\dim V=\operatorname{rank}(T)+\operatorname{nullity}(T)\).",
    ),
    (
        "compact K in hausdorff X then K closed",
        r"If \(K\) is a compact subset of a Hausdorff space \(X\), then \(K\) is closed in \(X\).",
    ),
    (
        "cont f X to Y compact X implies fX compact",
        r"If \(f:X\to Y\) is continuous and \(X\) is compact, then \(f(X)\) is compact.",
    ),
    (
        "if f cont on ab and N between fa fb exists c fc N",
        r"If \(f\) is continuous on \([a,b]\) and \(N\) lies between \(f(a)\) and \(f(b)\), then there exists \(c\in[a,b]\) such that \(f(c)=N\).",
    ),
    (
        "MVT f cont ab diff ab open exists c fprime secant",
        r"If \(f\) is continuous on \([a,b]\) and differentiable on \((a,b)\), then there exists \(c\in(a,b)\) such that \(f'(c)=\frac{f(b)-f(a)}{b-a}\).",
    ),
    (
        "if H subgroup finite G then order H divides order G",
        r"If \(G\) is a finite group and \(H\le G\), then \(|H|\) divides \(|G|\).",
    ),
    (
        "ring hom phi R S injective iff ker phi zero",
        r"A ring homomorphism \(\varphi:R\to S\) is injective if and only if \(\ker(\varphi)=\{0\}\).",
    ),
    (
        "linear operator diagonalizable iff basis eigenvectors",
        r"A linear operator \(T:V\to V\) is diagonalizable if and only if \(V\) has a basis consisting of eigenvectors of \(T\).",
    ),
    (
        "if gcd a b one then exists x y ax plus by one",
        r"If \(a\) and \(b\) are coprime integers, then there exist integers \(x,y\) such that \(ax+by=1\).",
    ),
    (
        "prime p divides ab then p divides a or b",
        r"If \(p\) is prime and \(p\mid ab\), then \(p\mid a\) or \(p\mid b\).",
    ),
    (
        "normal matrix unitary diagonalizable",
        r"Every complex normal matrix is unitarily diagonalizable.",
    ),
]

NEED_CHECK_CASES = [
    "Let thing be valid then A iff B maybe with x.",
    "The theorem says object maps into itself when conditions happen.",
    "If all parts are nice then the answer is stable.",
    "For any structure, the property follows from the rule.",
    "Assume the symbols match and therefore the formula works.",
    "The map does the same thing after changing the variable somehow.",
    "When the space is good, every object has the desired limit.",
    "A relation between A and B holds whenever the usual assumptions are true.",
    "The formula is valid if the operation respects the structure.",
    "Everything converges under the correct condition.",
    "If the diagram commutes then the theorem follows.",
    "The object is unique up to the appropriate equivalence.",
]


def garble(text: str, rng: random.Random) -> str:
    chars = list(text)
    if not chars:
        return text
    pool = "abcdefghijklmnopqrstuvwxyz0123456789$%^&*()_+-=[]{}|;':,./<>?"
    n = max(1, int(len(chars) * rng.uniform(0.05, 0.16)))
    for _ in range(n):
        chars[rng.randrange(len(chars))] = rng.choice(pool)
    return "".join(chars)


def wrap(text: str, rng: random.Random) -> str:
    return f"{rng.choice(WRAPPERS_PREFIX)}{text}{rng.choice(WRAPPERS_SUFFIX)}".strip()


def load_wiki_statements(limit: int = 300) -> list[str]:
    path = Path("data/wikipedia_theorems.json")
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for item in raw:
        stmt = (item.get("statement") or "").strip()
        if 40 <= len(stmt) <= 420 and any(k in stmt.lower() for k in ("if ", "every", "theorem", "is ", "=")):
            out.append(stmt)
        if len(out) >= limit:
            break
    return out


def build_jobs(target: int, rng: random.Random) -> list[dict]:
    fractions = {
        "correct": 0.15,
        "false": 0.20,
        "missing_condition": 0.20,
        "garbled_identifiable": 0.25,
        "formal_logic": 0.15,
        "need_check": 0.05,
    }
    counts = {k: int(target * v) for k, v in fractions.items()}
    counts["garbled_identifiable"] += target - sum(counts.values())

    wiki = load_wiki_statements()
    jobs = []

    for _ in range(counts["correct"]):
        _, stmt = rng.choice(BASE_CORRECT)
        if wiki and rng.random() < 0.4:
            stmt = rng.choice(wiki)
        jobs.append({"category": "correct", "input": wrap(stmt, rng), "reference": stmt})

    for _ in range(counts["false"]):
        wrong, correct = rng.choice(FALSE_CASES)
        jobs.append({"category": "false", "input": wrap(wrong, rng), "reference": correct})

    for _ in range(counts["missing_condition"]):
        incomplete, complete = rng.choice(MISSING_CASES)
        jobs.append({"category": "missing_condition", "input": wrap(incomplete, rng), "reference": complete})

    for _ in range(counts["garbled_identifiable"]):
        source = rng.choice([x[1] for x in BASE_CORRECT] + [x[1] for x in MISSING_CASES] + [x[1] for x in FORMAL_CASES])
        jobs.append({"category": "garbled_identifiable", "input": wrap(garble(source, rng), rng), "reference": source})

    for _ in range(counts["formal_logic"]):
        bad, good = rng.choice(FORMAL_CASES)
        if rng.random() < 0.35:
            bad = garble(bad, rng)
        jobs.append({"category": "formal_logic", "input": wrap(bad, rng), "reference": good})

    for _ in range(counts["need_check"]):
        jobs.append({"category": "need_check", "input": wrap(rng.choice(NEED_CHECK_CASES), rng), "reference": UNCLEAR_OUTPUT})

    rng.shuffle(jobs)
    return jobs


def build_teacher_prompt(job: dict) -> str:
    return f"""Create the final answer for one v3 theorem judgment training example.

Category: {job['category']}
User input:
{job['input']}

Reference target or intended theorem:
{job['reference']}

Requirements:
- Return JSON only: {{"answer": "..."}}
- Do not write a <think> block. The training script will add it.
- If category is need_check, answer must be exactly: {UNCLEAR_OUTPUT}
- If category is garbled_identifiable, false, missing_condition, or formal_logic, answer must be a complete rigorous theorem/statement, not a near-copy of the malformed input.
- If category is correct, answer should be the clean mathematical statement.
- Use LaTeX where helpful.
"""


def build_teacher_batch_prompt(jobs: list[dict]) -> str:
    items = []
    for i, job in enumerate(jobs):
        items.append(
            f"""ID: {i}
Category: {job['category']}
User input:
{job['input']}
Reference target or intended theorem:
{job['reference']}"""
        )
    joined = "\n\n---\n\n".join(items)
    return f"""Create final answers for these v3 theorem judgment training examples.

Return JSON only, as an array:
[{{"id": 0, "answer": "..."}}, {{"id": 1, "answer": "..."}}]

Requirements:
- Do not write <think> blocks. The training script will add them.
- For category need_check, answer must be exactly: {UNCLEAR_OUTPUT}
- For garbled_identifiable, false, missing_condition, or formal_logic, answer must be a complete rigorous theorem/statement, not a near-copy of malformed input.
- For correct, answer should be the clean mathematical statement.
- Use LaTeX where helpful.

Examples:
{joined}
"""


def openai_generate(prompt: str, model: str, base_url: str, timeout: int, retries: int, max_tokens: int = 1200) -> tuple[str, int]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is not set; pass --base-url or set the environment variable")

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }
    data = json.dumps(body).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    endpoint = base_url.rstrip("/") + "/chat/completions"

    last_error = None
    for attempt in range(retries):
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            msg = (payload.get("choices") or [{}])[0].get("message") or {}
            text = (msg.get("content") or "").strip()
            usage = payload.get("usage") or {}
            tokens = int(usage.get("total_tokens") or usage.get("completion_tokens") or max(len(prompt + text) // 4, 1))
            return text, tokens
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {e.code}: {detail[:500]}"
            if e.code not in (408, 409, 429, 500, 502, 503, 504):
                break
        except Exception as e:
            last_error = repr(e)
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"API request failed after {retries} attempts: {last_error}")


def strip_json_answer_wrapper(answer: str) -> str:
    answer = (answer or "").strip()
    for _ in range(3):
        candidate = answer.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            candidate = fenced.group(1).strip()
        if not (candidate.startswith("{") and candidate.endswith("}")):
            break
        try:
            obj = json.loads(candidate)
        except Exception:
            break
        nested = obj.get("answer") or obj.get("final") or obj.get("output")
        if not isinstance(nested, str):
            break
        answer = nested.strip()
    loose = re.fullmatch(r"\{\s*['\"]answer['\"]\s*:\s*['\"](.*)['\"]\s*\}", answer.strip(), flags=re.DOTALL)
    if loose:
        return loose.group(1).strip()
    return answer.strip()


def parse_teacher_json(text: str) -> tuple[str | None, str | None]:
    text = text.strip()
    if not text:
        return None, None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None, strip_json_answer_wrapper(text)
        try:
            obj = json.loads(match.group(0))
        except Exception:
            return None, strip_json_answer_wrapper(text)
    think = str(obj.get("think") or "").strip()
    answer = strip_json_answer_wrapper(str(obj.get("answer") or obj.get("final") or obj.get("output") or "").strip())
    return think, answer


def parse_teacher_batch_json(text: str) -> dict[int, str]:
    text = (text or "").strip()
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
            except Exception:
                obj = None
    if isinstance(obj, dict):
        obj = obj.get("answers") or obj.get("items") or obj.get("results") or []
    if not isinstance(obj, list):
        return {}
    answers: dict[int, str] = {}
    for item in obj:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id"))
        except Exception:
            continue
        answer = strip_json_answer_wrapper(str(item.get("answer") or item.get("final") or item.get("output") or "").strip())
        if answer:
            answers[idx] = answer
    return answers


def fallback_answer(job: dict) -> str:
    if job["category"] == "need_check":
        return UNCLEAR_OUTPUT
    return str(job.get("reference") or "").strip()


def too_similar(a: str, b: str) -> bool:
    import difflib

    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.88


def validate_record(job: dict, think: str | None, answer: str | None) -> tuple[bool, str]:
    answer = strip_json_answer_wrapper(answer or "")
    if not answer:
        return False, "empty_answer"
    stripped = answer.strip()
    if stripped.startswith('{"answer"') or stripped.startswith("{'answer'"):
        return False, "json_wrapper_answer"
    if stripped.startswith("{") and not stripped.endswith("}"):
        return False, "truncated_json_answer"
    if job["category"] == "need_check":
        if answer != UNCLEAR_OUTPUT:
            return False, "need_check_not_exact"
        return True, "ok"
    if answer == UNCLEAR_OUTPUT:
        return False, "unexpected_unclear"
    reference = str(job.get("reference") or "").strip()
    if job["category"] in {"false", "missing_condition"} and reference and too_similar(reference, answer):
        return True, "ok"
    if job["category"] in {"garbled_identifiable", "formal_logic"} and too_similar(job["input"], answer):
        return False, "near_copy"
    return True, "ok"


def auto_think(job: dict, answer: str) -> str:
    category = job["category"]
    if category == "correct":
        return "Category: CORRECT. The input already states a recognizable mathematical theorem, so only a clean statement is needed."
    if category == "false":
        return "Category: FALSE. The input asserts an invalid mathematical statement, so the final answer replaces it with a correct theorem or fact."
    if category == "missing_condition":
        return "Category: INCOMPLETE. The input identifies a theorem but omits necessary hypotheses, so the final answer adds the missing conditions."
    if category == "garbled_identifiable":
        return "Category: GARBLED_BUT_IDENTIFIABLE. The input is noisy but the intended theorem is recognizable, so the final answer rewrites it cleanly."
    if category == "formal_logic":
        return "Category: GARBLED_BUT_IDENTIFIABLE. The input is formal or domain-specific but identifiable, so the final answer gives the rigorous statement."
    if category == "need_check":
        return "Category: UNCLEAR. The input does not identify a specific theorem with enough confidence, so the fixed fallback answer is required."
    return "Category: UNKNOWN. The final answer is formatted as a clean mathematical statement."


def build_record(job: dict, think: str, answer: str) -> dict:
    think = (think or "").strip() or auto_think(job, answer)
    answer = strip_json_answer_wrapper(answer)
    return {
        "category": job["category"],
        "instruction": V3_SYSTEM_PROMPT,
        "input": job["input"],
        "output": f"<think>\n{think}\n</think>\n{answer}",
    }


def final_answer(record: dict) -> str:
    return str(record.get("output", "")).split("</think>", 1)[-1].strip()


def clean_existing_records(records: list[dict]) -> tuple[list[dict], int, int]:
    cleaned = []
    fixed = 0
    dropped = 0
    for record in records:
        output = str(record.get("output", ""))
        if "</think>" not in output:
            dropped += 1
            continue
        prefix, answer = output.split("</think>", 1)
        answer = answer.strip()
        new_answer = strip_json_answer_wrapper(answer)
        if new_answer != answer:
            fixed += 1
        if not new_answer or new_answer.startswith("{") or '{"answer"' in new_answer:
            dropped += 1
            continue
        if record.get("category") == "need_check" and new_answer != UNCLEAR_OUTPUT:
            dropped += 1
            continue
        record["output"] = prefix.rstrip() + "</think>\n" + new_answer
        cleaned.append(record)
    return cleaned, fixed, dropped


def clean_records_in_place(records: list[dict]) -> list[dict]:
    cleaned, _, _ = clean_existing_records(records)
    return cleaned


def save_records(records: list[dict], path: str) -> list[dict]:
    cleaned = clean_records_in_place(records)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    return cleaned


def estimated_text_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    ratio = min(1.0, max(0.0, done / total))
    filled = int(round(ratio * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def progress_line(
    label: str,
    done: int,
    total: int,
    accepted: int,
    failed: int,
    total_tokens: int,
    started_at: float,
    category: str = "",
) -> str:
    elapsed = time.time() - started_at
    ratio = done / total if total else 0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    cat = f" cat={category}" if category else ""
    return (
        f"{progress_bar(done, total)} {ratio:6.2%} "
        f"{label} {done}/{total}{cat} "
        f"accepted={accepted} failed={failed} "
        f"tokens~{total_tokens} elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
    )


def write_stats(records: list[dict], path: str, total_tokens: int) -> None:
    category_counts = Counter(r.get("category", "unknown") for r in records)
    exact_unclear = sum(
        final_answer(r) == UNCLEAR_OUTPUT
        for r in records
        if r.get("category") == "need_check"
    )
    need_total = category_counts.get("need_check", 0)
    input_tokens = [estimated_text_tokens(str(r.get("input", ""))) for r in records]
    answer_tokens = [estimated_text_tokens(final_answer(r)) for r in records]
    lines = [
        "=== judgment v3 stats ===",
        f"total records: {len(records)}",
        f"estimated total api tokens: {total_tokens}",
        "",
        "category counts:",
    ]
    for cat, count in sorted(category_counts.items()):
        lines.append(f"  {cat}: {count}")
    lines.extend([
        "",
        f"need_check exact fixed output: {exact_unclear}/{need_total}",
        "",
        "copy ratios by category:",
    ])
    for cat in sorted(category_counts):
        cat_records = [r for r in records if r.get("category") == cat]
        exact = sum(str(r.get("input", "")).strip().lower() == final_answer(r).lower() for r in cat_records)
        near = sum(too_similar(str(r.get("input", "")), final_answer(r)) for r in cat_records)
        count = max(1, len(cat_records))
        lines.append(f"  {cat}: exact={exact}/{len(cat_records)} ({exact / count:.1%}), near={near}/{len(cat_records)} ({near / count:.1%})")
    lines.extend([
        "",
        "estimated token length distribution:",
        f"  input p50/p90/max: {percentile(input_tokens, 0.50)}/{percentile(input_tokens, 0.90)}/{max(input_tokens, default=0)}",
        f"  answer p50/p90/max: {percentile(answer_tokens, 0.50)}/{percentile(answer_tokens, 0.90)}/{max(answer_tokens, default=0)}",
    ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[OK] Stats written to {path}")


def write_dev_cases(path: str) -> None:
    cases = [
        {
            "category": "garbled_identifiable",
            "input": "Let I be in of h and let A be a wff, if v and w are val such that v(x_i)=w(x_i) the variable x_i appears free in A then v satisfies A iff w",
            "expected": r"Let \(A\) be a well-formed formula of a first-order language, and let \(v\) and \(w\) be variable assignments. If \(v(x)=w(x)\) for every variable \(x\) free in \(A\), then \(v \models A\) if and only if \(w \models A\).",
        },
        {"category": "need_check", "input": "Let thing be valid then A iff B maybe with x.", "expected": UNCLEAR_OUTPUT},
        {"category": "correct", "input": "Every differentiable function is continuous.", "expected": "Every differentiable function is continuous."},
        {
            "category": "missing_condition",
            "input": "A continuous function on an interval achieves its maximum and minimum.",
            "expected": "Every continuous function on a closed and bounded interval achieves its maximum and minimum.",
        },
        {"category": "false", "input": "Every continuous function is differentiable.", "expected": "Every differentiable function is continuous."},
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Dev cases written to {path}")


def run_job(job: dict, args: argparse.Namespace) -> tuple[dict | None, int, str, str]:
    prompt = build_teacher_prompt(job)
    last_reason = "not_run"
    total_tokens = 0
    for _ in range(args.quality_retries):
        try:
            raw, tokens = openai_generate(
                prompt,
                model=args.model,
                base_url=args.base_url,
                timeout=args.api_timeout,
                retries=args.api_http_retries,
            )
            total_tokens += tokens
            think, answer = parse_teacher_json(raw)
            if not answer:
                answer = fallback_answer(job)
                last_reason = "fallback_reference"
            ok, reason = validate_record(job, think, answer)
            if last_reason != "fallback_reference":
                last_reason = reason
            if ok:
                return build_record(job, think or "", answer or ""), total_tokens, last_reason, ""
        except Exception as e:
            last_reason = repr(e)
        time.sleep(1)
    return None, total_tokens, last_reason, job["category"]


def run_batch(jobs: list[dict], args: argparse.Namespace) -> tuple[list[tuple[dict, str, int]], int, Counter[str]]:
    prompt = build_teacher_batch_prompt(jobs)
    total_tokens = 0
    fail_counts: Counter[str] = Counter()
    parsed: dict[int, str] = {}
    try:
        raw, tokens = openai_generate(
            prompt,
            model=args.model,
            base_url=args.base_url,
            timeout=args.api_timeout,
            retries=args.api_http_retries,
            max_tokens=max(1200, args.batch_max_tokens),
        )
        total_tokens += tokens
        parsed = parse_teacher_batch_json(raw)
    except Exception:
        parsed = {}

    out: list[tuple[dict, str, int]] = []
    for i, job in enumerate(jobs):
        answer = strip_json_answer_wrapper(parsed.get(i) or fallback_answer(job))
        reason = "api" if i in parsed else "fallback_reference"
        ok, validation_reason = validate_record(job, None, answer)
        if not ok:
            fail_counts[job["category"]] += 1
            continue
        out.append((build_record(job, "", answer), reason, total_tokens if i == 0 else 0))
    return out, total_tokens, fail_counts


def generate(args: argparse.Namespace) -> None:
    if args.teacher_provider != "openai":
        raise RuntimeError("v3 generation currently supports --teacher-provider openai")

    rng = random.Random(args.seed)
    jobs = build_jobs(args.num if args.dry_run else args.target, rng)
    if args.dry_run:
        jobs = jobs[: args.num]

    completed = []
    total_tokens = 0
    if not args.dry_run and Path(args.partial).exists():
        completed = json.loads(Path(args.partial).read_text(encoding="utf-8"))
        print(f"[RESUME] Loaded {len(completed)} records from {args.partial}")
        completed, fixed_existing, dropped_existing = clean_existing_records(completed)
        if fixed_existing or dropped_existing:
            print(f"[RESUME] Cleaned existing records: fixed={fixed_existing} dropped={dropped_existing}")
            Path(args.partial).write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8")

    fail_counts: Counter[str] = Counter()
    accepted_since_save = 0
    started_at = time.time()
    target_records = args.num if args.dry_run else args.target
    jobs_seen = 0

    if not args.dry_run and len(completed) >= args.target:
        print(f"[INFO] Already have {len(completed)} >= target {args.target}. Writing final files.")
        completed = save_records(completed[:args.target], args.output)
        write_stats(completed[:args.target], args.stats, total_tokens)
        write_dev_cases(args.dev_cases)
        return

    max_jobs = target_records if args.dry_run else max(args.target * args.max_jobs_factor, args.target + 100)
    pending_jobs = jobs[len(completed):] if args.dry_run else []

    def next_job() -> dict | None:
        nonlocal pending_jobs, jobs_seen
        if jobs_seen >= max_jobs:
            return None
        if not pending_jobs:
            if args.dry_run:
                return None
            remaining = max(1, target_records - len(completed))
            pending_jobs = build_jobs(remaining, rng)
        if not pending_jobs:
            return None
        jobs_seen += 1
        return pending_jobs.pop(0)

    if args.batch_api_size > 1 and not args.dry_run:
        with ThreadPoolExecutor(max_workers=max(1, args.api_workers)) as executor:
            futures = {}

            def submit_batch() -> bool:
                batch = []
                reserved = sum(len(existing) for existing in futures.values())
                while len(batch) < args.batch_api_size and len(completed) + reserved + len(batch) < target_records:
                    job = next_job()
                    if job is None:
                        break
                    batch.append(job)
                if not batch:
                    return False
                cats = ",".join(Counter(job["category"] for job in batch).keys())
                print(
                    progress_line(
                        "BATCH",
                        len(completed),
                        target_records,
                        len(completed),
                        sum(fail_counts.values()),
                        total_tokens,
                        started_at,
                        cats,
                    )
                    + f" size={len(batch)}",
                    flush=True,
                )
                futures[executor.submit(run_batch, batch, args)] = batch
                return True

            while len(completed) < target_records or futures:
                while len(futures) < max(1, args.api_workers) and len(completed) + sum(len(batch) for batch in futures.values()) < target_records:
                    if not submit_batch():
                        break
                if not futures:
                    break

                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    batch = futures.pop(future)
                    try:
                        records, tokens, batch_fail_counts = future.result()
                    except Exception:
                        records, tokens, batch_fail_counts = [], 0, Counter(job["category"] for job in batch)
                    total_tokens += tokens
                    for cat, count in batch_fail_counts.items():
                        fail_counts[cat] += count
                    for record, reason, _ in records:
                        completed.append(record)
                        accepted_since_save += 1
                        if reason == "fallback_reference":
                            print("[FALLBACK] Used reference answer inside batch", flush=True)
                    print(
                        progress_line(
                            "OK",
                            len(completed),
                            target_records,
                            len(completed),
                            sum(fail_counts.values()),
                            total_tokens,
                            started_at,
                        )
                        + f" batch_ok={len(records)}/{len(batch)} tokens~{tokens}",
                        flush=True,
                    )
                    if accepted_since_save >= args.save_every:
                        completed = save_records(completed, args.partial)
                        print(f"[SAVE] Partial written to {args.partial} ({len(completed)} records)", flush=True)
                        accepted_since_save = 0

                if jobs_seen >= max_jobs and len(completed) < target_records and not futures:
                    print(
                        f"[FATAL] Stopping after {jobs_seen} jobs with {len(completed)}/{target_records} accepted. "
                        "Check API output quality or increase --max-jobs-factor.",
                        flush=True,
                    )
                    break
    else:

        with ThreadPoolExecutor(max_workers=max(1, args.api_workers)) as executor:
            futures = {}
            while len(completed) < target_records or futures:
                while len(completed) + len(futures) < target_records and len(futures) < max(1, args.api_workers):
                    job = next_job()
                    if job is None:
                        break
                    print(
                        progress_line(
                            "START",
                            len(completed),
                            target_records,
                            len(completed),
                            sum(fail_counts.values()),
                            total_tokens,
                            started_at,
                            job["category"],
                        ),
                        flush=True,
                    )
                    futures[executor.submit(run_job, job, args)] = job

                if not futures:
                    break

                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    job = futures.pop(future)
                    try:
                        rec, tokens, reason, failed_category = future.result()
                    except Exception as e:
                        rec, tokens, reason, failed_category = None, 0, repr(e), job["category"]
                    total_tokens += tokens
                    if rec:
                        completed.append(rec)
                        accepted_since_save += 1
                        print(
                            progress_line(
                                "OK",
                                len(completed),
                                target_records,
                                len(completed),
                                sum(fail_counts.values()),
                                total_tokens,
                                started_at,
                                job["category"],
                            )
                            + f" last_tokens~{tokens}",
                            flush=True,
                        )
                        if reason == "fallback_reference":
                            print(f"[FALLBACK] Used reference answer for {job['category']} because API returned no parseable answer", flush=True)
                        if args.dry_run:
                            print("INPUT:", rec["input"][:160])
                            print("OUTPUT:", rec["output"][:300])
                    else:
                        fail_counts[failed_category or job["category"]] += 1
                        print(
                            progress_line(
                                "FAIL",
                                len(completed),
                                target_records,
                                len(completed),
                                sum(fail_counts.values()),
                                total_tokens,
                                started_at,
                                job["category"],
                            )
                            + f" reason={reason}",
                            flush=True,
                        )

                    if not args.dry_run and accepted_since_save >= args.save_every:
                        completed = save_records(completed, args.partial)
                        print(f"[SAVE] Partial written to {args.partial} ({len(completed)} records)", flush=True)
                        accepted_since_save = 0

                if jobs_seen >= max_jobs and len(completed) < target_records and not futures:
                    print(
                        f"[FATAL] Stopping after {jobs_seen} jobs with {len(completed)}/{target_records} accepted. "
                        "Check API output quality or increase --max-jobs-factor.",
                        flush=True,
                    )
                    break

    if args.dry_run:
        print(f"[DRY RUN DONE] accepted={len(completed)} failed={sum(fail_counts.values())} tokens~{total_tokens}")
        return

    completed = save_records(completed, args.output)
    completed = save_records(completed, args.partial)
    write_stats(completed, args.stats, total_tokens)
    write_dev_cases(args.dev_cases)
    if fail_counts:
        print(f"[WARN] Failures by category: {dict(fail_counts)}")
    print(f"[OK] Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-provider", choices=["openai"], default="openai")
    p.add_argument("--model", default="mimo-v2.5-pro")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    p.add_argument("--target", type=int, default=8000)
    p.add_argument("--output", default="data/student_notes_train_v3.json")
    p.add_argument("--partial", default="data/student_notes_train_v3_partial.json")
    p.add_argument("--stats", default="data/student_notes_train_v3_stats.txt")
    p.add_argument("--dev-cases", default="data/dev/judgment_rewrite_cases.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--quality-retries", type=int, default=3)
    p.add_argument("--api-workers", type=int, default=1,
                   help="Number of concurrent API requests for full generation")
    p.add_argument("--batch-api-size", type=int, default=1,
                   help="Generate this many records per API call in full generation")
    p.add_argument("--batch-max-tokens", type=int, default=5000,
                   help="max_tokens for batch API calls")
    p.add_argument("--api-timeout", type=int, default=120)
    p.add_argument("--api-http-retries", type=int, default=5)
    p.add_argument("--max-jobs-factor", type=int, default=5,
                   help="Full generation stops if attempted jobs exceed target * this value")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--num", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    try:
        generate(parse_args())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)
