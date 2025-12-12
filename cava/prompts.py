# --- Prompts ------------------------------------------------------------------

WORKER_PROMPT = lambda i, query, chunk, prev: f"""
You are Worker {i} in a chain solving a long-context question answering task.

Use ONLY:
- the current source text (chunk)
- the previous worker summary

Task:
- Write a new summary that combines:
  (a) all information from the previous summary that is relevant to the query, and
  (b) any new relevant information in the current chunk.
- If the current chunk adds no new relevant information, simply repeat the previous summary unchanged.

Constraints:
- Maximum length: about 300 tokens.
- Output only the new summary, no commentary about your process.

Query:
{query}

Current source text (CHUNK {i}):
{chunk}

Previous worker summary:
{prev}

Now output the combined summary:
"""


MANAGER_PROMPT = lambda query, final_worker_json: f"""
You are the Manager in a HotpotQA question answering system.

Task:
- Read the summary of evidence.
- Reason briefly about the answer.
- Then output the final answer as a short span, try to find the closest answer.

Output format (very important):
1. First, write a short reasoning paragraph if needed.
2. On the LAST line of your response, write exactly:

   Final answer: <answer>

Rules for <answer>:
- Use the shortest possible span (a name, location, date, number, or "yes"/"no").
- For yes/no questions, answer exactly "yes" or "no".
- Do NOT add any text after <answer> on that line.
- Do NOT write anything after the "Final answer: ..." line (no notes, no extra sentences).

Query:
{query}

Summary of evidence:
{final_worker_json}
"""

# ===== CoVe =====
PLAN_VERIFICATIONS_PROMPT = lambda query, chunk, baseline_summary: f"""
You are verifying a summary used in a long-context QA pipeline.

Original Query: {query}

Source chunk: {chunk}

Baseline summary: {baseline_summary}

Task:
Generate a small list of concrete verification questions (2–4) that help check:
- factual correctness
- coverage of key information relevant to the query
- absence of unsupported claims
Return the verification questions as a numbered list.
"""


EXEC_VERIFICATIONS_PROMPT = lambda query, chunk, qa_block: f"""
You are answering verification questions about a summary for a long-context QA pipeline.

Original Query: {query}

Source chunk: {chunk}

Here is a list of verification questions:
{qa_block}

For each question, answer concisely.
Formatting rules (very important):
- Return your answers as a **single numbered list**.
- Use exactly one line per answer.
- Do NOT repeat the list.
- Do NOT restate the questions.
- The format must be:

  1. <answer to Q1>
  2. <answer to Q2>
  3. <answer to Q3>
  ...
"""


GEN_FINAL_RESPONSE_PROMPT = lambda query, chunk, baseline_summary, questions, answers: f"""
You are revising a summary for a long-context QA pipeline.

Original Query: {query}

Source chunk: {chunk}

Baseline summary: {baseline_summary}

Verification Q&A:
{chr(10).join(f"Q: {q} A: {a}" for q, a in zip(questions, answers))}

Task:
Write a revised summary that:
- corrects any factual errors in the baseline summary
- adds missing key information supported by the source chunk
- removes unsupported or speculative claims
- remains concise and focused on information relevant to the question

Return ONLY the revised summary.
"""


EXTRACT_ANSWER_PROMPT = lambda query, manager_output: f"""
You are post-processing the output of a QA system on the HotpotQA dataset.

Your task: extract the **final answer string** that should be evaluated against the gold answer.

Constraints (very important):
- Return **only** the minimal answer span.
- Do **not** include explanations, reasoning, or extra words.
- Do **not** include phrases like "The answer is", "It is", "Final answer", etc.
- Do **not** add punctuation at the beginning or end unless it is part of the entity (e.g., "U.S.").
- Do **not** output multiple sentences.
- If the question is yes/no, answer with exactly **yes** or **no**.
- If the model’s answer is clearly wrong or missing, output exactly **no answer**.

Output format:
- Your entire response must be **only** the answer string, with no quotation marks and no additional text.

Query:
{query}

Model's answer:
{manager_output}

Now output the answer string only:
"""
