import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.runnables import RunnableLambda
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
from cava.states import VerificationTrace, CoAState
from cava.prompts import WORKER_PROMPT, MANAGER_PROMPT, PLAN_VERIFICATIONS_PROMPT, EXEC_VERIFICATIONS_PROMPT, GEN_FINAL_RESPONSE_PROMPT, EXTRACT_ANSWER_PROMPT
import re
from typing import List
# One time config, you can also move this outside the function
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

MAX_NEW_TOKENS = 128
CHUNK_SIZE = 2000

# LOCAL_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

MODEL_LOGS = []

# Disable grad for torch
torch.set_grad_enabled(False)

def load_local_llm(model_id, max_new_tokens=MAX_NEW_TOKENS):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        # torch_dtype=torch.float16,
        dtype=torch.float16,
    )
    model.eval()
    def _generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        # slice off prompt tokens
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()

    return RunnableLambda(lambda x: _generate(x))

def load_google_llm(
    model_name: str,
    max_new_tokens: int = 256,   # or MAX_NEW_TOKENS if you have it defined
):
    """
    Wrap a Google AI Studio model (Gemini) as a LangChain Runnable that you can call with .invoke(prompt).
    """

    model = genai.GenerativeModel(model_name)

    def _generate(prompt: str) -> str:
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_new_tokens,
                "temperature": 0.0,   # deterministic for QA
                "top_p": 1.0,
            },
        )
        # response.text is already the concatenated text of all parts
        return (response.text or "").strip()

    return RunnableLambda(lambda x: _generate(x))



# Loading text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=int(CHUNK_SIZE*0.1),
    separators=["\n\n", "\n", ". ", " "]
)

# Loading model
# llm_strong = load_local_llm(LOCAL_MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
llm_strong = load_google_llm(model_name=GEMINI_MODEL_NAME,max_new_tokens=MAX_NEW_TOKENS)

worker = llm_strong
manager = llm_strong
verifier = llm_strong
extractor = llm_strong

def worker_node(state: CoAState):
    i = state["i"]
    chunk = state["chunks"][i]
    if i == 0:
        prev = "No Previous summaries"
    else:
        # Get previous worker's output
        # print(state["worker_outputs"][i-1])
        # prev = state["worker_outputs"][i-1].content
        prev = state["worker_outputs"][i-1]
    prompt = WORKER_PROMPT(i, state["query"], chunk, prev)
    worker_msg = f"Worker {i} with Prompt: \n######{prompt}\n#######\n"
    MODEL_LOGS.append(worker_msg)
    if state["verbose"]:
        print(f"Worker {i} with Prompt: \n######{prompt}\n#######\n")
        print("worker invoke")
    out = worker.invoke(prompt)

    if state["verbose"]:
        print("worker invoke -- done")
    # Note new outut
    state["worker_outputs"].append(out)
    state["i"] += 1
    worker_output = f"Worker {i} Outputs: \n{out}\n"
    MODEL_LOGS.append(worker_output)
    if state["verbose"]:
        # print(f"Outputs: {out.content}\n------------------\n\n")
        print(f"Outputs: {out}\n------------------\n\n")

    return state

def manager_node(state:CoAState):
    if state["verbose"]:
        state["worker_outputs"][-1]
    # last_worker_output = state["worker_outputs"][-1].content
    last_worker_output = state["worker_outputs"][-1]
    prompt = MANAGER_PROMPT(state["query"], last_worker_output)
    manager_prompt = f"Manager with Prompt: \n######{prompt}\n#######\n"
    MODEL_LOGS.append(manager_prompt)
    if state["verbose"]:
        print(f"Manager with Prompt: \n######{prompt}\n#######\n")
    final_answer = manager.invoke(prompt)
    # store final summary as last output
    state["manager_output"] = final_answer
    manager_output = f"Manager Final Output: \n#############\n{final_answer}"
    MODEL_LOGS.append(manager_output)
    if state["verbose"]:
        # print(f"Manager Final Output: \n#############\n{final_answer.content}")
        print(f"Manager Final Output: \n#############\n{final_answer}")

    return state

def parse_numbered_answers(exec_text: str, num_questions: int) -> List[str]:
    """
    Parse a numbered list like:
        1. Yes
        2) No
        3 - Maybe
    into ["Yes", "No", "Maybe"], capped at num_questions.

    Behavior:
    - Ignore ALL lines until the first line beginning with a number.
    - After that, parse consecutive numbered answers.
    - Stop after num_questions items.
    - Handle lines like "4. Yes 1. Yes..." by keeping only the first segment.
    """

    answers: List[str] = []
    started = False  # track when we hit first numbered line

    for line in exec_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Detect first numbered line
        if not started:
            if line[0].isdigit():
                started = True
            else:
                continue  # skip until list starts

        # From here on, only accept numbered-list lines
        if not line[0].isdigit():
            continue

        # Strip the leading number (1., 1), 1 -, etc.)
        cleaned = re.sub(r"^\d+\s*[\.\)\-]\s*", "", line).strip()

        # Remove any second inline numbering (avoid "Yes 1. Yes, ..."), keep only the part before it
        parts = re.split(r"\s+\d+\s*[\.\)\-]\s*", cleaned)
        cleaned = parts[0].strip()

        if cleaned:
            answers.append(cleaned)
            if len(answers) >= num_questions:
                break

    # Fallback if no valid parsed answers
    if not answers:
        return [exec_text.strip()]

    return answers


def run_cove(query: str, chunk: str, baseline_summary: str, worker_idx: int, verbose: bool = False) -> VerificationTrace:
    # 1. Baseline response = baseline_summary (already produced by worker)
    # 2. Plan verification questions
    plan_prompt = PLAN_VERIFICATIONS_PROMPT(query, chunk, baseline_summary)

    cove_plan = f"[CoVe][Worker {worker_idx}] Plan prompt:\n{plan_prompt}\n"
    MODEL_LOGS.append(cove_plan)
    if verbose:
        print(f"[CoVe][Worker {worker_idx}] Plan prompt:\n{plan_prompt}\n")

    plan_resp = verifier.invoke(plan_prompt)
    plan_text = str(getattr(plan_resp, "content", plan_resp)) # Depending on the LLM wrapper, llm.invoke() may return a plain string or a message object AIMessage(..., content="some text", ...)

    # crude parsing: split into lines that look like questions
    questions = [
        line.strip(" -0123456789.").strip()
        for line in plan_text.split("\n")
        if "?" in line
    ]
    questions = [q for q in questions if q]

    cove_questions = f"[CoVe][Worker {worker_idx}] Verification Questions:\n{questions}\n"
    MODEL_LOGS.append(cove_questions)
    if verbose:
        print(f"[CoVe][Worker {worker_idx}] Verification Questions:\n{questions}\n")

    # 3. Execute verifications (factored: one call per question)
    answers = []
    qa_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    exec_prompt = EXEC_VERIFICATIONS_PROMPT(query, chunk, qa_block)
    exec_resp = verifier.invoke(exec_prompt)
    exec_text = str(getattr(exec_resp, "content", exec_resp)).strip() #

    if verbose:
        print(f"verification q answer: {exec_text}")

    # answer parsing
    parsed_answers = parse_numbered_answers(exec_text, num_questions=len(questions))

    if len(parsed_answers) != len(questions):
        questions_for_gen = ["\n".join(questions)]
        answers_for_gen = [exec_text]
    else:
        questions_for_gen = questions
        answers_for_gen = parsed_answers

    cove_ans = f"[CoVe][Worker {worker_idx}] Answers:\n{answers_for_gen}\n"
    MODEL_LOGS.append(cove_ans)
    if verbose:
        print(f"[CoVe][Worker {worker_idx}] Answers:\n{answers_for_gen}\n")

    # 4. Generate final verified summary
    final_prompt = GEN_FINAL_RESPONSE_PROMPT(query, chunk, baseline_summary, questions_for_gen, answers_for_gen)

    cove_final_prompt = f"[CoVe][Worker {worker_idx}] final_prompt: {final_prompt}\n"
    MODEL_LOGS.append(cove_final_prompt)
    
    if verbose:
        print(f"final_prompt: {final_prompt}")
    
    final_resp = verifier.invoke(final_prompt)
    final_summary = str(getattr(final_resp, "content", final_resp)).strip() # [TODO] add in worker node? Depending on the LLM wrapper, llm.invoke() may return a plain string or a message object AIMessage(..., content="some text", ...)

    cove_final = f"[CoVe][Worker {worker_idx}] Final verified summary:\n{final_summary}\n"
    MODEL_LOGS.append(cove_final)
    if verbose:
        print(f"[CoVe][Worker {worker_idx}] Final verified summary:\n{final_summary}\n")

    trace: VerificationTrace = {
        "worker_idx": worker_idx,
        "baseline_summary": baseline_summary,
        "verification_questions": questions,
        "verification_answers": answers,
        "verified_summary": final_summary,
    }
    return trace


def verification_node(state: CoAState, worker_idx: int):
    query = state["query"]
    chunk = state["chunks"][worker_idx]

    raw_summary = state["worker_outputs"][worker_idx]
    baseline_summary = str(getattr(raw_summary, "content", raw_summary)) #

    trace = run_cove(
        query=query,
        chunk=chunk,
        baseline_summary=baseline_summary,
        worker_idx=worker_idx,
        verbose=state["verbose"],
    )

    # replace worker summary with verified one
    state["worker_outputs"][worker_idx] = trace["verified_summary"]

    # store trace if needed
    if state.get("store_verification_traces", False): # defaults to False if the key doesn’t exist
        state["verification_traces"].append(trace)

    return state


def maybe_run_verification(state: CoAState) -> CoAState:
    """
    determine if the latest generated summary needs to be verified

    Param:
    state (returned by worker_node())

    Return:
    updated state
    """
    mode = state["verification_mode"] # "none" | "every" | "every_k"
    k = state["verification_k"]
    current_worker_idx = state["i"] - 1 # cuz in worker_node() before returning state it does state["i"] += 1

    if mode == "none":
        return state
    if mode == "every":
        return verification_node(state, current_worker_idx)
    if mode == "every_k" and (current_worker_idx + 1) % k == 0:
        return verification_node(state, current_worker_idx)

    return state

def run_cava(query, context, verbose=True, verification_mode="none", verification_k=1, store_verification_traces=True, postprocess=True): # chunk_size
    # Split context
    # chunks = split_text(context, chunk_size=chunk_size)
    chunks = splitter.split_text(context)
    if verbose:
        print("Text Chunks: ",chunks)
    # assert 1==2
    # Initialize initial CoAState
    init_state = {
        "query": query,
        "chunks": chunks,
        "i": 0,
        "worker_outputs": [],
        "verbose": verbose,
        "manager_output": "",
        # [CoVe]
        "verification_mode": verification_mode,
        "verification_k": verification_k,
        "store_verification_traces": store_verification_traces,
        "verification_traces": []
    }
    state = init_state
    length = len(chunks)
    if verbose:
        print("Num Chunks: ", length)
    # Worker nodes, for each chunk
    for i, chunk in enumerate(chunks):
        # Run worker node and get new state
        if verbose:
            print(f"Running Worker {i}")
        state = worker_node(state)
        if verbose:
            print(f"Running Worker {i} -- Done")
        # [TODO]
        if verbose:
            print(f"Verifying Worker {i}")
        state = maybe_run_verification(state)
        if verbose:
            print(f"Verifying Worker {i} -- Done")

    # At the end of the loop, state["i"] should be == len(chunks)
    assert state["i"] == len(chunks), "Total states worked does not equal to number of text chunks"

    # Finally run manager at last
    if verbose:
        print(f"Manager producing output")
    state = manager_node(state)
    # final_ans = state["worker_outputs"][-1].content
    final_ans = state["manager_output"]
    if verbose:
        print("Final Answer before process: ", final_ans)
    if "Final Answer: ".lower() in final_ans.lower():
        if verbose:
            print("splitting parsing")
        final_ans = final_ans.lower().split("Final answer: ".lower())[-1]
    if verbose:
        print(f"Manager producing output -- Done")

    # Post processing
    if postprocess and False:
        if verbose:
            print(f"Extractor")
        prompt = EXTRACT_ANSWER_PROMPT(query, state["manager_output"])
        resp = extractor.invoke(prompt)
        final_ans = str(getattr(resp, "content", resp)).strip()
        if verbose:
            print(f"Extractor -- Done")

    final_ans = f"Query: {state['query']}\nFinal Answer: {final_ans}"
    MODEL_LOGS.append(final_ans)
    if verbose:
        print(f"Query: {state['query']}\nFinal Answer: {final_ans}")
    return final_ans