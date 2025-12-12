# CAVA
Chain of Agent and Verification

Yueh-An Liao, Ray Wen

## Step by Step Tutorial

### File descriptions

Our main file is `CAVA.ipynb`. It is a notebook containing all the code and set up needed to build our system and run our experiments.

### Installation

The first block of the notebook contains all the library installations we may need to run this notebook. You can simply run that cell in order to install.

It will connect to your Google Drive, if you are running locally, please comment out:
```
from google.colab import drive
drive.mount('/content/drive')
```

### Set up

The log files are saved to the location specified in `LOG_PATH`.

Text splitter chunk size is specified in `CHUNK_SIZE`

### Experiments

#### Verification settings

To run CAVA with verification on every worker, please set `VERIFICATION_MODE="every"`.

To run CAVA with verification on every K worker, please set `VERIFICATION_MODE="every_k"` and set `VERIFICATION_K = K`, where K is how many workers in between each verification step.

To run CoA, please set `VERIFICATION_MODE="none"`.

To run Full Context Baseline, please make the following changes

```
# Comment out this line
# final_ans = run_coa(query=question, context=merged_context, verbose=False, verification_mode=VERIFICATION_MODE, verification_k=VERIFICATION_K, store_verification_traces=True)

# Uncomment line below
final_ans = raw_model(question, merged_context)
```

#### Dataset loading

We provide two dataset loaders depending on the experimental setting:

Random subset from validation split

```
data = load_hotpotqa(split="validation", max_samples=NUM_SAMPLES_TO_LOAD)
```

Balanced subset by difficulty level

```
data = load_hotpotqa_balanced(split="train", per_level=NUM_SAMPLES_TO_LOAD_PER_LEVEL, levels=LEVELS, seed=SEED)
```


### Running the file

After you have modified the setting mentioned above, you can simply run all the cells in the notebook and it will run the HotpotQA Benchmark.

It will save an output log to your specified path. You can also print the results variable below to see the metrics obtained.


## Analysis

`Analysis.ipynb` aggregates per-example JSONL logs from HotpotQA runs and produces:
- Overall mean F1 and Exact Match (EM) per method
- Subgroup breakdowns by question type (bridge/comparison) and level (easy/medium/hard)
- Paired t-tests between methods
- Plots used in the report/slides
- Candidate examples for qualitative analysis (e.g., largest ΔF1 between two CoA and CAVA_every3)

### Required Data 

JSONL logs with (at minimum) fields like:
- `id`
    Unique identifier of the example from the HotpotQA dataset.
- `idx`
    Integer index of the example within the evaluated subset (for bookkeeping and reproducibility).
- `type`
    Question type: bridge or comparison
- `level`
    Difficulty level: easy, medium, or hard
- `question`
    The natural-language question given to the model.
- `gold_answer`
    Ground-truth answer from the HotpotQA dataset.
- `prediction`
    Final answer produced by the model.
- `f1`
    Token-level F1 score between the prediction and the gold answer, computed after standard normalization (lowercasing, punctuation removal, and article stripping).
- `em`
    Exact Match score (0 or 1), indicating whether the normalized prediction exactly matches the normalized gold answer.

### Configure log paths

You can provide one file or a list of files per method:

```
LOG_FILES = {
  "CoA": [
    "/log_dir/.../coa_easy.jsonl",
    "/log_dir/.../coa_medium.jsonl",
  ],
  "CAVA_every3": "/log_dir/.../cava_every3.jsonl",
  "full_context": "/log_dir/.../full_context.jsonl",
  ...
}
```
All files under the same method will be concatenated and deduplicated by (id, idx, method)

### Analysis Results

The notebook prints summary tables and shows plots.

Use the “Qualitative / error analysis” section to export a shortlist of example IDs to re-run with traces.


## Demo

### Demo Installation

In addition to the packages listed in the notebook, please install `gradio` using the command: `pip install gradio`. 

### API Key
The current demo code is using Google Gemini's API, thus you must set a Google API Key in your environment. You can do so by running

```
Mac / Linux
export GOOGLE_API_KEY=<your api key>

Windows
$env:GOOGLE_API_KEY="<your api key>"
```

Or if you would like to run huggingface models locally, you may do so by modifying the file in `cava/cava.py`

```
LOCAL_MODEL_NAME="model name"
...
# Uncomment this line
llm_strong = load_local_llm(LOCAL_MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
...
# Comment out this line 
llm_strong = load_google_llm(model_name=GEMINI_MODEL_NAME,max_new_tokens=MAX_NEW_TOKENS)

```

### Run Demo

After you have made those changes, you can simply run this command to launch your gradio app. 
```
python app.py
```
