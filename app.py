from cava.cava import run_cava
import gradio as gr

VERIFICATION_MODE = "every_k" # "none" | "every" | "every_k"
VERIFICATION_K = 3

# Wrapper function to call cava
def run_agent(question, context):
    """
    question: user input question string
    context: user input context string (can be long)
    """
    final_answer = run_cava(query=question, context=context, verbose=True, verification_mode=VERIFICATION_MODE, verification_k=VERIFICATION_K, store_verification_traces=True)

    return final_answer


# ======================================================
# Gradio UI
# ======================================================

with gr.Blocks() as cava_demo:
    gr.Markdown("# CAVA: **C**hain of **A**gent with chain of **V**erific**A**tion Demo\nEnter a question and context below.")

    with gr.Row():
        question_box = gr.Textbox(
            label="Question",
            placeholder="Enter your question here",
            lines=3
        )

    with gr.Row():
        context_box = gr.Textbox(
            label="Context",
            placeholder="Enter supporting context here",
            lines=12
        )

    run_button = gr.Button("Run Agent")
    output_box = gr.Textbox(
        label="Agent Output",
        lines=10
    )

    run_button.click(
        fn=run_agent,
        inputs=[question_box, context_box],
        outputs=output_box
    )

cava_demo.launch()
