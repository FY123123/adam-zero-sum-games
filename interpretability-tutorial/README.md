# Opening the Black Box: a first course in LLM interpretability

A tutorial of about 40 pages. The compiled text is `llm-interpretability-tutorial.pdf`.

- Build the PDF: `latexmk -pdf -jobname=llm-interpretability-tutorial main.tex`
- Regenerate the GPT-2 small figures and numbers (about 4 minutes on a laptop CPU):
  `pip install torch transformer_lens matplotlib && python code/gpt2_examples.py`.
  The numbers quoted in the text are stored in `code/results.json`.
- `code/minimal_patching.py` is the listing printed in Section 9.

Tested with TransformerLens 4.4, PyTorch 2.15 and Python 3.13.
