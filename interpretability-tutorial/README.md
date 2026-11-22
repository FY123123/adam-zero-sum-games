# Opening the Black Box: eight seminar sessions on mechanistic interpretability

Lecture notes for a reading seminar of eight two-hour sessions. They start from what a language model is and
cover the core of mechanistic interpretability, with an experiment for every idea. The compiled text is
`llm-interpretability-tutorial.pdf`.

- Build the PDF: `latexmk -pdf -jobname=llm-interpretability-tutorial main.tex`
- Run the experiments (no GPU needed): `pip install torch transformer_lens datasets matplotlib`, then for example
  `python code/exp_ioi.py`. Each script writes `code/results_*.json` and the figures in `figures/`.
  The table in Appendix A of the notes lists all scripts with their running times.
- `python code/exp_modadd.py --replot` repeats the analysis of the last session without retraining, if a
  finished run is present.

Tested with TransformerLens 4.4, PyTorch 2.15 and Python 3.13.
