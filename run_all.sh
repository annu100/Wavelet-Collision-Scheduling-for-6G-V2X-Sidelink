#!/usr/bin/env bash
# Reproduces every number and figure of the letter and the supplementary material (~80 min, one CPU core).
set -e
cd "$(dirname "$0")"
python3 experiments/exp_theory.py
python3 experiments/run_3gpp.py main140 main70 sweep speed density pen aoi nonstat sens
python3 experiments/make_figures_sl.py
cd paper
B(){ pdflatex -interaction=nonstopmode "$@" > /dev/null; }
B main.tex; bibtex main; B main.tex; B main.tex
B -jobname=main_1col "\def\ONECOL{1}\input{main}"; bibtex main_1col
B -jobname=main_1col "\def\ONECOL{1}\input{main}"; B -jobname=main_1col "\def\ONECOL{1}\input{main}"
B supplement.tex; bibtex supplement; B supplement.tex; B supplement.tex
B cover_letter.tex
