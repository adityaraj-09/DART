# DART arXiv-style preprint
.PHONY: all pdf clean

all: pdf

pdf: dart_idd.pdf

dart_idd.pdf: dart_idd.tex references.bib arxiv.sty
	latexmk -pdf -interaction=nonstopmode -halt-on-error dart_idd.tex

clean:
	latexmk -C
	rm -f dart_idd.bbl dart_idd.synctex.gz
