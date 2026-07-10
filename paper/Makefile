# paper/Makefile — build the Elsevier "Internet of Things" manuscript.
#
# Reuses the existing thesis container image (rl-iot-thesis), which already
# ships elsarticle.cls + elsarticle-num.bst. No host TeX install required.
#
# Usage (from repo root or this dir):
#   make -C paper build      # full build: pdflatex -> bibtex -> pdflatex x2
#   make -C paper numbers    # refresh numbers.tex from docs/results/
#   make -C paper draft      # single fast pdflatex pass (no bibtex)
#   make -C paper wordcount  # approximate manuscript word count
#   make -C paper clean      # remove build aux files
#   make -C paper verify     # build + assert manuscript.pdf exists

SHELL       := /bin/bash
PAPER_DIR   := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
REPO_ROOT   := $(abspath $(PAPER_DIR)/..)
IMAGE       ?= rl-iot-thesis
MAIN        := manuscript
PYTHON      ?= $(REPO_ROOT)/.venv/bin/python

# Container engine: podman preferred, docker fallback (matches tex/build.sh).
ENGINE := $(shell command -v podman >/dev/null 2>&1 && echo podman || echo docker)
RUN    := $(ENGINE) run --rm -v "$(PAPER_DIR)":/work -w /work --entrypoint "" "$(IMAGE)"

.PHONY: build draft numbers wordcount clean verify

## numbers: regenerate numbers.tex (macro-driven) from docs/results/ JSONs.
numbers:
	cd "$(REPO_ROOT)" && PYTHONPATH=. $(PYTHON) -m scripts.thesis.render_tables
	cp "$(REPO_ROOT)/tex/generated/numbers.tex" "$(PAPER_DIR)/numbers.tex"
	@echo "==> numbers.tex refreshed in paper/"

## build: full manuscript compile (numbers first, then LaTeX + BibTeX).
build: numbers
	$(RUN) pdflatex -interaction=nonstopmode -file-line-error $(MAIN).tex
	$(RUN) bibtex $(MAIN) || true
	$(RUN) pdflatex -interaction=nonstopmode -file-line-error $(MAIN).tex
	$(RUN) pdflatex -interaction=nonstopmode -file-line-error $(MAIN).tex
	@mkdir -p "$(PAPER_DIR)/build"
	@cp "$(PAPER_DIR)/$(MAIN).pdf" "$(PAPER_DIR)/build/$(MAIN).pdf" 2>/dev/null || true
	@echo "==> Built $(MAIN).pdf"

## draft: single fast pass (no bibliography), for quick iteration.
draft:
	$(RUN) pdflatex -interaction=nonstopmode -file-line-error $(MAIN).tex
	@echo "==> Draft $(MAIN).pdf (references may be unresolved)"

## wordcount: approximate body word count (strips LaTeX via detex if present).
wordcount:
	$(RUN) bash -c "detex $(MAIN).tex 2>/dev/null | wc -w || texcount -inc -total $(MAIN).tex"

## verify: build and assert the PDF exists.
verify: build
	@test -f "$(PAPER_DIR)/$(MAIN).pdf" && echo "==> OK: $(MAIN).pdf present" \
		|| { echo "==> FAIL: $(MAIN).pdf missing"; exit 1; }

## clean: remove LaTeX aux files (keep the PDF and sources).
clean:
	@rm -f "$(PAPER_DIR)"/$(MAIN).{aux,bbl,blg,log,out,toc,lof,lot,brf,spl}
	@echo "==> cleaned aux files"
