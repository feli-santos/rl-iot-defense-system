# Journal paper: *Internet of Things* (Elsevier) submission

Condensed manuscript derived from the MSc dissertation (`tex/main.pdf`).
Target venue: **Elsevier *Internet of Things*** (ISSN 2542-6605).
Template: **elsarticle**, double-column (`\documentclass[3p,twocolumn]`).

> **Status:** submitted; the submission snapshot is published as GitHub release
> [`v0.8.4`](https://github.com/feli-santos/rl-iot-defense-system/releases/tag/v0.8.4)
> (cited in the manuscript's Data-availability statement).

## Build

No host TeX install is needed; the build reuses the thesis container image
`rl-iot-thesis` (which already ships `elsarticle.cls` + `elsarticle-num.bst`).

```bash
make -C paper build      # numbers -> pdflatex -> bibtex -> pdflatex x2
make -C paper draft      # quick single pass
make -C paper numbers    # regenerate numbers.tex from docs/results/ JSONs
make -C paper wordcount  # approximate word count
make -C paper clean      # remove aux files
```

Output: `paper/manuscript.pdf` (also copied to `paper/build/`).

## Layout

```
paper/
  manuscript.tex          # the paper (elsarticle, double-column)
  numbers.tex             # macro-driven numbers (copy of tex/generated/numbers.tex)
  refs.bib                # bibliography (superset of cited entries)
  highlights.tex          # 3-5 highlight bullets (<=85 chars each)
  cover-letter.md         # editor cover letter
  declarations/
    credit-statement.md         # CRediT roles
    competing-interests.docx    # separate Word declaration for submission
    funding.md                  # funding statement
    genai-declaration.md        # generative-AI-use disclosure
    data-availability.md        # Option-C statement
  figs/                   # vector PDF figures reused from tex/figs/
  Makefile
  build/                  # gitignored build output
```

## Numbers are macro-driven

Every reported number resolves through `numbers.tex`, regenerated from the
canonical experiment JSONs under `docs/results/`. Never hand-type a result;
run `make -C paper numbers` to refresh, then rebuild.

> Contributor note: submission constraints (10-page hard limit, abstract
> word ceiling, float-placement convention, footprint-macro provenance) and
> the Elsevier pre-submission checklist live in the repo-root `AGENTS.md`
> under "Journal paper", not here.
