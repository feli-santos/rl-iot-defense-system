# Journal paper: *Internet of Things* (Elsevier) submission

Condensed manuscript derived from the MSc dissertation (`tex/main.pdf`).
Target venue: **Elsevier *Internet of Things*** (ISSN 2542-6605).
Template: **elsarticle**, double-column (`\documentclass[3p,twocolumn]`).

## Build

No host TeX install is needed; the build reuses the thesis container image
`rl-iot-thesis` (which already ships `elsarticle.cls` + `elsarticle-num.bst`).

```bash
make -C paper build      # numbers -> pdflatex -> bibtex -> pdflatex x2
make -C paper draft      # quick single pass
make -C paper wordcount  # approximate word count
make -C paper clean      # remove aux files
```

Output: `paper/manuscript.pdf` (also copied to `paper/build/`).

## Length: 10 pages (HARD LIMIT)

**The compiled `manuscript.pdf` must be at most 10 pages.** This is a hard
limit — do not exceed it. When edits push the build over 10 pages, reclaim
space by trimming filler prose and/or stripping redundant `doi = {...}` fields
from `refs.bib` (keep every reference and every reported number). Do **not**
drop cited references to fit. Verify after every build, e.g.:

```bash
make -C paper build
python -c "import fitz; print(fitz.open('paper/manuscript.pdf').page_count)"  # must be <= 10
```

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
    data-availability.md        # Option-C statement + pre-submission checklist
  figs/                   # vector PDF figures reused from tex/figs/
  Makefile
  build/                  # gitignored build output
```

## Numbers are macro-driven

Every reported number resolves through `numbers.tex`, regenerated from the
canonical experiment JSONs under `docs/results/`. Never hand-type a result;
run `make -C paper numbers` to refresh, then rebuild.

## Pre-submission checklist (Elsevier *Internet of Things* guide)

- [x] Title page with full affiliation + corresponding author contact — in `manuscript.tex`.
- [x] Compiled PDF <= 10 pages — **hard limit** (see "Length" above).
- [x] Abstract <= 250 words — done (~242).
- [x] Keywords 1-7, short indexing terms — done (7).
- [x] Highlights file (3-5 bullets, <=85 chars) — `highlights.tex`.
- [x] CRediT statement — in manuscript + `declarations/credit-statement.md`.
- [x] Declaration of generative-AI use (section before references) — in manuscript.
- [x] Declaration of competing interest (separate .docx) — `declarations/competing-interests.docx`.
- [x] Funding statement — in manuscript + `declarations/funding.md`.
- [x] Data-availability statement (Option C) — in manuscript + checklist in `declarations/data-availability.md`.
- [x] Acknowledgements directly before references — in manuscript.
- [x] Editable source (.tex) supplied; figures as separate files — `figs/`.
- [ ] Replace Zenodo DOI placeholder (`10.5281/zenodo.XXXXXXX`) and confirm GitHub repo is public.
- [x] Author emails filled (`f233292@dac.unicamp.br`, `denisf@unicamp.br`).
- [ ] Optional: graphical abstract, SSRN preprint, MethodsX/Data-in-Brief co-submission.
```
