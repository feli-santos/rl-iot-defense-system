# Minimal TeX Live image for compiling the FEEC/UNICAMP dissertation
# (template: FEEC modelo tese CCPG 001-2015, based on abnTeX2).
#
# Packages selected to exactly satisfy tex/main.tex + tex/preamble.tex +
# tex/unicamp.sty:
#   - texlive-latex-base       core LaTeX
#   - texlive-latex-recommended article, geometry, fancyhdr, setspace, ...
#   - texlive-latex-extra      memoir, multirow, rotating, pdfpages, subfig, ...
#   - texlive-fonts-extra      stix, lmodern, etc.
#   - texlive-publishers       abntex2, abntex2cite, abntex2-lipsum (REQUIRED)
#   - texlive-bibtex-extra     biber, biblatex, biblatex-abnt, logreq, ...
#   - biber                    biblatex backend (legacy; the template uses BibTeX via abntex2cite)
#   - texlive-lang-portuguese  babel Portuguese (template default)
#   - texlive-science          siunitx, mathtools, algorithm, algorithmic, ...
#   - texlive-pictures         caption, subfig deps
#   - texlive-plain-generic    xstring, etoolbox, ...
#   - lmodern                  Latin Modern fonts (template's default)
#
# Build once:  bash tex/build.sh --rebuild
# Then compile: bash tex/build.sh

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      # Core LaTeX
      texlive-latex-base \
      texlive-latex-recommended \
      texlive-latex-extra \
      # Fonts
      texlive-fonts-extra \
      texlive-fonts-recommended \
      lmodern \
      # Publishers (REQUIRED for abntex2 / abntex2cite)
      texlive-publishers \
      # Bibliography (biblatex + biblatex-abnt + biber; the template uses BibTeX/abntex2cite too)
      texlive-bibtex-extra \
      biber \
      # Language support
      texlive-lang-portuguese \
      texlive-lang-english \
      # Science / math (siunitx, mathtools, algorithm, algorithmic)
      texlive-science \
      # Figure / caption (subfig, caption, booktabs, epstopdf)
      texlive-pictures \
      # Misc generic packages (xstring, etoolbox, xpatch, glossaries ...)
      texlive-plain-generic \
      # PostScript / EPS conversion (epstopdf shell-out)
      ghostscript \
 && rm -rf /var/lib/apt/lists/* \
 && echo "TeX Live installation complete."

WORKDIR /work
