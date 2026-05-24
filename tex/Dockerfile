# Minimal TeX Live image for compiling the rl-iot-defense-system thesis.
#
# Packages selected to exactly satisfy tex/thesis.cls + tex/thesis.tex:
#   - texlive-latex-base       core LaTeX
#   - texlive-latex-recommended article, geometry, fancyhdr, setspace, …
#   - texlive-latex-extra      silence, subfig, enumitem, footmisc, …
#   - texlive-fonts-extra      libertine, libertinust1math, stix, nomencl fonts
#   - texlive-bibtex-extra     biblatex, biblatex-abnt, logreq, …
#   - biber                    biblatex backend
#   - texlive-lang-portuguese  babel Portuguese (thesis option)
#   - texlive-science          siunitx, mathtools, …
#   - texlive-pictures         caption, subfig deps
#   - texlive-plain-generic    xstring, etoolbox, …
#   - lmodern                  Latin Modern fonts fallback
#
# Build once:  docker build -t rl-iot-thesis -f tex/Dockerfile tex/
# Then compile: docker run --rm -v "$PWD/tex":/work rl-iot-thesis
#          (or just: make thesis)

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      # Core LaTeX
      texlive-latex-base \
      texlive-latex-recommended \
      texlive-latex-extra \
      # Fonts (libertine + libertinust1math + stix + nomencl)
      texlive-fonts-extra \
      texlive-fonts-recommended \
      lmodern \
      # Bibliography (biblatex + biblatex-abnt + biber)
      texlive-bibtex-extra \
      biber \
      # Language support
      texlive-lang-portuguese \
      # Science / math (siunitx, mathtools)
      texlive-science \
      # Figure / caption (subfig, caption, booktabs)
      texlive-pictures \
      # Misc generic packages (xstring, etoolbox, xpatch …)
      texlive-plain-generic \
 && rm -rf /var/lib/apt/lists/* \
 && echo "TeX Live installation complete."

WORKDIR /work

# Default command: full 3-pass pdflatex + biber compile
CMD ["bash", "-c", \
  "set -e && \
   pdflatex -interaction=nonstopmode thesis.tex && \
   biber thesis && \
   pdflatex -interaction=nonstopmode thesis.tex && \
   pdflatex -interaction=nonstopmode thesis.tex && \
   echo '==> thesis.pdf compiled successfully.'"]
