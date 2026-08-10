#!/usr/bin/env bash
# Rebuild the report and strip document metadata from the public PDF.
# Requires either TeX Live (xelatex and bibtex) or Tectonic, plus uv.
set -euo pipefail
cd "$(dirname "$0")"

if command -v xelatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  xelatex -interaction=nonstopmode main.tex
  bibtex main
  xelatex -interaction=nonstopmode main.tex
  xelatex -interaction=nonstopmode main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic -X compile main.tex --keep-intermediates --keep-logs
else
  echo "error: install TeX Live (xelatex and bibtex) or Tectonic" >&2
  exit 1
fi

uv run --no-project --python 3.12.10 --with pypdf \
  python ../../../tools/sanitize_report_pdf.py \
  main.pdf ../GOAI_virtual_cell_preliminary_report.pdf

echo "----- report build -----"
echo -n "pages: "
grep -a "Output written" main.log | sed 's/.*(\([0-9]*\) pages).*/\1/'
echo -n "undefined citations: "
grep -ac "Citation.*undefined" main.log || true
echo -n "undefined references: "
grep -ac "Reference.*undefined" main.log || true
echo -n "missing glyphs: "
grep -ac "Missing character" main.log || true
echo -n "overfull boxes (>20pt): "
grep -a "Overfull \\\\hbox" main.log | awk -F'[()]' '{print $2}' | awk '$1>20' | wc -l
echo "output: ../GOAI_virtual_cell_preliminary_report.pdf"
