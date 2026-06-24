# Open-source paper

This folder contains the electronic open-source version of the paper.

- `main.tex` is the LaTeX entry point.
- `paperstyle.sty` contains the electronic-reading layout.
- `content/` contains the paper text.
- `assets/figures/` contains only figures referenced by the paper.
- Cover pages, acknowledgements, declaration/authorization pages, and other undergraduate-thesis-only pages are omitted.

Build with:

```bash
latexmk -xelatex main.tex
```

The compiled `main.pdf` is intentionally not committed in this repository.
The public web version is hosted at <https://yexuhang.com/projects/leap-hand-inhand-rotation/paper/>.
