# Bundled panel-alignment helper

`audit_panel_alignment.py` measures the final rendered Matplotlib plot-area
rectangles and checks the declared comparable panel groups at 1.5 pt tolerance.
The renderer uses this bundled copy by default, so rebuilding the gallery does
not require a separate skills installation. `--qa-scripts` or
`NATURE_FIGURE_QA` can select another helper directory.

Source: [Yuan1z0825/nature-skills](https://github.com/Yuan1z0825/nature-skills),
commit `8630593a086755d2003cbd1a79cd1118c6963202`, file
`skills/nature-figure/scripts/audit_panel_alignment.py`. Git blob
`48f6f2b477b2b12cd4f83004d2e83aad9ac7e4b4` matches exactly; this copy is
unmodified. Authorship remains with the upstream contributors. The original
[Apache License 2.0](LICENSE.Apache-2.0) is retained, and the project license does
not replace this third-party license. No third-party reference figures or
plotting templates are copied.

Source preflight, PDF glyph-size and PDF collision checks are separate optional
post-export tools. Their final reports are already included under `figures/qa`.
To rerun them, point `scripts/audit_visual_evidence.py --qa-scripts` at an
installed nature-figure scripts directory and provide PyMuPDF for PDF inspection.
