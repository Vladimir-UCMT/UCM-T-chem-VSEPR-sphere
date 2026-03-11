# UCM-T chem / VSEPR sphere v.1

Standalone GitHub version of the UCM-T chem / VSEPR sphere v.1 project.

This repository contains the reproducibility bundle accompanying the v.1 paper on a minimal sphere model for emergent VSEPR-like geometries.

The project originated from the broader UCM-T research program, but at the current stage it is maintained as an independent research artifact rather than as a module of the calibration pipeline.

Zenodo release: DOI 10.5281/zenodo.18924861

## Repository contents

- paper.tex - main LaTeX source of the article
- paper.pdf - compiled PDF version of the article
- figures/ - publication figures used in the paper
- tables/ - LaTeX table source used in the paper
- data/ - derived summary data used for the N=6 phase-map figure
- source_json/ - representative source JSON files for the main cases shown in the article
- map_series/ - full MAP_N6_NB2_NL4 series used to build the N=6 summary figure
- scripts/ - main and auxiliary scripts
- notes/ - internal manifest and bundle checklist used to organize the release

## Minimal reproducibility logic

1. Representative configurations were generated with scripts/ucm_vsepr_sphere_min_v2.py.
2. Sphere figures were generated from representative JSON outputs with scripts/viz_points_sphere.py.
3. The N=6 summary CSV was built from the MAP series with scripts/build_n6_map_summary.py.
4. The N=6 phase-map figure was generated from that CSV with scripts/plot_n6_phase_map.py.

## Notes

- GIF outputs were used for internal visual inspection, but PNG files are the publication figures.
- This v.1 bundle is intended as a compact reproducibility package for the article, not as an exhaustive archive of every exploratory run.

## Status

This repository is published as a standalone project related to UCM-T, but it is not currently part of the UCM-T calibration pipeline.