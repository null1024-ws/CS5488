# One Core to Many

Parallel data-computing study of the 2023 NYC yellow-taxi trips, for CS5488.

The reported corpus is the twelve TLC yellow-trip files for 2023 (38,310,226 raw rows). Cleaning, aggregation, and k-means run on one machine. FP-Growth is timed only as a case that does not speed up. Scaling curves compare the same code at 1, 2, 4, and 8 workers. This host has 4 physical CPUs, so p = 8 is oversubscription, not an 8-core server. The interpretive clusters are rebuilt with `python -m analysis.year2023_story`.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m pipeline.download
python -m pipeline.etl_parallel --workers 3
python -m analysis.prepare
python -m algorithms.tests.test_equivalence
python -m bench.run_matrix
python -m analysis.build_results
python -m analysis.make_figures
pdflatex -interaction=nonstopmode report.tex
bibtex report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

Run the LaTeX commands from `report/`. Raw Parquet files are downloaded into `data/raw/` and are not committed. The report cites the twelve files of 2023. `python -m analysis.year2023_story` rebuilds the borough counts, the histograms, and the k-means profiles from those files. The wall times in the report are the medians already stored in `results/runs.csv`. `python -m analysis.check_numbers` checks those medians against the text.

## Layout

- `pipeline/` download, R1–R7 cleaning, process-pool ETL
- `algorithms/` numba `prange` k-means and task-parallel FP-Growth
- `bench/` warmup, 5 repeats, median wall time
- `analysis/` figures in the figures4papers publication style
- `report/` English LaTeX report

Figures follow the palette, spine, and export rules in Chen Liu's [figures4papers](https://github.com/ChenLiu-1996/figures4papers) scientific-figure-making skill (`analysis/pubstyle.py`).
