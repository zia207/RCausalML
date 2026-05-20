# Example data (optional)

Synthetic examples in this folder use `synthetic_data()` and `make_uplift_classification()` in R and do **not** require these files.

For examples that use **IHDP** or **card** (2SLS) data, download from the Python CausalML repository:

- **Base URL**: https://github.com/uber/causalml/tree/master/docs/examples/data  
- **Files**:
  - `ihdp_npci_1.csv` … `ihdp_npci_9.csv` (IHDP semi-synthetic)
  - `card.csv` (Card 1995 wage/education IV example)

**Download via R** (run once):

```r
data_dir <- "inst/examples/data"  # or your path
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
base <- "https://raw.githubusercontent.com/uber/causalml/master/docs/examples/data"
for (f in c("card.csv", paste0("ihdp_npci_", 1:9, ".csv"))) {
  download.file(file.path(base, f), file.path(data_dir, f), mode = "wb")
}
```

Or download manually from:  
https://github.com/uber/causalml/tree/master/docs/examples/data
