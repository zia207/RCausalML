#!/usr/bin/env python3
"""Standardize setup sections in tutorials/*.qmd to match 02-03 pattern."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"

# Intro / prose-only notebooks — skip
SKIP = {
    "01-00-casual-tree-introduction-r.qmd",
    "02-00-meta-learners-introduction-r.qmd",
    "03-00-uplift-trees-introduction-r.qmd",
    "04-00-double-ml-introduction-r.qmd",
    "05-00-DeepCausalML-introduction-r.qmd",
    "05-01-00-DeepCausalML-treatment-effect-estimators-introduction-r.qmd",
    "05-02-00-DeepCausalML-generative-latent-variable-causal-models-introduction-r.qmd",
}

# Already fully compliant — only add pkg_root if missing
COMPLIANT = {
    "02-03-meta-learners-continuous-treatment-r.qmd",
    "02-04-tmle-model-r.qmd",
    "03-01-uplift-trees-classification-r.qmd",
}

PKG_ROOT_BLOCK = """pkg_root <- local({
  d <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  for (i in seq_len(8L)) {
    if (file.exists(file.path(d, "DESCRIPTION")) &&
        any(grepl("^Package:\\\\s*RCausalML", readLines(file.path(d, "DESCRIPTION"), n = 5L))))
      return(d)
    parent <- dirname(d)
    if (identical(parent, d)) break
    d <- parent
  }
  if (file.exists("DESCRIPTION")) normalizePath(".", winslash = "/") else NA_character_
})
if (!is.na(pkg_root) && requireNamespace("devtools", quietly = TRUE)) {
  try(devtools::load_all(pkg_root, quiet = TRUE), silent = TRUE)
}"""

SETUP_HEADINGS = (
    "### Check and Install Required R Packages",
    "### Install Missing Packages",
    "### Verify Installation",
    "### Load Required Libraries",
    "### Check Loaded Packages",
    "### Load and Check Required Libraries",
    "### Packages used in this notebook",
    "## Setup",
    "## Load and Install Required Packages",
    "### Setup",
    "### Setup and Libraries",
)

INSTALL_EXTRA = {
    "02-02-meta-learners-multiple-discrete-treatment-r.qmd": "# remotes::install_github(\"mlr-org/mlr3learners\")",
    "02-04-tmle-model-r.qmd": "# install.packages(c(\"tmle\", \"MatchIt\"))",
}


def packages_prose(pkgs: list[str]) -> str:
    return ", ".join(f"`{p}`" for p in pkgs)


def extract_packages(text: str) -> list[str] | None:
    m = re.search(r"packages\s*<-\s*c\s*\((.*?)\)", text, re.DOTALL)
    if not m:
        return None
    body = m.group(1)
    pkgs = re.findall(r"['\"]([^'\"]+)['\"]", body)
    return pkgs or None


def find_setup_region(text: str) -> tuple[int, int] | None:
    """Return (start, end) line indices for setup block to replace."""
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if any(stripped.startswith(h) for h in SETUP_HEADINGS):
            start = i
            break
        if stripped == "## Setup" or stripped.startswith("## Load and Install Required Packages"):
            start = i
            break
    if start is None:
        # try Implementation in R followed by setup subsections
        for i, line in enumerate(lines):
            if line.strip() in ("## Implementation in R", "## Implementation in in R", "## Uplift Trees in R"):
                # look for next setup-like heading within 30 lines
                for j in range(i + 1, min(i + 40, len(lines))):
                    s = lines[j].strip()
                    if s.startswith("### ") and any(
                        k in s
                        for k in (
                            "Load",
                            "Check and Install",
                            "Install Missing",
                            "Packages used",
                            "Setup",
                        )
                    ):
                        start = j
                        break
                if start is not None:
                    break
    if start is None:
        return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        s = lines[j].strip()
        if s.startswith("## ") and not s.startswith("### "):
            end = j
            break
    return start, end


def extract_trailing_setup_chunks(text: str, setup_end_line: int) -> str:
    """Keep custom chunks immediately after standard setup (e.g. setup-multi-arm, setup)."""
    lines = text.splitlines(keepends=True)
    trailing = []
    i = setup_end_line
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("## ") and not s.startswith("### "):
            break
        if re.match(r"^####? .+", s) and "setup" in s.lower():
            # include #### heading + following chunk
            trailing.append(lines[i])
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```{r"):
                trailing.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip().startswith("```{r"):
                trailing.append(lines[i])
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    trailing.append(lines[i])
                    i += 1
                if i < len(lines):
                    trailing.append(lines[i])
                    i += 1
            continue
        if s.startswith("```{r") and re.search(r"label:\s*(setup|setup-multi-arm|device-setup)", s):
            while i < len(lines):
                trailing.append(lines[i])
                i += 1
                if trailing[-1].strip() == "```":
                    break
            continue
        if s.startswith("### ") and s not in (
            "### Install Missing Packages",
            "### Verify Installation",
            "### Load Required Libraries",
            "### Check Loaded Packages",
            "### Check and Install Required R Packages",
        ):
            break
        i += 1
    return "".join(trailing)


def build_standard_setup(pkgs: list[str], filename: str, extra_install: str = "") -> str:
    prose = packages_prose(pkgs)
    install_lines = [
        "# Install missing packages",
        "# new_packages <- packages[!(packages %in% installed.packages()[, \"Package\"])]",
        "# if (length(new_packages)) install.packages(new_packages)",
    ]
    if extra_install:
        install_lines.append(extra_install)

    pkg_lines = ",\n  ".join(f'"{p}"' for p in pkgs)
    return f"""### Check and Install Required R Packages

Following R packages are required to run this notebook. If any of these packages are not installed, you can install them using the code below:

{prose}

```{{r}}
#| label: lst-packages-vector
#| lst-cap: "Required R package names used throughout the notebook."
packages <- c(
  {pkg_lines}
)
```

### Install Missing Packages

```{{r}}
#| label: lst-install-missing-packages
#| lst-cap: "Optional commands to install missing CRAN/GitHub dependencies (commented by default)."
#| warning: false
#| error: false
{chr(10).join(install_lines)}
```

### Verify Installation

```{{r}}
#| label: lst-verify-package-installation
#| lst-cap: "Check that each required package namespace is available."
# Verify installation
cat("Installed packages:\\n")
print(sapply(packages, requireNamespace, quietly = TRUE))
```

### Load Required Libraries

```{{r}}
#| label: load-required-libraries
#| warning: false
{PKG_ROOT_BLOCK}
invisible(lapply(packages, function(pkg) {{
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}}))
```

### Check Loaded Packages

```{{r}}
#| label: lst-check-loaded-packages
#| lst-cap: "Confirm which package environments are attached on the search path."
# Check loaded packages
cat("Successfully loaded packages:\\n")
print(search()[grepl("package:", search())])
```
"""


def add_pkg_root_to_load(text: str) -> str:
    if "pkg_root <- local({" in text or "find_pkg_root" in text:
        return text
    needle = '```{r}\n#| label: load-required-libraries\n#| warning: false\n\n'
    idx = text.find(needle)
    if idx == -1:
        return text
    after = text[idx + len(needle):]
    if after.startswith("pkg_root"):
        return text
    if after.startswith('if (requireNamespace("devtools"') or after.startswith('if (file.exists("DESCRIPTION")'):
        return text[: idx + len(needle)] + PKG_ROOT_BLOCK + "\n" + after
    return text


def process_file(path: Path, dry_run: bool = False) -> str:
    text = path.read_text()
    name = path.name

    if name in SKIP:
        return "skip (intro)"

    pkgs = extract_packages(text)
    if not pkgs:
        return "skip (no packages vector)"

    if name in COMPLIANT:
        new_text = add_pkg_root_to_load(text)
        if new_text != text:
            if not dry_run:
                path.write_text(new_text)
            return "updated (pkg_root only)"
        return "ok (already compliant)"

    region = find_setup_region(text)
    if region is None:
        return "skip (no setup region found)"

    start, end = region
    lines = text.splitlines(keepends=True)
    trailing = extract_trailing_setup_chunks(text, end)
    extra = INSTALL_EXTRA.get(name, "")
    new_setup = build_standard_setup(pkgs, name, extra)
    new_text = "".join(lines[:start]) + new_setup + "\n" + trailing + "".join(lines[end:])

    # Special: preserve notebook-specific load logic appended after standard load
    special_preserves = {
        "04-07-double-ml-cluster-robust-r.qmd": "_preserve_dml_source",
        "04-08-double-ml-ensemble-learners-r.qmd": "_preserve_dml_source",
        "05-02-02-DeepCausalML-variational-autoencoder-causalVAE.qmd": "_preserve_device",
        "05-02-02-DeepCausalML-identifiable-variational-autoencoders-CausaliVAEs.qmd": "_preserve_ivae",
        "05-02-03-DeepCausalML-identifiable-variational-autoencoders-CausaliVAEs.qmd": "_preserve_ivae",
        "05-02-06-DeepCausalML-deep-structural-causal-model-DSCM-r.qmd": "_preserve_dscm",
        "03-04-uplift-xgboost.qmd": "_preserve_xgboost_check",
    }
    if name in special_preserves:
        return f"manual ({special_preserves[name]})"

    if new_text == text:
        return "unchanged"

    if not dry_run:
        path.write_text(new_text)
    return "updated"


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    results = {}
    for path in sorted(TUTORIALS.glob("*.qmd")):
        results[path.name] = process_file(path, dry_run=dry_run)

    for name, status in sorted(results.items()):
        if not status.startswith("skip") and status != "ok (already compliant)":
            print(f"{name}: {status}")

    updated = sum(1 for s in results.values() if s.startswith("updated"))
    manual = sum(1 for s in results.values() if s.startswith("manual"))
    print(f"\nTotal: {updated} updated, {manual} need manual, {len(results)} files scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
