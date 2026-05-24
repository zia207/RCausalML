#!/usr/bin/env Rscript
# Inject pkgdown site navbar into Quarto-rendered tutorial HTML pages.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[[1]] else "."
root <- normalizePath(root, mustWork = TRUE)

pkgdown_path <- file.path(root, "_pkgdown.yml")
desc_path <- file.path(root, "DESCRIPTION")
tutorials_dir <- file.path(root, "docs", "tutorials")

if (!file.exists(pkgdown_path)) {
  stop("_pkgdown.yml not found in ", root)
}

if (!requireNamespace("yaml", quietly = TRUE)) {
  stop("Package 'yaml' is required.")
}

pkgdown_cfg <- yaml::yaml.load_file(pkgdown_path)
navbar_cfg <- pkgdown_cfg$navbar

version <- sub(
  ".*Version: ([^\n]+).*", "\\1",
  readLines(desc_path, warn = FALSE)[grep("^Version:", readLines(desc_path, warn = FALSE))]
)

`%||%` <- function(x, y) if (is.null(x)) y else x

slugify <- function(text) {
  gsub("[^a-zA-Z0-9]+", "-", tolower(text))
}

tutorial_href <- function(href) {
  if (is.null(href) || !nzchar(href)) {
    return("#")
  }
  if (grepl("^https?://", href)) {
    return(href)
  }
  if (grepl("^tutorials/", href)) {
    return(sub("^tutorials/", "", href))
  }
  paste0("../", href)
}

dropdown_items_html <- function(menu_items) {
  vapply(menu_items, function(item) {
    href <- tutorial_href(item$href %||% "")
    sprintf('    <li><a class="dropdown-item" href="%s">%s</a></li>', href, item$text)
  }, character(1))
}

nav_item_html <- function(item, active = FALSE) {
  id <- slugify(item$text)
  active_class <- if (isTRUE(active)) "active " else ""

  if (!is.null(item$menu)) {
    dropdown_id <- paste0("dropdown-", id)
    items <- paste(dropdown_items_html(item$menu), collapse = "\n")
    paste0(
      '        <li class="', active_class, 'nav-item dropdown">\n',
      '          <button class="nav-link dropdown-toggle" type="button" id="', dropdown_id,
      '" data-bs-toggle="dropdown" aria-expanded="false" aria-haspopup="true">', item$text, '</button>\n',
      '          <ul class="dropdown-menu" aria-labelledby="', dropdown_id, '">\n',
      items, "\n",
      '          </ul>\n',
      '        </li>'
    )
  } else {
    href <- tutorial_href(item$href %||% "")
    paste0(
      '        <li class="', active_class, 'nav-item">',
      '<a class="nav-link" href="', href, '">', item$text, '</a></li>'
    )
  }
}

right_item_html <- function(item) {
  href <- item$href %||% "#"
  label <- item$`aria-label` %||% item$text %||% ""
  if (!is.null(item$icon)) {
    icon <- item$icon
    if (!grepl("^fa-", icon)) {
      icon <- paste0("fa-", icon)
    }
    if (!grepl(" fa-lg$", icon)) {
      icon <- paste0(icon, " fa-lg")
    }
    paste0(
      '        <li class="nav-item"><a class="external-link nav-link" href="', href,
      '" aria-label="', label, '"><span class="fa ', icon, '"></span></a></li>'
    )
  } else {
    paste0(
      '        <li class="nav-item"><a class="external-link nav-link" href="', href,
      '">', item$text, '</a></li>'
    )
  }
}

left_items <- navbar_cfg$left
right_items <- navbar_cfg$right %||% list()

tutorial_topics <- c("Tutorials")

left_html <- vapply(left_items, function(item) {
  nav_item_html(item, active = item$text %in% tutorial_topics)
}, character(1))

right_html <- vapply(right_items, right_item_html, character(1))

navbar_type <- navbar_cfg$type %||% "default"
navbar_bg <- navbar_cfg$bg %||% if (identical(navbar_type, "dark")) "primary" else "light"
navbar_theme <- if (identical(navbar_cfg$type, "dark")) ' data-bs-theme="dark"' else ""

navbar_html <- paste0(
  '<a href="#quarto-document-content" class="visually-hidden-focusable">Skip to contents</a>\n',
  '<nav class="navbar navbar-expand-lg fixed-top bg-', navbar_bg, '"', navbar_theme,
  ' aria-label="Site navigation"><div class="container">\n',
  '    <a class="navbar-brand me-2" href="../index.html">RCausalML</a>\n',
  '    <small class="nav-text text-muted me-auto" data-bs-toggle="tooltip" data-bs-placement="bottom" title="">',
  version, '</small>\n',
  '    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#rcausalml-navbar" ',
  'aria-controls="rcausalml-navbar" aria-expanded="false" aria-label="Toggle navigation">\n',
  '      <span class="navbar-toggler-icon"></span>\n',
  '    </button>\n',
  '    <div id="rcausalml-navbar" class="collapse navbar-collapse ms-3">\n',
  '      <ul class="navbar-nav me-auto">\n',
  paste(left_html, collapse = "\n"), "\n",
  '      </ul>\n',
  '      <ul class="navbar-nav">\n',
  paste(right_html, collapse = "\n"), "\n",
  '      </ul>\n',
  '    </div>\n',
  '  </div>\n',
  '</nav>\n',
  '<div class="rcausalml-navbar-spacer" aria-hidden="true"></div>\n'
)

head_marker <- "<!-- RCausalML pkgdown tutorial head -->"
body_marker <- "<!-- RCausalML pkgdown tutorial navbar -->"

head_includes <- paste(
  readLines(file.path(root, "pkgdown", "tutorial-head-includes.html"), warn = FALSE),
  collapse = "\n"
)

navbar_block <- paste0(body_marker, "\n", navbar_html)

strip_block <- function(html, start_pattern, end_pattern, include_end = TRUE) {
  while (any(grepl(start_pattern, html, fixed = TRUE))) {
    start <- grep(start_pattern, html, fixed = TRUE)[1]
    end_candidates <- grep(end_pattern, html, fixed = TRUE)
    end_candidates <- end_candidates[end_candidates >= start]
    if (!length(end_candidates)) break
    end <- end_candidates[1]
    drop_end <- if (include_end) end else end - 1L
    if (drop_end < start) break
    html <- html[-(start:drop_end)]
  }
  html
}

strip_previous <- function(html) {
  html <- strip_block(html, head_marker, "</head>", include_end = FALSE)
  while (any(grepl(body_marker, html, fixed = TRUE))) {
    html <- strip_block(html, body_marker, "rcausalml-navbar-spacer", include_end = TRUE)
  }
  html <- gsub("\\s*rcausalml-tutorial", "", html)
  html
}

html_files <- list.files(tutorials_dir, pattern = "\\.html$", full.names = TRUE)
html_files <- html_files[!grepl("^_", basename(html_files))]

updated <- 0L
for (path in html_files) {
  html <- readLines(path, warn = FALSE)
  html <- strip_previous(html)

  head_idx <- grep("</head>", html, fixed = TRUE)[1]
  body_idx <- grep("<body", html)[1]

  if (is.na(head_idx) || is.na(body_idx)) {
    warning("Skipping (missing head/body): ", path)
    next
  }

  html <- c(
    html[seq_len(head_idx - 1)],
    head_includes,
    html[head_idx:body_idx],
    navbar_block,
    html[(body_idx + 1):length(html)]
  )

  html[body_idx + 1] <- sub(
    "<body class=\"([^\"]*)\"",
    '<body class="\\1 rcausalml-tutorial"',
    html[body_idx + 1]
  )
  if (!grepl("rcausalml-tutorial", html[body_idx + 1])) {
    html[body_idx + 1] <- sub("<body", '<body class="rcausalml-tutorial"', html[body_idx + 1])
  }

  writeLines(html, path, useBytes = TRUE)
  updated <- updated + 1L
}

cat("Updated", updated, "tutorial pages in", tutorials_dir, "\n")

writeLines(
  c(body_marker, navbar_html),
  file.path(root, "pkgdown", "tutorial-navbar.html"),
  useBytes = TRUE
)
