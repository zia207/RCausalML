# Graph drawing and layout (R port of NetworkX drawing module)
#
# Converts layout and drawing APIs from:
#   networkx/drawing/layout.py, networkx/drawing/nx_pylab.py
#
# Dependencies: igraph (for graph and some layouts), base R graphics.
# Layouts return a matrix with rows = nodes, columns = (x, y), or a list
# keyed by node id/name for compatibility.

# ------------------------------------------------------------------------------
# Helpers: graph from adjacency or igraph
# ------------------------------------------------------------------------------

#' Ensure graph is igraph; accept adjacency matrix or igraph
#' @param G igraph object or square adjacency matrix (numeric).
#' @param directed Whether to interpret matrix as directed (only if G is matrix).
#' @return igraph object.
#' @noRd
.drawing_as_igraph <- function(G, directed = TRUE) {
  if (inherits(G, "igraph")) return(G)
  M <- as.matrix(G)
  if (nrow(M) != ncol(M)) stop("Adjacency matrix must be square.")
  mode <- if (directed) "directed" else "undirected"
  igraph::graph_from_adjacency_matrix(
    (M != 0) * 1L,
    mode = mode,
    weighted = NULL,
    diag = FALSE
  )
}

#' Node list from graph (integer indices or names)
#' @noRd
.drawing_nodes <- function(G) {
  if (inherits(G, "igraph")) {
    igraph::V(G)
  } else {
    seq_len(nrow(as.matrix(G)))
  }
}

#' Number of nodes
#' @noRd
.drawing_nnodes <- function(G) {
  if (inherits(G, "igraph")) igraph::vcount(G) else nrow(as.matrix(G))
}

# ------------------------------------------------------------------------------
# rescale_layout (from layout.py)
# ------------------------------------------------------------------------------

#' Rescale layout positions to (-scale, scale) on all axes
#'
#' Centers positions by subtracting the mean per axis, then scales so the
#' largest absolute coordinate equals \code{scale}. Preserves aspect ratio.
#'
#' @param pos Numeric matrix; each row is a position (e.g. n x 2 for 2D).
#' @param scale Scale factor; extent is (-scale, scale) in each axis.
#' @return Matrix of rescaled positions (same dimensions as \code{pos}).
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # rescale_layout(...)
#' }
#' @export
rescale_layout <- function(pos, scale = 1) {
  if (!is.matrix(pos) || nrow(pos) == 0) return(pos)
  pos <- scale(pos, scale = FALSE)  # subtract column means
  lim <- max(abs(pos), na.rm = TRUE)
  if (lim > 0) pos <- pos * (scale / lim)
  pos
}

#' Rescale a layout given as named list or matrix
#'
#' @param pos Named list (node -> c(x,y)) or matrix with rownames.
#' @param scale Scale factor.
#' @return Same structure as input, with rescaled coordinates.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # rescale_layout_dict(...)
#' }
#' @export
rescale_layout_dict <- function(pos, scale = 1) {
  if (is.list(pos) && length(pos) == 0) return(pos)
  if (is.matrix(pos)) {
    pos_v <- pos
    rn <- rownames(pos)
  } else {
    nms <- names(pos)
    pos_v <- do.call(rbind, pos)
    rownames(pos_v) <- nms
  }
  pos_v <- rescale_layout(pos_v, scale = scale)
  if (is.list(pos) && !is.matrix(pos)) {
    out <- lapply(seq_len(nrow(pos_v)), function(i) pos_v[i, ])
    names(out) <- rownames(pos_v)
    return(out)
  }
  rownames(pos_v) <- rownames(pos)
  pos_v
}

# ------------------------------------------------------------------------------
# _process_params equivalent: center and dim
# ------------------------------------------------------------------------------
.drawing_process_params <- function(G, center, dim) {
  n <- .drawing_nnodes(G)
  if (is.null(center)) center <- rep(0, dim)
  center <- rep(center, length.out = dim)
  list(G = G, center = center, n = n)
}

# ------------------------------------------------------------------------------
# random_layout (from layout.py)
# ------------------------------------------------------------------------------

#' Random layout: place nodes uniformly at random in the unit square
#'
#' @param G igraph object or adjacency matrix.
#' @param center Optional center coordinates (length = dim).
#' @param dim Dimension of layout (2 or 3).
#' @param seed Optional random seed.
#' @return Matrix of positions (n x dim) with rownames = node ids; or list keyed by node if \code{as_list=TRUE}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # random_layout(...)
#' }
#' @export
random_layout <- function(G, center = NULL, dim = 2L, seed = NULL, as_list = FALSE) {
  if (!is.null(seed)) set.seed(seed)
  G <- .drawing_as_igraph(G)
  n <- igraph::vcount(G)
  params <- .drawing_process_params(G, center, dim)
  center <- params$center
  pos <- matrix(stats::runif(n * dim), nrow = n, ncol = dim) + rep(center, each = n)
  rownames(pos) <- as.character(igraph::V(G))
  if (as_list) pos <- .pos_matrix_to_list(pos)
  pos
}

# ------------------------------------------------------------------------------
# circular_layout (from layout.py)
# ------------------------------------------------------------------------------

#' Circular layout: place nodes on a circle
#'
#' @param G igraph object or adjacency matrix.
#' @param scale Scale factor for radius.
#' @param center Center coordinates.
#' @param dim Dimension (2 only for circle).
#' @return Matrix of positions (n x 2) or list if \code{as_list=TRUE}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # circular_layout(...)
#' }
#' @export
circular_layout <- function(G, scale = 1, center = NULL, dim = 2L, as_list = FALSE) {
  if (dim < 2) stop("circular_layout requires dim >= 2")
  G <- .drawing_as_igraph(G)
  n <- igraph::vcount(G)
  params <- .drawing_process_params(G, center, dim)
  center <- params$center
  if (n == 0) {
    pos <- matrix(numeric(0), ncol = 2)
  } else if (n == 1) {
    pos <- matrix(center[seq_len(2)], nrow = 1)
  } else {
    theta <- (seq_len(n) - 1) / n * 2 * pi
    pos <- cbind(cos(theta), sin(theta))
    pos <- rescale_layout(pos, scale = scale) + rep(center[seq_len(2)], each = n)
  }
  rownames(pos) <- as.character(igraph::V(G))
  if (as_list) pos <- .pos_matrix_to_list(pos)
  pos
}

.pos_matrix_to_list <- function(pos) {
  nms <- rownames(pos)
  out <- lapply(seq_len(nrow(pos)), function(i) pos[i, ])
  names(out) <- nms
  out
}

# ------------------------------------------------------------------------------
# shell_layout (from layout.py)
# ------------------------------------------------------------------------------

#' Shell layout: place nodes in concentric circles (shells)
#'
#' @param G igraph object or adjacency matrix.
#' @param nlist List of integer vectors; each vector gives node indices (1-based) in that shell. If NULL, all nodes in one shell.
#' @param rotate Angle in radians to rotate each shell (default pi/n_shells).
#' @param scale Scale factor.
#' @param center Center coordinates.
#' @return Matrix of positions (n x 2) or list if \code{as_list=TRUE}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # shell_layout(...)
#' }
#' @export
shell_layout <- function(G, nlist = NULL, rotate = NULL, scale = 1, center = NULL, as_list = FALSE) {
  G <- .drawing_as_igraph(G)
  n <- igraph::vcount(G)
  params <- .drawing_process_params(G, center, 2)
  center <- params$center
  if (n == 0) return(if (as_list) list() else matrix(numeric(0), ncol = 2))
  if (n == 1) {
    pos <- matrix(center[seq_len(2)], nrow = 1)
    rownames(pos) <- as.character(igraph::V(G))
    return(if (as_list) .pos_matrix_to_list(pos) else pos)
  }
  if (is.null(nlist)) nlist <- list(seq_len(n))
  n_shells <- length(nlist)
  radius_bump <- scale / n_shells
  radius <- if (length(nlist[[1]]) == 1) 0 else radius_bump
  if (is.null(rotate)) rotate <- pi / n_shells
  first_theta <- rotate
  pos_list <- list()
  for (shell in nlist) {
    k <- length(shell)
    theta <- seq(0, 2 * pi, length.out = k + 1)[-1] + first_theta
    xy <- radius * cbind(cos(theta), sin(theta)) + rep(center, each = k)
    for (i in seq_along(shell)) pos_list[[as.character(igraph::V(G)[shell[i]])]] <- xy[i, ]
    radius <- radius + radius_bump
    first_theta <- first_theta + rotate
  }
  nodes <- as.character(igraph::V(G))
  pos <- do.call(rbind, pos_list[nodes])
  rownames(pos) <- nodes
  if (as_list) pos <- .pos_matrix_to_list(pos)
  pos
}

# ------------------------------------------------------------------------------
# bipartite_layout (from layout.py)
# ------------------------------------------------------------------------------

#' Bipartite layout: two parallel lines (vertical or horizontal)
#'
#' @param G igraph object or adjacency matrix.
#' @param nodes Optional vector of node ids in the "top" set; if NULL and G is bipartite, uses bipartite partition.
#' @param align "vertical" (two columns) or "horizontal" (two rows).
#' @param scale Scale factor.
#' @param center Center coordinates.
#' @param aspect_ratio Width/height ratio of the layout box.
#' @return Matrix of positions (n x 2) or list if \code{as_list=TRUE}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # bipartite_layout(...)
#' }
#' @export
bipartite_layout <- function(G, nodes = NULL, align = c("vertical", "horizontal"),
                             scale = 1, center = NULL, aspect_ratio = 4/3, as_list = FALSE) {
  align <- match.arg(align)
  G <- .drawing_as_igraph(G)
  n <- igraph::vcount(G)
  params <- .drawing_process_params(G, center, 2)
  center <- params$center
  if (n == 0) return(if (as_list) list() else matrix(numeric(0), ncol = 2))
  vids <- as.character(igraph::V(G))
  if (is.null(nodes)) {
    if (igraph::is_bipartite(G)) {
      types <- igraph::V(G)$type
      if (is.null(types)) types <- igraph::bipartite_mapping(G)$type
      top <- vids[types]
      bottom <- vids[!types]
    } else {
      top <- vids[seq_len(ceiling(n/2))]
      bottom <- setdiff(vids, top)
    }
    nodes_ordered <- c(top, bottom)
  } else {
    top <- as.character(nodes)
    bottom <- setdiff(vids, top)
    nodes_ordered <- c(top, bottom)
  }
  height <- 1
  width <- aspect_ratio * height
  offset <- c(width / 2, height / 2)
  n_top <- length(top)
  n_bottom <- length(bottom)
  left_xs <- rep(0, n_top)
  right_xs <- rep(width, n_bottom)
  left_ys <- seq(0, height, length.out = max(1, n_top))
  right_ys <- seq(0, height, length.out = max(1, n_bottom))
  top_pos <- cbind(left_xs, left_ys) - rep(offset, each = n_top)
  bottom_pos <- cbind(right_xs, right_ys) - rep(offset, each = n_bottom)
  pos <- rbind(top_pos, bottom_pos)
  pos <- rescale_layout(pos, scale = scale) + rep(center, each = nrow(pos))
  if (align == "horizontal") pos <- pos[, c(2, 1)]
  rownames(pos) <- nodes_ordered
  if (as_list) pos <- .pos_matrix_to_list(pos)
  pos
}

# ------------------------------------------------------------------------------
# spring_layout / Fruchterman-Reingold (from layout.py; use igraph)
# ------------------------------------------------------------------------------

#' Spring layout (Fruchterman-Reingold force-directed)
#'
#' @param G igraph object or adjacency matrix.
#' @param seed Random seed for initial positions.
#' @param iterations Number of iterations.
#' @param weight Edge attribute name for weights (or NULL for unweighted).
#' @param scale Scale factor for final positions.
#' @param center Center coordinates.
#' @return Matrix of positions (n x 2) or list if \code{as_list=TRUE}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # spring_layout(...)
#' }
#' @export
spring_layout <- function(G, seed = NULL, iterations = 50, weight = "weight",
                          scale = 1, center = NULL, as_list = FALSE) {
  if (!is.null(seed)) set.seed(seed)
  G <- .drawing_as_igraph(G)
  n <- igraph::vcount(G)
  params <- .drawing_process_params(G, center, 2)
  center <- params$center
  if (n == 0) return(if (as_list) list() else matrix(numeric(0), ncol = 2))
  if (n == 1) {
    pos <- matrix(center, nrow = 1)
    rownames(pos) <- as.character(igraph::V(G))
    return(if (as_list) .pos_matrix_to_list(pos) else pos)
  }
  if (!is.null(weight) && weight %in% igraph::edge_attr_names(G)) {
    coords <- igraph::layout_with_fr(G, niter = iterations, weights = igraph::E(G)[[weight]])
  } else {
    coords <- igraph::layout_with_fr(G, niter = iterations)
  }
  pos <- rescale_layout(coords, scale = scale) + rep(center, each = n)
  rownames(pos) <- as.character(igraph::V(G))
  if (as_list) pos <- .pos_matrix_to_list(pos)
  pos
}

#' Alias for spring_layout (NetworkX name)
#' @return
#' Object returned by \code{function_name}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # function_name(...)
#' }
#' @export
fruchterman_reingold_layout <- spring_layout

# ------------------------------------------------------------------------------
# kamada_kawai_layout (from layout.py; use igraph)
# ------------------------------------------------------------------------------

#' Kamada-Kawai layout (path-length cost)
#'
#' @param G igraph object or adjacency matrix.
#' @param weight Edge attribute for weights (or NULL).
#' @param scale Scale factor.
#' @param center Center coordinates.
#' @return Matrix of positions (n x 2) or list if \code{as_list=TRUE}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # kamada_kawai_layout(...)
#' }
#' @export
kamada_kawai_layout <- function(G, weight = "weight", scale = 1, center = NULL, as_list = FALSE) {
  G <- .drawing_as_igraph(G)
  n <- igraph::vcount(G)
  params <- .drawing_process_params(G, center, 2)
  center <- params$center
  if (n == 0) return(if (as_list) list() else matrix(numeric(0), ncol = 2))
  if (n == 1) {
    pos <- matrix(center, nrow = 1)
    rownames(pos) <- as.character(igraph::V(G))
    return(if (as_list) .pos_matrix_to_list(pos) else pos)
  }
  w <- NULL
  if (!is.null(weight) && weight %in% igraph::edge_attr_names(G)) w <- igraph::E(G)[[weight]]
  coords <- igraph::layout_with_kk(G, weights = w)
  pos <- rescale_layout(coords, scale = scale) + rep(center, each = n)
  rownames(pos) <- as.character(igraph::V(G))
  if (as_list) pos <- .pos_matrix_to_list(pos)
  pos
}

# ------------------------------------------------------------------------------
# spectral_layout (from layout.py; use igraph or eigen on Laplacian)
# ------------------------------------------------------------------------------

#' Spectral layout (eigenvectors of graph Laplacian)
#'
#' @param G igraph object or adjacency matrix.
#' @param weight Edge attribute for weights (or NULL).
#' @param scale Scale factor.
#' @param center Center coordinates.
#' @return Matrix of positions (n x 2) or list if \code{as_list=TRUE}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # spectral_layout(...)
#' }
#' @export
spectral_layout <- function(G, weight = "weight", scale = 1, center = NULL, as_list = FALSE) {
  G <- .drawing_as_igraph(G)
  n <- igraph::vcount(G)
  params <- .drawing_process_params(G, center, 2)
  center <- params$center
  if (n == 0) return(if (as_list) list() else matrix(numeric(0), ncol = 2))
  if (n == 1) {
    pos <- matrix(center, nrow = 1)
    rownames(pos) <- as.character(igraph::V(G))
    return(if (as_list) .pos_matrix_to_list(pos) else pos)
  }
  if (n == 2) {
    pos <- rbind(rep(0, 2), center * 2)
    rownames(pos) <- as.character(igraph::V(G))
    if (as_list) pos <- .pos_matrix_to_list(pos)
    return(pos)
  }
  A <- igraph::as_adjacency_matrix(G, attr = if (weight %in% igraph::edge_attr_names(G)) weight else NULL, sparse = FALSE)
  if (igraph::is_directed(G)) A <- A + t(A)
  D <- diag(rowSums(A))
  L <- D - A
  ev <- eigen(L, symmetric = TRUE)
  idx <- order(ev$values)[2:3]
  pos <- Re(ev$vectors[, idx])
  pos <- rescale_layout(pos, scale = scale) + rep(center, each = n)
  rownames(pos) <- as.character(igraph::V(G))
  if (as_list) pos <- .pos_matrix_to_list(pos)
  pos
}

# ------------------------------------------------------------------------------
# spiral_layout (from layout.py)
# ------------------------------------------------------------------------------

#' Spiral layout
#'
#' @param G igraph object or adjacency matrix.
#' @param scale Scale factor.
#' @param center Center coordinates.
#' @param resolution Compactness (lower = more compressed).
#' @param equidistant If TRUE, equidistant nodes along spiral; else equal angles.
#' @return Matrix of positions (n x 2) or list if \code{as_list=TRUE}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # spiral_layout(...)
#' }
#' @export
spiral_layout <- function(G, scale = 1, center = NULL, resolution = 0.35,
                          equidistant = FALSE, as_list = FALSE) {
  G <- .drawing_as_igraph(G)
  n <- igraph::vcount(G)
  params <- .drawing_process_params(G, center, 2)
  center <- params$center
  if (n == 0) return(if (as_list) list() else matrix(numeric(0), ncol = 2))
  if (n == 1) {
    pos <- matrix(center, nrow = 1)
    rownames(pos) <- as.character(igraph::V(G))
    return(if (as_list) .pos_matrix_to_list(pos) else pos)
  }
  if (equidistant) {
    chord <- 1
    step <- 0.5
    theta <- resolution
    theta <- theta + chord / (step * theta)
    pos <- matrix(0, nrow = n, ncol = 2)
    for (i in seq_len(n)) {
      r <- step * theta
      theta <- theta + chord / r
      pos[i, ] <- c(cos(theta) * r, sin(theta) * r)
    }
  } else {
    dist <- seq_len(n) - 1
    angle <- resolution * dist
    pos <- cbind(dist * cos(angle), dist * sin(angle))
  }
  pos <- rescale_layout(pos, scale = scale) + rep(center, each = n)
  rownames(pos) <- as.character(igraph::V(G))
  if (as_list) pos <- .pos_matrix_to_list(pos)
  pos
}

# ------------------------------------------------------------------------------
# multipartite_layout (from layout.py)
# ------------------------------------------------------------------------------

#' Multipartite (layered) layout
#'
#' @param G igraph object or adjacency matrix.
#' @param subset_key Either character (node attribute name giving layer) or named list mapping layer id -> node ids.
#' @param align "vertical" or "horizontal".
#' @param scale Scale factor.
#' @param center Center coordinates.
#' @return Matrix of positions (n x 2) or list if \code{as_list=TRUE}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # multipartite_layout(...)
#' }
#' @export
multipartite_layout <- function(G, subset_key = "subset", align = c("vertical", "horizontal"),
                                scale = 1, center = NULL, as_list = FALSE) {
  align <- match.arg(align)
  G <- .drawing_as_igraph(G)
  n <- igraph::vcount(G)
  params <- .drawing_process_params(G, center, 2)
  center <- params$center
  if (n == 0) return(if (as_list) list() else matrix(numeric(0), ncol = 2))
  vids <- as.character(igraph::V(G))
  if (is.list(subset_key) && !is.null(names(subset_key))) {
    layers <- subset_key
    if (length(unlist(layers)) != n) stop("all nodes must be in one subset of subset_key dict")
  } else {
    attr_name <- subset_key
    node_attr <- igraph::vertex_attr(G, attr_name)
    if (is.null(node_attr) || any(is.na(node_attr))) stop("all nodes need subset_key attribute: ", attr_name)
    layers <- split(vids, node_attr)
  }
  pos_list <- list()
  nodes_ordered <- character(0)
  width <- length(layers)
  for (i in seq_along(layers)) {
    layer <- as.character(layers[[i]])
    height <- length(layer)
    xs <- rep(i - 1, height)
    ys <- seq(0, height - 1)
    offset <- c((width - 1) / 2, (height - 1) / 2)
    layer_pos <- cbind(xs, ys) - rep(offset, each = height)
    for (j in seq_along(layer)) {
      pos_list[[layer[j]]] <- layer_pos[j, ]
      nodes_ordered <- c(nodes_ordered, layer[j])
    }
  }
  pos <- do.call(rbind, pos_list[nodes_ordered])
  pos <- rescale_layout(pos, scale = scale) + rep(center, each = nrow(pos))
  if (align == "horizontal") pos <- pos[, c(2, 1)]
  rownames(pos) <- nodes_ordered
  if (as_list) pos <- .pos_matrix_to_list(pos)
  pos
}

# ------------------------------------------------------------------------------
# bfs_layout (from layout.py)
# ------------------------------------------------------------------------------

#' BFS layout: layers by breadth-first search from a start node
#'
#' @param G igraph object or adjacency matrix.
#' @param start Start node (vertex id or name).
#' @param align "vertical" or "horizontal".
#' @param scale Scale factor.
#' @param center Center coordinates.
#' @return Matrix of positions (n x 2) or list if \code{as_list=TRUE}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # bfs_layout(...)
#' }
#' @export
bfs_layout <- function(G, start, align = c("vertical", "horizontal"),
                       scale = 1, center = NULL, as_list = FALSE) {
  G <- .drawing_as_igraph(G)
  n <- igraph::vcount(G)
  if (n == 0) return(if (as_list) list() else matrix(numeric(0), ncol = 2))
  start_id <- if (is.character(start)) match(start, as.character(igraph::V(G))) else start
  if (is.na(start_id)) stop("start node not found in graph")
  dmat <- igraph::distances(G, v = start_id, mode = "out")
  dvec <- as.numeric(dmat[1, ])
  dvec[is.infinite(dvec)] <- max(dvec[is.finite(dvec)], 0) + 1
  layers_list <- split(as.character(igraph::V(G)), dvec)
  layers_list <- layers_list[order(as.numeric(names(layers_list)))]
  names(layers_list) <- seq_along(layers_list)
  multipartite_layout(G, subset_key = layers_list, align = align, scale = scale, center = center, as_list = as_list)
}

# ------------------------------------------------------------------------------
# draw_network: high-level plot (from nx_pylab draw/draw_networkx)
# ------------------------------------------------------------------------------

#' Draw a network (graph) with nodes and edges
#'
#' Plots nodes at given positions and edges as segments or arrows. Uses base R
#' graphics. For more control use \code{draw_networkx_nodes}, \code{draw_networkx_edges},
#' \code{draw_networkx_labels} separately.
#'
#' @param G igraph object or adjacency matrix.
#' @param pos Optional layout matrix (n x 2) or list of (x,y); if NULL, \code{spring_layout(G)} is used.
#' @param layout Layout function name or function; used when \code{pos} is NULL. One of \code{"spring"}, \code{"circular"}, \code{"random"}, \code{"kamada_kawai"}, \code{"spectral"}, \code{"shell"}.
#' @param vertex.size Node radius (scalar or vector).
#' @param vertex.color Node color(s).
#' @param vertex.frame.color Node border color.
#' @param edge.color Edge color.
#' @param edge.width Edge width.
#' @param with_labels Whether to draw node labels.
#' @param vertex.label Node labels (default: vertex names).
#' @param vertex.label.cex Label font size.
#' @param vertex.label.color Label color.
#' @param arrow.size Arrow head size for directed edges (0 to disable arrows).
#' @param add If TRUE, add to existing plot; otherwise create new plot.
#' @param ... Passed to \code{plot()} when \code{add=FALSE} (e.g. \code{main}, \code{xlim}).
#' @return Invisibly, the layout matrix used.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # draw_network(...)
#' }
#' @export
draw_network <- function(G, pos = NULL, layout = "spring",
                         vertex.size = 8, vertex.color = "lightblue", vertex.frame.color = "darkblue",
                         edge.color = "gray40", edge.width = 1,
                         with_labels = TRUE, vertex.label = NULL, vertex.label.cex = 0.8, vertex.label.color = "black",
                         arrow.size = 0.2, add = FALSE, ...) {
  G <- .drawing_as_igraph(G)
  if (is.null(pos)) {
    layout_fun <- switch(
      layout,
      spring = spring_layout,
      circular = circular_layout,
      random = random_layout,
      kamada_kawai = kamada_kawai_layout,
      spectral = spectral_layout,
      shell = shell_layout,
      bipartite = bipartite_layout,
      spiral = spiral_layout,
      if (is.function(layout)) layout else spring_layout
    )
    pos <- layout_fun(G, as_list = FALSE)
  }
  if (is.list(pos)) pos <- do.call(rbind, pos)
  nodes <- rownames(pos)
  if (is.null(nodes)) nodes <- as.character(seq_len(nrow(pos)))
  rownames(pos) <- nodes

  if (!add) {
    rx <- range(pos[, 1], na.rm = TRUE)
    ry <- range(pos[, 2], na.rm = TRUE)
    pad <- 0.05 * max(diff(rx), diff(ry), 1)
    plot(NA, xlim = rx + c(-pad, pad), ylim = ry + c(-pad, pad),
         xlab = "", ylab = "", axes = FALSE, ...)
  }

  draw_networkx_edges(G, pos, edge.color = edge.color, edge.width = edge.width, arrow.size = arrow.size)
  draw_networkx_nodes(G, pos, vertex.size = vertex.size, vertex.color = vertex.color, vertex.frame.color = vertex.frame.color)
  if (with_labels) {
    labs <- vertex.label
    if (is.null(labs)) labs <- as.character(igraph::V(G))
    draw_networkx_labels(G, pos, labels = labs, cex = vertex.label.cex, col = vertex.label.color)
  }
  invisible(pos)
}

# ------------------------------------------------------------------------------
# draw_networkx_nodes (from nx_pylab)
# ------------------------------------------------------------------------------

#' Draw network nodes at given positions
#'
#' @param G igraph object (used for node order/names).
#' @param pos Matrix (n x 2) or named list of coordinates; rownames/names must match node ids.
#' @param vertex.size Scalar or vector of node sizes.
#' @param vertex.color Scalar or vector of node colors.
#' @param vertex.frame.color Border color(s).
#' @return
#' Object returned by \code{draw_networkx_nodes}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # draw_networkx_nodes(...)
#' }
#' @export
draw_networkx_nodes <- function(G, pos, vertex.size = 8, vertex.color = "lightblue", vertex.frame.color = "darkblue") {
  G <- .drawing_as_igraph(G)
  if (is.list(pos)) pos <- do.call(rbind, pos)
  nodes <- as.character(igraph::V(G))
  idx <- match(nodes, rownames(pos))
  idx <- idx[!is.na(idx)]
  if (length(idx) == 0) {
    idx <- seq_len(min(igraph::vcount(G), nrow(pos)))
    if (length(idx) == 0) return(invisible(NULL))
  }
  xy <- pos[idx, , drop = FALSE]
  if (length(vertex.size) == 1) vertex.size <- rep(vertex.size, nrow(xy))
  if (length(vertex.color) == 1) vertex.color <- rep(vertex.color, nrow(xy))
  if (length(vertex.frame.color) == 1) vertex.frame.color <- rep(vertex.frame.color, nrow(xy))
  usr <- par("usr")
  r <- vertex.size * 0.01 * max(usr[2] - usr[1], usr[4] - usr[3])
  if (length(r) == 1) r <- rep(r, nrow(xy))
  symbols(xy[, 1], xy[, 2], circles = r, add = TRUE, inches = FALSE,
          fg = vertex.frame.color, bg = vertex.color)
}

# ------------------------------------------------------------------------------
# draw_networkx_edges (from nx_pylab)
# ------------------------------------------------------------------------------

#' Draw network edges
#'
#' @param G igraph object.
#' @param pos Matrix or list of node positions.
#' @param edge.color Color(s) for edges.
#' @param edge.width Width(s).
#' @param arrow.size Arrow head length for directed edges; 0 for no arrows.
#' @return
#' Object returned by \code{draw_networkx_edges}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # draw_networkx_edges(...)
#' }
#' @export
draw_networkx_edges <- function(G, pos, edge.color = "gray40", edge.width = 1, arrow.size = 0.2) {
  G <- .drawing_as_igraph(G)
  if (is.list(pos)) pos <- do.call(rbind, pos)
  nodes <- rownames(pos)
  if (is.null(nodes)) nodes <- as.character(seq_len(nrow(pos)))
  el <- igraph::as_edgelist(G, names = TRUE)
  if (nrow(el) == 0) return(invisible(NULL))
  from <- el[, 1]
  to <- el[, 2]
  from_idx <- match(from, nodes)
  to_idx <- match(to, nodes)
  valid <- !is.na(from_idx) & !is.na(to_idx)
  from_idx <- from_idx[valid]
  to_idx <- to_idx[valid]
  x0 <- pos[from_idx, 1]
  y0 <- pos[from_idx, 2]
  x1 <- pos[to_idx, 1]
  y1 <- pos[to_idx, 2]
  if (length(edge.width) == 1) edge.width <- rep(edge.width, length(x0))
  if (length(edge.color) == 1) edge.color <- rep(edge.color, length(x0))
  directed <- igraph::is_directed(G)
  for (i in seq_along(x0)) {
    if (directed && arrow.size > 0) {
      arrows(x0[i], y0[i], x1[i], y1[i], length = arrow.size, col = edge.color[i], lwd = edge.width[i])
    } else {
      segments(x0[i], y0[i], x1[i], y1[i], col = edge.color[i], lwd = edge.width[i])
    }
  }
  invisible(NULL)
}

# ------------------------------------------------------------------------------
# draw_networkx_labels (from nx_pylab)
# ------------------------------------------------------------------------------

#' Draw node labels
#'
#' @param G igraph object.
#' @param pos Matrix or list of node positions.
#' @param labels Character vector of labels (same order as nodes in G).
#' @param cex Font size.
#' @param col Label color.
#' @return
#' Object returned by \code{draw_networkx_labels}.
#' @seealso
#' \code{\link{RCausalML-package}}
#' @examples
#' \dontrun{
#' # Basic usage
#' # draw_networkx_labels(...)
#' }
#' @export
draw_networkx_labels <- function(G, pos, labels = NULL, cex = 0.8, col = "black") {
  G <- .drawing_as_igraph(G)
  if (is.list(pos)) pos <- do.call(rbind, pos)
  nodes <- as.character(igraph::V(G))
  idx <- match(nodes, rownames(pos))
  idx <- idx[!is.na(idx)]
  if (length(idx) == 0) idx <- seq_len(min(igraph::vcount(G), nrow(pos)))
  xy <- pos[idx, , drop = FALSE]
  if (is.null(labels)) labels <- nodes
  if (length(labels) != nrow(xy)) labels <- rep(labels, length.out = nrow(xy))
  text(xy[, 1], xy[, 2], labels = labels, cex = cex, col = col)
  invisible(NULL)
}
