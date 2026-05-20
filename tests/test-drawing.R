# Test drawing.R (layout and draw functions; port of NetworkX drawing module)
# Run from package root: Rscript tests/test-drawing.R
# Requires: igraph

pkg_root <- if (file.exists("R/drawing.R")) "." else if (file.exists("../R/drawing.R")) ".." else stop("Run from Causal_ML package root")
source(file.path(pkg_root, "R/drawing.R"))

if (!requireNamespace("igraph", quietly = TRUE)) {
  stop("Package 'igraph' is required for drawing tests. Install with install.packages('igraph').")
}

message("========== drawing.R tests ==========")
message("")

# --- Test graph: small DAG (3 nodes) ---
set.seed(42)
adj <- matrix(c(0, 1, 0, 0, 0, 1, 0, 0, 0), 3, 3)
G_igraph <- igraph::graph_from_adjacency_matrix(adj, mode = "directed")

# --- 1. rescale_layout ---
message("---- 1. rescale_layout ----")
pos_raw <- matrix(rnorm(6), 3, 2)
pos_rescaled <- rescale_layout(pos_raw, scale = 2)
stopifnot(is.matrix(pos_rescaled), nrow(pos_rescaled) == 3, ncol(pos_rescaled) == 2)
stopifnot(max(abs(pos_rescaled)) <= 2 + 1e-10)
message("  OK")
message("")

# --- 2. rescale_layout_dict (list and matrix) ---
message("---- 2. rescale_layout_dict ----")
pos_list <- list(a = c(0, 0), b = c(1, 1), c = c(-0.5, 0.5))
r1 <- rescale_layout_dict(pos_list, scale = 1)
stopifnot(is.list(r1), length(r1) == 3, all(names(r1) == c("a", "b", "c")))
pos_mat <- matrix(1:6, 3, 2)
rownames(pos_mat) <- c("x", "y", "z")
r2 <- rescale_layout_dict(pos_mat, scale = 1)
stopifnot(is.matrix(r2), nrow(r2) == 3)
message("  OK")
message("")

# --- 3. random_layout (igraph and matrix) ---
message("---- 3. random_layout ----")
pos_r <- random_layout(G_igraph, seed = 1)
stopifnot(is.matrix(pos_r), nrow(pos_r) == 3, ncol(pos_r) == 2, !is.null(rownames(pos_r)))
pos_r2 <- random_layout(adj, seed = 1)
stopifnot(is.matrix(pos_r2), nrow(pos_r2) == 3)
pos_r_list <- random_layout(G_igraph, seed = 1, as_list = TRUE)
stopifnot(is.list(pos_r_list), length(pos_r_list) == 3)
message("  OK")
message("")

# --- 4. circular_layout ---
message("---- 4. circular_layout ----")
pos_c <- circular_layout(G_igraph, scale = 1)
stopifnot(is.matrix(pos_c), nrow(pos_c) == 3, ncol(pos_c) == 2)
# Points on circle: equal radius (no duplicate angles)
d <- sqrt(rowSums(pos_c^2))
stopifnot(all(d > 0.01), max(d) - min(d) < 0.01)
message("  OK")
message("")

# --- 5. shell_layout ---
message("---- 5. shell_layout ----")
pos_s <- shell_layout(G_igraph, nlist = list(1, c(2, 3)))
stopifnot(is.matrix(pos_s), nrow(pos_s) == 3, ncol(pos_s) == 2)
pos_s2 <- shell_layout(G_igraph)
stopifnot(nrow(pos_s2) == 3)
message("  OK")
message("")

# --- 6. bipartite_layout ---
message("---- 6. bipartite_layout ----")
G_bip <- igraph::make_bipartite_graph(c(TRUE, TRUE, FALSE, FALSE), c(1,3, 1,4, 2,3, 2,4), directed = FALSE)
pos_b <- bipartite_layout(G_bip, align = "vertical")
stopifnot(is.matrix(pos_b), nrow(pos_b) == 4, ncol(pos_b) == 2)
pos_b2 <- bipartite_layout(G_igraph, nodes = c("1"), align = "horizontal")
stopifnot(nrow(pos_b2) == 3)
message("  OK")
message("")

# --- 7. spring_layout / fruchterman_reingold_layout ---
message("---- 7. spring_layout ----")
pos_fr <- spring_layout(G_igraph, seed = 1)
stopifnot(is.matrix(pos_fr), nrow(pos_fr) == 3, ncol(pos_fr) == 2)
pos_fr2 <- fruchterman_reingold_layout(G_igraph, seed = 1)
stopifnot(all.equal(pos_fr, pos_fr2))
message("  OK")
message("")

# --- 8. kamada_kawai_layout ---
message("---- 8. kamada_kawai_layout ----")
pos_kk <- kamada_kawai_layout(G_igraph)
stopifnot(is.matrix(pos_kk), nrow(pos_kk) == 3, ncol(pos_kk) == 2)
message("  OK")
message("")

# --- 9. spectral_layout ---
message("---- 9. spectral_layout ----")
pos_sp <- spectral_layout(G_igraph)
stopifnot(is.matrix(pos_sp), nrow(pos_sp) == 3, ncol(pos_sp) == 2)
message("  OK")
message("")

# --- 10. spiral_layout ---
message("---- 10. spiral_layout ----")
pos_spiral <- spiral_layout(G_igraph, equidistant = FALSE)
stopifnot(is.matrix(pos_spiral), nrow(pos_spiral) == 3, ncol(pos_spiral) == 2)
message("  OK")
message("")

# --- 11. multipartite_layout ---
message("---- 11. multipartite_layout ----")
G_mp <- igraph::graph_from_edgelist(matrix(c(1,2, 2,3, 3,4), ncol = 2, byrow = TRUE))
igraph::V(G_mp)$subset <- c("L1", "L1", "L2", "L2")
pos_mp <- multipartite_layout(G_mp, subset_key = "subset")
stopifnot(is.matrix(pos_mp), nrow(pos_mp) == 4, ncol(pos_mp) == 2)
layers <- list(L1 = c("1", "2"), L2 = c("3", "4"))
pos_mp2 <- multipartite_layout(G_mp, subset_key = layers)
stopifnot(nrow(pos_mp2) == 4)
message("  OK")
message("")

# --- 12. bfs_layout ---
message("---- 12. bfs_layout ----")
pos_bfs <- bfs_layout(G_igraph, start = 1)
stopifnot(is.matrix(pos_bfs), nrow(pos_bfs) == 3, ncol(pos_bfs) == 2)
pos_bfs2 <- bfs_layout(G_igraph, start = "1", align = "horizontal")
stopifnot(nrow(pos_bfs2) == 3)
message("  OK")
message("")

# --- 13. draw_network (no display) ---
message("---- 13. draw_network ----")
pdf(NULL)
draw_network(G_igraph, pos = pos_fr, with_labels = FALSE)
pos_drawn <- draw_network(G_igraph, layout = "circular", with_labels = TRUE)
dev.off()
stopifnot(is.matrix(pos_drawn), nrow(pos_drawn) == 3)
message("  OK")
message("")

# --- 14. draw_networkx_nodes / edges / labels (no display) ---
message("---- 14. draw_networkx_* ----")
pdf(NULL)
plot(NA, xlim = c(-2, 2), ylim = c(-2, 2))
draw_networkx_edges(G_igraph, pos_fr, edge.color = "gray", arrow.size = 0.15)
draw_networkx_nodes(G_igraph, pos_fr, vertex.size = 10, vertex.color = "lightblue")
draw_networkx_labels(G_igraph, pos_fr, labels = c("A", "B", "C"), cex = 0.7)
dev.off()
message("  OK")
message("")

# --- 15. Edge cases: empty graph, single node ---
message("---- 15. Edge cases ----")
G_empty <- igraph::make_empty_graph(0)
pos_empty <- random_layout(G_empty)
stopifnot(is.matrix(pos_empty), nrow(pos_empty) == 0)
pos_empty_c <- circular_layout(G_empty)
stopifnot(nrow(pos_empty_c) == 0)

G_one <- igraph::make_empty_graph(1)
pos_one <- spring_layout(G_one)
stopifnot(is.matrix(pos_one), nrow(pos_one) == 1, ncol(pos_one) == 2)
pos_one_c <- circular_layout(G_one)
stopifnot(nrow(pos_one_c) == 1)
message("  OK")
message("")

# --- 16. Input as adjacency matrix (non-igraph) ---
message("---- 16. Adjacency matrix input ----")
pos_adj <- spring_layout(adj, seed = 2)
stopifnot(is.matrix(pos_adj), nrow(pos_adj) == 3)
message("  OK")
message("")

message("========== test-drawing.R done ==========")
