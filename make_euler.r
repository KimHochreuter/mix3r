library(eulerr)
library(data.table)
library(grid)

args <- commandArgs(trailingOnly = TRUE)
# args[1] : path to *.parameters.csv file produced by extract_p.py; "bdbiobank_bdclinical_bdselfreport_sep12.parameters.csv"
# args[2] : path to output file; "bdbiobank_bdclinical_bdselfreport_sep12.parameters.euler.png"
# args[3], args[4], args[5] : labels for trait 1, trait 2 and trait 3 in Euler diagram; "BD biobank" "BD clinical" "BD self-repotr"
# args[6], args[7], args[8] : hex colors for trait 1, trait 2 and trait 3 in Euler diagram; "#77AADD" "#EE8866" "#EEDD88"

if (length(args) < 5) {
  stop("Missing arguments!")
} else {
  fname <- args[1]
  outf <- args[2]
  labels <- c(args[3], args[4], args[5])
  if (length(args) == 8) {
    colors <- c(args[6], args[7], args[8])
  } else {
    colors <- c("#77AADD", "#EE8866", "#EEDD88")
  }
} 

df <- fread(fname)
i_best_run = which.min(df$rank_p_proportion_deviation_from_median)
best = df[i_best_run]

acceptable_negative_gap = -1E-6
p1 <- best$p_1 - best$p_12 - best$p_13 + best$p_123
if (p1<0 & p1>acceptable_negative_gap) {p1 <- 0}
p2 <- best$p_2 - best$p_12 - best$p_23 + best$p_123
if (p2<0 & p2>acceptable_negative_gap) {p2 <- 0}
p3 <- best$p_3 - best$p_13 - best$p_23 + best$p_123
if (p3<0 & p3>acceptable_negative_gap) {p3 <- 0}
p12 <- best$p_12 - best$p_123
if (p12<0 & p12>acceptable_negative_gap) {p12 <- 0}
p13 <- best$p_13 - best$p_123
if (p13<0 & p13>acceptable_negative_gap) {p13 <- 0}
p23 <- best$p_23 - best$p_123
if (p23<0 & p23>acceptable_negative_gap) {p23 <- 0}
p123 <- best$p_123
if (p123<0 & p1>acceptable_negative_gap) {p123 <- 0}
factor = 10 # otherwise behaves badly with small numbers
vec2plot = c("1"=factor*p1,"2"=factor*p2,"3"=factor*p3,"1&2"=factor*p12,"1&3"=factor*p13,"2&3"=factor*p23,"1&2&3"=factor*p123)

fit <- euler(vec2plot, input="disjoint", shape="ellipse")

fmt_p <- function(x) {
  ifelse(is.na(x), "NA", sprintf("%.3g", x))
}

fmt_r <- function(x) {
  ifelse(is.na(x), "NA", sprintf("%.2f", x))
}

concordance_from_rho <- function(x) {
  x <- pmax(pmin(x, 1), -1)
  0.5 + asin(x) / pi
}

mean_or_na <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  mean(x, na.rm=TRUE)
}

summary_values <- data.table(
  Metric = c(
    "Trait 1 only", "Trait 2 only", "Trait 3 only",
    "1 + 2 only", "1 + 3 only", "2 + 3 only", "1 + 2 + 3",
    "rg 1-2", "rg 1-3", "rg 2-3",
    "rho 1-2", "rho 1-3", "rho 2-3",
    "concordant 1-2", "concordant 1-3", "concordant 2-3"
  ),
  Value = c(
    fmt_p(mean_or_na(df$p_1 - df$p_12 - df$p_13 + df$p_123)),
    fmt_p(mean_or_na(df$p_2 - df$p_12 - df$p_23 + df$p_123)),
    fmt_p(mean_or_na(df$p_3 - df$p_13 - df$p_23 + df$p_123)),
    fmt_p(mean_or_na(df$p_12 - df$p_123)),
    fmt_p(mean_or_na(df$p_13 - df$p_123)),
    fmt_p(mean_or_na(df$p_23 - df$p_123)),
    fmt_p(mean_or_na(df$p_123)),
    fmt_r(mean_or_na(df$rg_12)),
    fmt_r(mean_or_na(df$rg_13)),
    fmt_r(mean_or_na(df$rg_23)),
    fmt_r(mean_or_na(df$rho_12)),
    fmt_r(mean_or_na(df$rho_13)),
    fmt_r(mean_or_na(df$rho_23)),
    fmt_r(mean_or_na(concordance_from_rho(df$rho_12))),
    fmt_r(mean_or_na(concordance_from_rho(df$rho_13))),
    fmt_r(mean_or_na(concordance_from_rho(df$rho_23)))
  )
)

draw_summary_table <- function(summary_values, labels) {
  correlation_rows <- summary_values[8:16]
  correlation_rows[, Metric := c(
    paste0("r[g]~'", labels[1], "-", labels[2], "'"),
    paste0("r[g]~'", labels[1], "-", labels[3], "'"),
    paste0("r[g]~'", labels[2], "-", labels[3], "'"),
    paste0("rho[beta]~'", labels[1], "-", labels[2], "'"),
    paste0("rho[beta]~'", labels[1], "-", labels[3], "'"),
    paste0("rho[beta]~'", labels[2], "-", labels[3], "'"),
    paste0("'CR ", labels[1], "-", labels[2], "'"),
    paste0("'CR ", labels[1], "-", labels[3], "'"),
    paste0("'CR ", labels[2], "-", labels[3], "'")
  )]

  draw_block <- function(rows, title, layout_col) {
    grid.text(title, x=0.02, y=0.86, just=c("left", "top"), gp=gpar(fontsize=11, fontface="bold"))
    y <- c(0.72, 0.64, 0.56, 0.43, 0.35, 0.27, 0.17, 0.10, 0.03)
    grid.text(parse(text=rows$Metric), x=0.03, y=y, just=c("left", "center"), gp=gpar(fontsize=9))
    grid.text(rows$Value, x=0.95, y=y, just=c("right", "center"), gp=gpar(fontsize=9, fontface="bold"))
    separator_y <- c(0.50, 0.22)
    grid.lines(x=unit(c(0.03, 0.95), "npc"), y=unit(rep(separator_y[1], 2), "npc"), gp=gpar(col="grey82", lwd=1))
    grid.lines(x=unit(c(0.03, 0.95), "npc"), y=unit(rep(separator_y[2], 2), "npc"), gp=gpar(col="grey82", lwd=1))
  }
  draw_block(correlation_rows, "Mean pairwise correlations", 1)
}

draw_plot <- function() {
  euler_grob <- plot(fit,
       fills = list(fill=colors, alpha=alpha),
       labels = list(labels=labels, col=label_color, fontsize=label_fontsize),
       edges = list(col=edge_color, lex=edge_width),
       lty = edge_lty,
       quantities = list(type="percent", cex=1, fontsize=quantile_fontsize)
  )

  grid.newpage()
  pushViewport(viewport(layout=grid.layout(2, 1, heights=unit(c(4.4, 1.45), "null"))))
  pushViewport(viewport(layout.pos.row=1))
  grid.draw(euler_grob)
  popViewport()
  pushViewport(viewport(layout.pos.row=2))
  grid.rect(gp=gpar(col=NA, fill="white"))
  grid.lines(x=unit(c(0.04, 0.96), "npc"), y=unit(c(0.98, 0.98), "npc"), gp=gpar(col="grey75", lwd=1))
  draw_summary_table(summary_values, labels)
  popViewport()
  popViewport()
}

edge_lty = 1:1:1
alpha = 1
label_color = "black"
edge_color = "white"
label_fontsize = 24
quantile_fontsize = 24
edge_width = 4

png(filename=paste0(outf,".png"), width=700, height=820, units="px", pointsize=12, bg="white", res=NA)
draw_plot()
supressed_output = dev.off()

svg(filename=paste0(outf,".svg"), width=7, height=8.2, bg="white", onefile=TRUE)
draw_plot()
supressed_output = dev.off()

cat(paste0("Figure saved to: ", outf, ".png", "\n"))
cat(paste0("Figure saved to: ", outf, ".svg", "\n"))
