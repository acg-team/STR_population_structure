library(tidyverse)
library(ggthemes)
library(ggplot2)
library(dplyr)
library(tidyr)
library(UpSetR)
library(ComplexHeatmap)

setwd("C:\\Users\\Public\\Files\\PopSTR")

pos <- read.csv(".\\vis\\figure4\\kg_pos.csv")
neg <- read.csv(".\\vis\\figure4\\kg_neg.csv")

K <- ncol(pos)
top_k <- round(nrow(pos) * 0.05)

# Build the list of sets: union of top_k loci from hs_neg and hs_pos
sets <- lapply(seq_len(K), function(i){
  top_neg <- order(neg[[i]], decreasing = TRUE)[1:top_k]
  top_pos <- order(pos[[i]], decreasing = TRUE)[1:top_k]
  unique(c(top_neg, top_pos))
})
names(sets) <- paste0("S", seq_len(K))

s_order <- paste0("S", seq_len(K))
m1 <- make_comb_mat(sets)
m1 <- m1[comb_size(m1) >=500]

ht <- UpSet(m1,
      set_order = s_order,
      pt_size = unit(5, "mm"),
      top_annotation = HeatmapAnnotation(
        "Number of STRs" = anno_barplot(comb_size(m1), 
           #ylim = c(0, max(cs)*1.1),
           border = FALSE, 
           gp = gpar(fontsize = 15, fill = "black"), 
           height = unit(5, "cm"),
           axis_param = list(gp = gpar(fontsize=12))),
           annotation_name_side = "left",
           annotation_name_rot = 90),
      row_names_gp = gpar(fontsize = 12),  
      right_annotation = NULL,
      comb_order = order(comb_size(m1), decreasing = TRUE))

ht = draw(ht, padding = unit(c(5, 5, 10, 5), "mm"))
od = column_order(ht)
cs <- comb_size(m1)
decorate_annotation("Number of STRs", {
  grid.text(cs[od], x = seq_along(cs), y = unit(cs[od], "native") + unit(1, "pt"), 
            default.units = "native", just = c("left", "bottom"), 
            gp = gpar(fontsize = 10, col = "#404040"), rot = 45)
})

