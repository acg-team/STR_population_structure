library(ggplot2)
library(ggpubr)
library(RColorBrewer)
library(ggthemes)
library(shades)

setwd("C:\\Users\\Public\\Files\\PopSTR")

df_sig_pos <- read.csv(".\\vis\\figure4\\kg_sig_neg.csv")

# Make composite key: period + sig/nonsig
df_sig_pos$group <- paste0(df_sig_pos$period, "_", 
                           ifelse(df_sig_pos$is_significant, "sig", "ns"))
df_sig_pos$period <- factor(df_sig_pos$period, levels = seq(6,1))
# Base colors for periods
cols <- tableau_color_pal("Classic 20")(20)
# Expand: bold (sig) vs desaturated (ns)
pal <- c(
  "1_sig" = cols[1],
  "1_ns"  = saturation(cols[2], 0.2),
  "2_sig" = cols[3],
  "2_ns"  = saturation(cols[4], 0.2),
  "3_sig" = cols[5],
  "3_ns"  = saturation(cols[6], 0.2),
  "4_sig" = cols[7],
  "4_ns"  = saturation(cols[8], 0.2),
  "5_sig" = cols[9],
  "5_ns"  = saturation(cols[10], 0.2),
  "6_sig" = cols[17],
  "6_ns"  = saturation(cols[18], 0.2)
)

# hs
#df_sig_pos$component <- factor(df_sig_pos$component,
#    levels = c("S1", "S10", "S2", "S6", "S8", "S9", "S3", "S11", "S7", "S4", "S5"))
# 1kg
df_sig_pos$component <- factor(df_sig_pos$component,
                               levels = c("S1", "S11", "S2", "S6", "S7", "S8",
                                          "S10", "S12", "S3", "S9", "S5","S4"))
#df_sig_pos <- df_sig_pos[df_sig_pos$is_significant == "True",]

ggplot(df_sig_pos, aes(x=component, y=enrichment_ratio, fill = group)) +
  geom_col(position="dodge") +
  geom_hline(yintercept=1, linetype="dashed", color="red") +
  scale_fill_manual(values=pal, guide = "none") +  # remove doubled legend
  coord_flip() + 
  #scale_x_discrete(position = "top") +
  theme_bw(base_size = 20) +
  theme(
   # axis.title = element_text(face = "bold"),
    panel.grid = element_blank(),   # remove vertical gridlines
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.title = element_blank()
  ) +
  labs(x = "", y = "Enrichment ratio") +
 # ylim(0,2.5) +
 geom_vline(xintercept = seq(1.5, length(unique(df_sig_pos$component)) + 0.5, 1),
               color = "grey80", linewidth = 0.6)+
   scale_y_reverse(limits = c(2.5, 0))



period_per <- read.csv(".\\vis\\figure4\\hs_region_per_pos.csv")
#period_per$period <- factor(period_per$period)
pal <-  tableau_color_pal("Tableau 10")(10)
period_per$set <- factor(period_per$set,
                         levels = paste("S", seq(12,1), sep = ""))
ggplot(period_per, aes(x = set, y = proportion, fill = region)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values=pal) +
  theme_bw(base_size = 23) +
  coord_flip() + 
  theme(
    #axis.title = element_text(face = "bold"),
    panel.grid = element_blank(),   # remove vertical gridlines
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_blank()
  ) +
  labs(x = "", y = "fraction") 


period_per <- read.csv(".\\vis\\figure4\\kg_period_per_pos.csv")
period_per$period <- factor(period_per$period)
pal <-  tableau_color_pal("Classic 10")(10)
cols <- saturation(c(pal[1], pal[2], pal[3], pal[4], pal[5], pal[9]), 0.7)
period_per$set <- factor(period_per$set,
                         levels = paste("S", seq(12,1), sep = ""))
ggplot(period_per, aes(x = set, y = proportion, fill = period)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = cols)+
  theme_bw(base_size = 22) +
  coord_flip() + 
  theme(
    #axis.title = element_text(face = "bold"),
    panel.grid = element_blank(),   # remove vertical gridlines
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_blank()
  ) +
  labs(x = "", y = "fraction") 
  #guides(fill = guide_legend(nrow = 1)) 


























