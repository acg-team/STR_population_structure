# library
library(ggplot2)
library(reshape2)
library(scales) 
library(ggthemes)

setwd("C:\\Users\\Public\\Files\\PopSTR")
df_prob <- read.csv(".\\vis\\figure2\\1kg_h3a_prob.csv")

d <- dist(df_prob[,c(1,2,3,4,5,6,7)])
hc <- hclust(d)
df_prob$sample <- factor(df_prob$sample, levels = df_prob$sample[hc$order])

###### sg_pop
#df_prob$sample <- rownames(df_prob)
melted_df <- melt(df_prob,
                  id.vars = c("sample", "pop"))

# hgdp+sgdp
melted_df$super <- factor(melted_df$super,
                           levels = c(
                             "AFR", "MIE", "EUR", "SAS", "CAS", "EAS",
                             "AMR",  "OCE"
                             ))

tableau10medium <- tableau_color_pal("Superfishel Stone")(7)
#  c("#59A14F", "#F28E2B","#B07AA1", "#4E79A7", "#E15759")
  #tableau_color_pal("Superfishel Stone")(5)
ggplot(melted_df, aes(fill = variable, 
                      y = value, 
                      x = sample)) + 
  geom_bar(position = "fill",stat = "identity") +
  facet_grid(~factor(pop),
             scales = "free", space = "free") +
  scale_fill_manual(values = tableau10medium) +
  theme(
    text=element_text(size=18),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid = element_blank(),
    panel.spacing = unit(0.1, "lines"),
    panel.background = element_blank(),
    strip.background.x = element_blank(),
    strip.text.x.top = element_text(
      # size = 15,
      angle = 45,
      hjust = 0,
      margin = margin(b = -18, unit = "pt")),
    strip.clip = "off",
    plot.margin = margin(t = 50, unit = "pt")
  ) + labs(fill ="") + xlab("") + ylab("Prediction Probabilities")

############ superpopulation
df_subset <- df_prob#[1:100,]

# h3a
df_subset$sample <- rownames(df_subset)
df_subset$true_pop <- "H3Africa"
##

melted_df <- melt(df_subset,
                  id.vars = c("sample", "Population"))

# Stacked + percent
ggplot(melted_df, aes(fill = variable, 
        y = value, 
        x = sample)) + 
        geom_bar(position = "fill",stat = "identity") +
        facet_grid(~factor(Population),
                     #, levels = c("AFR", "AMR", "EUR", "SAS", "EAS")),
                   scales = "free", space = "free") +
        theme(text = element_text(size = 15),
          axis.title.x=element_blank(),
              axis.text.x=element_blank(),
              axis.ticks.x=element_blank()) +
         labs(fill ="") 

# subpopulation

df_subset <- df_prob[df_prob$Superpopulation == "AFR",]

df_subset <- df_subset[c(unique(df_subset$true_pop), "sample", "true_pop")]

melted_df <- melt(df_subset,
                  id.vars = c("sample", "true_pop"))
#h3a
df_subset <- df_prob[c("ACB", "ASW", "ESN", "GWD", "LWK", "MSL", "YRI", "sample", "Population")]

melted_df <- melt(df_subset,
                  id.vars = c("sample", "Population"))

ggplot(melted_df, aes(fill = variable, 
                      y = value, 
                      x = sample)) + 
  geom_bar(position = "fill", stat = "identity") +
  facet_grid(~factor(Population),
             scales = "free", space = "free") +
  theme(text = element_text(size = 15),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  scale_fill_brewer(palette="Set2") +
  labs(fill ="")

