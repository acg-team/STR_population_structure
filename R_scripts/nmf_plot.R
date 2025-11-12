library(tidyverse)
library(ggthemes)
library(ggplot2)

setwd("C:\\Users\\Public\\Files\\PopSTR\\NMF")

# Read ancestry proportions
prop <- read.table(".\\hs\\hs_12_neg.csv", sep = ",", header = TRUE)
d <- dist(prop[,-ncol(prop)])
hc <- hclust(d)
prop$sample <- factor(prop$sample, levels = prop$sample[hc$order])
# "..\\kg1\\sample_type.csv"    
sample_type <- read.table("..\\overlap\\hg_sg\\hs_samples.csv", 
                          sep = ",", header = TRUE)
# (sample_type$Superpopulation == "AFR") |
#sample_type <- sample_type[ (sample_type$Superpopulation == "AFR") | (sample_type$Superpopulation == "H3Africa"),]
#sample_type[sample_type$Population == "Cameroon_24", "Population"] <- "Cameroon"
#sample_type[sample_type$Population == "Cameroon_26", "Population"] <- "Cameroon"
#sample_type$Superpopulation <- sample_type$Superpopulation %>% str_remove(fixed("(HGDP)"))
#sample_type <- sample_type[sample_type$data == "hgdp",]

# Reshape to long format for ggplot2
prop_long <- prop %>%
  pivot_longer(
    cols = starts_with("X"),
    names_to   = "Component",
    values_to  = "Proportion",
    names_transform = list(
      Component = \(x) paste0("S", as.integer(sub("^X", "", x)) + 1)
    )
  )
prop_long <- merge(prop_long, sample_type, by.x = "sample", by.y = "Sample")

prop_long$Component <- factor(prop_long$Component, 
            levels = paste("S", seq(1,12), sep = ""))

prop_long$pop_id <- factor(prop_long$pop_id, levels = c("AFR", "MIE",
                          "EUR", "CAS", "SAS", "EAS", "AMR", "OCE"))

prop_long$Population <- factor(prop_long$Population, 
                               levels = c("LWK", "ESN", "YRI", "MSL", "GWD", "ACB", "ASW",
                                          "CLM", "MXL", "PUR", "PEL",
                                          "TSI", "IBS", "GBR", "CEU", "FIN",
                                          "PJL", "GIH", "ITU", "STU", "BEB",
                                          "CDX", "KHV", "CHS", "CHB", "JPT"))
prop_long$Population <- factor(prop_long$Population,
                               levels = c('Benin', 'Burkina_Faso-Ghana', 'Nigeria', 
                                          'Cameroon', 'Mali', 'Botswana', 'Zambia', 
                                          "LWK", "ESN", "YRI", "MSL", "GWD", "ACB", "ASW" ))


color_pale <- tableau_color_pal("Miller Stone")(11) #
color_pale <- append(color_pale,"#bb7693")

color_pale <- tableau_color_pal("Tableau 20")(12)
## sgdp hgdp
#prop_sub <- prop_long[prop_long$pop_id == "OCE",]
ggplot(prop_long, aes(x = sample, y = Proportion, fill = Component)) +
  geom_bar(stat = "identity") +
  facet_grid(. ~ pop_id, 
             scales = "free_x", 
             space = "free_x") +
  scale_fill_manual(values = color_pale) +
  theme(
    text=element_text(size=20),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid = element_blank(),
    panel.spacing = unit(0.1, "lines"),
    panel.background = element_blank(),
    strip.background.x = element_blank(),
    strip.text.x.top = element_text(
      size = 15, #15
      angle = 45,
      hjust = 0,
      margin = margin(b = -20, unit = "pt")), #20
    strip.clip = "off",
    plot.margin = margin(t = 50, unit = "pt"),
    legend.title=element_blank()
  ) +
  labs(#title = "NMF Admixture Plot", 
    x = "", 
    y = "")


 