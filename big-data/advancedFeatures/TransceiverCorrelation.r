library(dplyr)
library(VIM)

blank_df <- read.csv("../sujet/vessel-total-clean.csv")
blank_df[blank_df == "\\N"] <- NA

df_Transceiver <- blank_df %>%
    mutate(
        nb_NA_in_line = rowSums(is.na(.)),
        TransceiverClass = as.factor(TransceiverClass)
    ) %>%
    select(TransceiverClass, nb_NA_in_line)

library(ggplot2)

p <- ggplot(df_Transceiver, aes(x = TransceiverClass, y = nb_NA_in_line, fill = TransceiverClass)) +
  geom_boxplot(alpha = 0.7, outlier.colour = "red", outlier.size = 1.5) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 4, color = "darkred") +
  stat_summary(fun = mean, geom = "text", aes(label = round(..y.., 1)), 
               vjust = -0.7, color = "darkred", fontface = "bold", size = 4) +
  scale_fill_manual(values = c("A" = "#3498db", "B" = "#e74c3c")) +
  labs(
    title = "Classe de Transpondeur VS NA",
    x = "TransceiverClass",
    y = "Nombre de Na",
    fill = "Classe"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold", color = "#2c3e50"),
    plot.subtitle = element_text(hjust = 0.5, size = 12, color = "#7f8c8d"),
    axis.title = element_text(face = "bold", size = 12),
    legend.position = "bottom",
    panel.grid.major = element_line(color = "#ecf0f1", linewidth = 0.5),
    panel.background = element_rect(fill = "#fafafa")
  )

ggsave("transceiver_na_correlation.png", p, 
       width = 10, height = 7, dpi = 300, bg = "white")

tapply(df_Transceiver$nb_NA_in_line, df_Transceiver$TransceiverClass, mean, na.rm = TRUE)