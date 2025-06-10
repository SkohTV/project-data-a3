data <- read.csv("../result/export_IA.csv")

data$coeff_taille <- data$Length/data$Width

result <- data[, c("MMSI", "coeff_taille")]

write.csv(result, "vessels_enhanced_parameters.csv", row.names = FALSE)
