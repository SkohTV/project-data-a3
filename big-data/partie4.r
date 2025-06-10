# library(ggplot2)
# library(dplyr)
# library(maps)
# library(corrplot)
# library(RColorBrewer)



if (!dir.exists("result")) {
  dir.create("result")
}

df_golfe <- read.csv("result/export_IA.csv")


# classification des types de bateaux
df_golfe$Type_Simple <- "Autre"
df_golfe$Type_Simple[df_golfe$VesselType >= 60 & df_golfe$VesselType <= 69] <- "Passenger"
df_golfe$Type_Simple[df_golfe$VesselType >= 70 & df_golfe$VesselType <= 79] <- "Cargo"
df_golfe$Type_Simple[df_golfe$VesselType >= 80 & df_golfe$VesselType <= 89] <- "Tanker"

# print(paste(nrow(df_golfe), "observations dans le golf du mexique")) # nb de bateaux dans le golf du mexique




# fonction pour faire la matrice de corrélation

analyse_matrice_correlation <- function(df_golfe, variables = c("SOG", "Length", "Width", "LAT", "LON", "Draft", "Heading", "COG")) {

  variables_num <- df_golfe[, variables]
  variables_num <- na.omit(variables_num) # pour etre sur qu'il n'y a pas de NA
  
  # échantillonnage pour performance (max 5000 lignes)
  # if (nrow(variables_num) > 5000) {
  #   variables_num <- variables_num[sample(nrow(variables_num), 5000), ]
  # }
  
  cor_matrix <- cor(variables_num) # calcul des corrélations
  
  # graphique de la matrice de corrélation
  png("result/matrice_correlation.png", width = 800, height = 800) # plot des correlations
  corrplot(cor_matrix, 
           method = "color",
           type = "upper",
           order = "hclust",
           tl.cex = 1.2,
           tl.col = "black",
           addCoef.col = "black",
           number.cex = 0.8,
           col = colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))(200)) # 200 est le nombre de couleurs générées par la palette
  title("Matrice de corrélation des variables numériques", line = 3)
  dev.off()
  
  return(cor_matrix)
}

# fonction pour faire l'analyse bivariée

analyse_correlation_bivariee <- function(df_golfe, var1, var2, type_col = "Type_Simple", 
                                        limite_var1 = c(0, 400), limite_var2 = c(0, 100), 
                                        nom_fichier = "correlation", max_points = 1000) {
  
  # données pour scatter plot # va filtrer df_golf selon les criteres: 
  df_analyse <- df_golfe[
    !is.na(df_golfe[[var1]]) & !is.na(df_golfe[[var2]]) & # si var1 et var2 = NA on supprime la ligne
      df_golfe[[var1]] > limite_var1[1] & df_golfe[[var1]] < limite_var1[2] & # limites var1
      df_golfe[[var2]] > limite_var2[1] & df_golfe[[var2]] < limite_var2[2] & # limites var2
      df_golfe[[type_col]] %in% c("Cargo", "Tanker", "Passenger"), # et on garde uniquement ces types de bateaux
  ]
  
  # filtrage?
  if (nrow(df_analyse) > max_points) { # on ne prend que max_points points aléatoires car sinon il y a trop de points sur le plot
    df_analyse <- df_analyse[sample(nrow(df_analyse), max_points), ]
  }
  
  p <- ggplot(df_analyse, aes_string(x = var1, y = var2, color = type_col)) +
    geom_point(alpha = 0.6, size = 2) +
    geom_smooth(method = "lm", color = "red", linewidth = 1) + # par defaut il y a se = TRUE qui met le gris autour de la droite, c'est l'incertitude a 95%, c'est pour voir la fiablilité de la droite
    labs(title = paste("Corrélation", var1, "vs", var2, "par type de navire"),
         x = var1, y = var2,
         color = "Type") +
    theme_minimal() +
    scale_color_brewer(type = "qual", palette = "Set1")
  
  # print(p)
  ggsave(paste0("result/", nom_fichier, "_", var1, "_", var2, ".png"), p, width = 10, height = 7)
  
  # calcul de la corrélation affichage dans el terminal
  correlation <- cor(df_analyse[[var1]], df_analyse[[var2]])
  cat("Corrélation", var1, "-", var2, ":", round(correlation, 3), "\n")
  
  return(list(plot = p, correlation = correlation, data = df_analyse))
}

# fonction pour faire les tableaux croisés et le test du chi2 le plot est un mosaicplot



# fonction aui implement le chi2

analyse_tableau_croise <- function(df_golfe, var_cat1, var_cat2, seuil_effectif = 100) {
  
  
  # tableau croisé avec filtrage des NA
  tableau_data <- df_golfe[!is.na(df_golfe[[var_cat2]]) & 
                            df_golfe[[var_cat1]] %in% c("Cargo", "Tanker", "Passenger"), ]
  
  
  if (nrow(tableau_data) > 5000) { # on prend 5000 valeurs max
    tableau_data <- tableau_data[sample(nrow(tableau_data), 5000), ]
  }
  
  
  tableau_croise <- table(tableau_data[[var_cat1]], tableau_data[[var_cat2]])
  tableau_croise <- tableau_croise[rowSums(tableau_croise) > seuil_effectif, ]
  
  print(paste("TABLEAU CROISÉ :", var_cat1, "vs", var_cat2))
  print(tableau_croise)
  
  print("Pourcentages par ligne :")
  proportions <- round(prop.table(tableau_croise, 1) * 100, 1)
  print(proportions)
  
  

  # chi2
  chi2_test <- chisq.test(tableau_croise)
  cat("Chi^2 =", round(chi2_test$statistic, 2), "\n")
  cat("p-value =", format(chi2_test$p.value, scientific = TRUE), "\n")
  
  
  # mosaicplot
  png(paste0("result/mosaicplot_", var_cat1, "_", var_cat2, ".png"), width = 800, height = 600)
  mosaicplot(tableau_croise, 
             main = paste("Mosaicplot :", var_cat1, "vs", var_cat2),
             xlab = var_cat1, 
             ylab = var_cat2,
             color = brewer.pal(4, "Set3"),
             cex.axis = 0.8)
  dev.off()
  
  return(list(tableau = tableau_croise, chi2 = chi2_test, proportions = proportions))
}

# heatmap pour les correlations par type de navire (cargo, tanker, passenger)
analyse_correlations_par_type <- function(df_golfe, variables = c("SOG", "Length", "Width"), 
                                         type_col = "Type_Simple", 
                                         types = c("Cargo", "Tanker", "Passenger")) {
  
  cor_by_type <- data.frame()
  
  for (type in types) {
    df_type <- df_golfe[df_golfe[[type_col]] == type, variables]
    df_type <- na.omit(df_type)
    
    if (nrow(df_type) > 50) {  # assez de données
      cor_sog_length <- cor(df_type[[variables[1]]], df_type[[variables[2]]])
      cor_sog_width <- cor(df_type[[variables[1]]], df_type[[variables[3]]])
      cor_length_width2 <- cor(df_type[[variables[2]]], df_type[[variables[3]]])
      
      cor_by_type <- rbind(cor_by_type, data.frame(
        Type = type,
        Cor_Var1_Var2 = cor_sog_length,
        Cor_Var1_Var3 = cor_sog_width,
        Cor_Var2_Var3 = cor_length_width2
      ))
    }
  }
  
  names(cor_by_type) <- c("Type", 
                          paste(variables[1], variables[2], sep = "_"),
                          paste(variables[1], variables[3], sep = "_"),
                          paste(variables[2], variables[3], sep = "_"))
  
  print("Correlations par type de navire :")
  print(cor_by_type)
  
  return(cor_by_type)
}



# fonction pour faire l'anova: quantitative vs qualitative


analyse_anova <- function(df_golfe, var_continue, var_categorie = "Type_Simple", 
                         limite_var = c(0, 25), types = c("Cargo", "Tanker", "Passenger")) {
  
  
  df_anova <- df_golfe[
    !is.na(df_golfe[[var_continue]]) & !is.na(df_golfe[[var_categorie]]) &     # pas de NA
      df_golfe[[var_continue]] > limite_var[1] & df_golfe[[var_continue]] <= limite_var[2] &   # limites réalistes
      df_golfe[[var_categorie]] %in% types, # types sélectionnés
  ]
  
  
  df_anova <- do.call(rbind, lapply(split(df_anova, df_anova[[var_categorie]]), function(groupe) { # on prend un échantillon de max 1500 elements pour avoir un F coherent
    if (nrow(groupe) > 1500) {
      return(groupe[sample(nrow(groupe), 1500), ])
    } else {
      return(groupe)
    }
  }))
  
  
  # affichage dans la console
  
  print(paste(var_continue, "moyenne par", var_categorie, ":"))

  moyennes <- aggregate(df_anova[[var_continue]], by = list(df_anova[[var_categorie]]), mean)
  names(moyennes) <- c(var_categorie, paste("Moyenne", var_continue))

  print(moyennes)
  
  

  # ANOVA

  formule <- as.formula(paste(var_continue, "~", var_categorie))
  modele <- aov(formule, data = df_anova)
  resultat <- summary(modele)
  
  print(resultat)
  
  
  p_value <- resultat[[1]][1, "Pr(>F)"] # le Pr(>F) est = au p-value, dans le cas de notre dataset, il est tres faible
  f_value <- resultat[[1]][1, "F value"] # dans notre cas il est tres elevé (250 pour un échantillon de 5000 lignes)
  print(paste("F =", round(f_value, 2)))
  print(paste("p-value =", format(p_value, scientific = TRUE)))
  

  
  # boxplot
  p <- ggplot(df_anova, aes_string(x = var_categorie, y = var_continue, fill = var_categorie)) +
    geom_boxplot() +
    labs(title = paste(var_continue, "par", var_categorie, "(ANOVA)"),
         subtitle = paste("F =", round(f_value, 1), ", p <", format(p_value, digits = 2)),
         x = var_categorie, y = var_continue) +
    theme_minimal() +
    theme(legend.position = "none") # on n'affiche pas la legende par defaut
  
  ggsave(paste0("result/anova_", var_continue, "_", var_categorie, ".png"), p, width = 8, height = 6)
  
  return(list(plot = p, anova = resultat, p_value = p_value, f_value = f_value, moyennes = moyennes))
}





# # fonction pour faire le pdf récapitulatif (pas nécessaire)
# generer_rapport_pdf <- function(liste_plots, liste_resultats, cor_matrix = NULL, cor_by_type = NULL, nom_fichier = "rapport_analyse") {
  
#   pdf(paste0("result/", nom_fichier, ".pdf"), width = 12, height = 8)
  
#   # MATRICE DE CORRÉLATION en première page
#   if(!is.null(cor_matrix)) {
#     corrplot(cor_matrix, 
#              method = "color",
#              type = "upper",
#              order = "hclust",
#              tl.cex = 1.2,
#              tl.col = "black",
#              addCoef.col = "black",
#              number.cex = 0.8,
#              col = colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))(200))
#     title("Matrice de corrélation des variables numériques", line = 3)
#   }
  
#   # graphiques ggplot
#   for(plot in liste_plots) {
#     if("ggplot" %in% class(plot)) {
#       print(plot)
#     }
#   }
  
#   # MOSAICPLOT (recherche dans les résultats)
#   for(result in liste_resultats) {
#     if(!is.null(result$tableau)) {
#       mosaicplot(result$tableau, 
#                  main = "Mosaicplot : Type de navire vs Catégorie de vitesse",
#                  xlab = "Type de navire", 
#                  ylab = "Catégorie de vitesse",
#                  color = brewer.pal(4, "Set3"),
#                  cex.axis = 0.8)
#     }
#   }
  
#   # page de résultats texte
#   plot.new()
#   text(0.5, 0.95, "RÉSULTATS DES ANALYSES STATISTIQUES", cex = 1.8, font = 2, col = "darkblue")
  
#   # position verticale pour les résultats
#   y_pos <- 0.85
  
#   # affichage des résultats principaux
#   for(i in 1:length(liste_resultats)) {
#     result <- liste_resultats[[i]]
    
#     if(!is.null(result$correlation)) {
#       text(0.05, y_pos, paste("Corrélation :", round(result$correlation, 3)), cex = 1.1, adj = 0)
#       y_pos <- y_pos - 0.05
#     }
    
#     if(!is.null(result$chi2)) {
#       text(0.05, y_pos, paste("Chi² =", round(result$chi2$statistic, 2), 
#                              ", p =", format(result$chi2$p.value, scientific = TRUE)), cex = 1.1, adj = 0)
#       y_pos <- y_pos - 0.05
#     }
    
#     if(!is.null(result$p_value)) {
#       text(0.05, y_pos, paste("ANOVA: p =", round(result$p_value, 6)), cex = 1.1, adj = 0)
#       y_pos <- y_pos - 0.05
#     }
    
#     y_pos <- y_pos - 0.03 # espace entre les analyses
#   }
  
#   # CORRÉLATIONS PAR TYPE (à la fin)
#   if(!is.null(cor_by_type)) {
#     y_pos <- y_pos - 0.05 # espace supplémentaire
#     text(0.05, y_pos, "CORRÉLATIONS PAR TYPE DE NAVIRE :", cex = 1.3, font = 2, adj = 0, col = "darkgreen")
#     y_pos <- y_pos - 0.05
    
#     # affichage du tableau cor_by_type
#     for(i in 1:nrow(cor_by_type)) {
#       type_name <- cor_by_type[i, 1]  # première colonne = Type
#       text(0.05, y_pos, paste(type_name, ":"), cex = 1.1, font = 2, adj = 0)
#       y_pos <- y_pos - 0.04
      
#       # affichage des corrélations pour ce type
#       for(j in 2:ncol(cor_by_type)) {
#         col_name <- names(cor_by_type)[j]
#         correlation_value <- cor_by_type[i, j]
#         text(0.1, y_pos, paste("  ", col_name, ":", round(correlation_value, 3)), cex = 1.0, adj = 0)
#         y_pos <- y_pos - 0.03
#       }
#       y_pos <- y_pos - 0.02 # espace entre les types
#     }
#   }
  
#   dev.off()
# }




####################################################################################################


# matrice
cor_matrix <- analyse_matrice_correlation(df_golfe)

####################################################################################################



# correlation bivariée quantitative-quantitative
result_longueur_largeur <- analyse_correlation_bivariee(df_golfe, "Length", "Width", 
                                                       limite_var1 = c(0, 400), limite_var2 = c(0, 100),
                                                       nom_fichier = "correlation_longueur_largeur")

####################################################################################################

# autre exemple de correlation bivariée (quantitative-quantitative)
result_vitesse_longueur <- analyse_correlation_bivariee(df_golfe, "Length", "SOG", 
                                                       limite_var1 = c(0, 300), limite_var2 = c(0, 25),
                                                       nom_fichier = "correlation_vitesse_longueur")


####################################################################################################

# tableau croisé et test du chi2 pour faire du qualitatif-qualitatif


df_golfe$Vitesse_Cat <- cut(df_golfe$SOG, 
                               breaks = c(0, 1, 5, 15, 25), 
                               labels = c("Arrêté", "Lent", "Moyen", "Rapide"),
                               include.lowest = TRUE)

result_tableau_croise <- analyse_tableau_croise(df_golfe, "Type_Simple", "Vitesse_Cat")


####################################################################################################

cor_by_type <- analyse_correlations_par_type(df_golfe)

####################################################################################################


# exemple d'anova entre une variable quantitative et une variable qualitative
result_anova <- analyse_anova(df_golfe, "SOG", "Type_Simple")

####################################################################################################








# pour la création du rapport PDF

# liste_plots <- list(result_longueur_largeur$plot, result_vitesse_longueur$plot, result_anova$plot)
# liste_resultats <- list(result_longueur_largeur, result_vitesse_longueur, result_tableau_croise, result_anova)

# generer_rapport_pdf(liste_plots, liste_resultats, cor_matrix, cor_by_type, "fonctionnalite_4")







# #corrélations (comme dans votre code original)
# cat("Longueur-Largeur correlations :", round(result_longueur_largeur$correlation, 3), "\n")
# cat("Vitesse-Longueur correlations:", round(result_vitesse_longueur$correlation, 3), "\n")

#Tankers → Souvent arrêtés ou lents (chargement/déchargement)
#Cargo → Vitesses intermédiaires
#Passenger → Plus souvent à vitesse moyenne/rapide

# donc il y a un tres for lien entre type et rapidité donc le p-value est proche de 0, c'est bien (ce n'est pas lié au hazard)
# pour le chi^2 il est elevé car on a un échantillon énorme, est il nous dit qu'il y a une relation entre 2 valeurs de tategorie (type_bateau (cargo, passenger, tanker) et  vitesse_categorie (arreté, lent, moyen rapide))



