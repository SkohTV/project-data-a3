# Chargement des packages nécessaires
library(dplyr)
library(nnet)
library(combinat)

# Chargement des données
df <- read.csv("result/export_IA.csv")

# Mapping VesselType
map_vessel_type <- function(vessel_code) {
  case_when(
    vessel_code == 0 ~ "Not Available",
    vessel_code >= 1 & vessel_code <= 29 ~ "Other",
    vessel_code == 30 ~ "Fishing",
    vessel_code >= 31 & vessel_code <= 32 ~ "Tug Tow",
    vessel_code == 35 ~ "Military",
    vessel_code >= 36 & vessel_code <= 37 ~ "Pleasure Craft/Sailing",
    vessel_code >= 60 & vessel_code <= 69 ~ "Passenger",
    vessel_code >= 70 & vessel_code <= 79 ~ "Cargo",
    vessel_code >= 80 & vessel_code <= 89 ~ "Tanker",
    TRUE ~ "Other"
  )
}

# Colonnes à exclure (identifiants non prédictifs)
excluded_cols <- c("id", "MMSI", "IMO", "CallSign", "VesselType", "COG", "Heading", "LAT", "LON")

# Préparation des données (inclure toutes les colonnes numériques disponibles)
df_adapted <- df %>%
  mutate(VesselType = map_vessel_type(VesselType),
         VesselType = as.factor(VesselType)) %>%
  filter(!is.na(VesselType)) %>%
  # Traitement spécial pour SOG : exclure les valeurs 0
  filter(is.na(SOG) | SOG != 0) %>%
  select_if(is.numeric) %>%  # Sélectionner toutes les colonnes numériques
  bind_cols(VesselType = df %>% 
            mutate(VesselType = map_vessel_type(VesselType),
                   VesselType = as.factor(VesselType)) %>%
            filter(!is.na(VesselType)) %>%
            # Appliquer le même filtre SOG
            filter(is.na(SOG) | SOG != 0) %>%
            pull(VesselType)) %>%
  na.omit()  # Retirer les lignes avec des NA

# Identifier les colonnes prédictives (exclure VesselType et les colonnes d'identifiants)
predictive_cols <- names(df_adapted)[!names(df_adapted) %in% excluded_cols]
print(paste("Colonnes disponibles pour la prédiction:", paste(predictive_cols, collapse = ", ")))

# Fonction pour générer toutes les combinaisons possibles
generate_all_combinations <- function(cols) {
  all_combos <- list()
  for (i in 1:length(cols)) {
    combos_i <- combn(cols, i, simplify = FALSE)
    all_combos <- c(all_combos, combos_i)
  }
  return(all_combos)
}

# Fonction d'évaluation d'une combinaison de colonnes
evaluate_combination <- function(cols_combo, df_data, train_ratio = 0.8) {
  # Sélectionner les colonnes + VesselType
  df_subset <- df_data %>% select(all_of(c(cols_combo, "VesselType")))
    
  # Division train/test
  set.seed(579)  # Pour la reproductibilité
  train_indices <- sample(nrow(df_subset), floor(nrow(df_subset) * train_ratio))
  train_data <- df_subset[train_indices, ]
  test_data <- df_subset[-train_indices, ]
    
    
  # Entraînement du modèle
  model <- multinom(VesselType ~ ., data = train_data, trace = FALSE)
    
  # Prédictions
  predictions <- predict(model, newdata = test_data)
    
  # Calcul de la précision
  accuracy <- mean(predictions == test_data$VesselType, na.rm = TRUE)
    
  return(list(accuracy = accuracy, error = NULL, model = model))
}

# Test de toutes les combinaisons
print("Début du test de toutes les combinaisons de colonnes...")
all_combinations <- generate_all_combinations(predictive_cols)

# Limitation du nombre de combinaisons si trop important (pour éviter un temps de calcul excessif)
max_combinations <- 100
if (length(all_combinations) > max_combinations) {
  print(paste("Nombre de combinaisons réduit de", length(all_combinations), "à", max_combinations, "pour des raisons de performance"))
  all_combinations <- sample(all_combinations, max_combinations)
}

print(paste("Test de", length(all_combinations), "combinaisons..."))

# Évaluation de chaque combinaison
results <- data.frame(
  combination_id = integer(),
  columns = character(),
  nb_columns = integer(),
  accuracy = numeric(),
  error = character(),
  stringsAsFactors = FALSE
)

for (i in 1:length(all_combinations)) {
  combo <- all_combinations[[i]]
  result <- evaluate_combination(combo, df_adapted)
  
  results <- rbind(results, data.frame(
    combination_id = i,
    columns = paste(combo, collapse = ", "),
    nb_columns = length(combo),
    accuracy = result$accuracy,
    error = ifelse(is.null(result$error), "", result$error),
    stringsAsFactors = FALSE
  ))
  
  # Affichage du progrès
  if (i %% 10 == 0) {
    print(paste("Progression:", i, "/", length(all_combinations)))
  }
}

results <- results %>% 
  arrange(desc(accuracy)) %>%
  filter(error == "" | is.na(error))

print("=== TOP 10 DES MEILLEURES COMBINAISONS ===")
print(head(results, 10))

best_combination <- strsplit(results$columns[1], ", ")[[1]]
best_accuracy <- results$accuracy[1]

print(paste("=== MEILLEURE COMBINAISON TROUVÉE ==="))
print(paste("Colonnes:", paste(best_combination, collapse = ", ")))
print(paste("Précision:", round(best_accuracy * 100, 2), "%"))

# ON ENTRAÎNER AVEC LE MEILLEUR MODÈLE
print("=== ENTRAÎNEMENT DU MODÈLE FINAL ===")
df_best <- df_adapted %>% select(all_of(c(best_combination, "VesselType")))
final_model <- multinom(VesselType ~ ., data = df_best, trace = FALSE)

# TEST ÉCHANTILLON
set.seed(578)
df_sample <- df_best %>%
  select(-VesselType) %>%
  sample_n(min(1000, nrow(df_best)))

predicted_vessel_type <- predict(final_model, newdata = df_sample)

true_indices <- match(rownames(df_sample), rownames(df_best))
true_vessel_type <- df_best$VesselType[true_indices]

df_sample$VesselType_Predit <- predicted_vessel_type
df_sample$VesselType_Reel <- true_vessel_type
final_accuracy <- mean(df_sample$VesselType_Predit == df_sample$VesselType_Reel, na.rm = TRUE)

print(paste("=== RÉSULTATS FINAUX ==="))
print(paste("Précision du modèle optimisé:", round(final_accuracy * 100, 2), "%"))
print("Échantillon des prédictions:")
print(head(df_sample, 30))

# Sauvegarde des résultats
write.csv(results, "optimization_results.csv", row.names = FALSE)
print("Résultats sauvegardés dans 'optimization_results.csv'")

# Résumé des performances par nombre de colonnes
summary_by_nb_cols <- results %>%
  group_by(nb_columns) %>%
  summarise(
    nb_combinations = n(),
    accuracy_mean = mean(accuracy),
    accuracy_max = max(accuracy),
    accuracy_min = min(accuracy),
    .groups = 'drop'
  ) %>%
  arrange(desc(accuracy_max))

print("=== RÉSUMÉ PAR NOMBRE DE COLONNES ===")
print(summary_by_nb_cols)
