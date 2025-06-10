library(dplyr)
library(sf)
library(geosphere)
library(rnaturalearth)
library(rnaturalearthdata)
library(ggplot2)

df <- read.csv("../result/export_IA.csv")

# MAPPING DES VESSELTYPE SUIVANT LA DOCUMENTATION -> https://api.vtexplorer.com/docs/ref-aistypes.html
# - MAPPING UNIQUEMENT SUR CES TYPECODE CI 
#(D'AUTRES TYPECODES PRÉSENTS COMME 5 OU 31 MAIS TRÈS PEU NOMBREUX VOIRE UNIQUE DONC INUTILE D'ENTRAÎNER UN MODÈLE POUR PRÉDIRE UN CAS UNIQUE)

map_VesselType <- function(vessel_code) {
  case_when(
    vessel_code >= 60 & vessel_code <= 69 ~ "Passenger",
    vessel_code >= 70 & vessel_code <= 79 ~ "Cargo",
    vessel_code >= 80 & vessel_code <= 89 ~ "Tanker",
  )
}

list_columns <- names(df) # LISTING DES COLONNES

columns_to_exclude <- c("id", "VesselName", "IMO", "CallSign", "TransceiverClass","COG", "Heading", "Length", "Width", "Cargo") # COLONNES À EXCLURE D'OFFICE ARBITRAIREMENT

df_excluded <- df %>% select(-(columns_to_exclude))

df_excluded <- df_excluded[round(df_excluded$Draft,2) == df_excluded$Draft,] # POUR DRAFT : ON ARRONDI CAR CERTAINS SONT DES ESTIMATIONS DE LA PARTIE 1 ET ÇA FAUSSE LA PERTINENCE DU CALCUL DU CHANGEMENT DE DRAFT SIGNIFICATIF

# CONVERSION DU DF EN SF
df_sf <- st_as_sf(df_excluded, coords = c("LON", "LAT"), crs = 4326, remove = FALSE)

# ON LOAD LA CARTE DES TERRES
onshore <- ne_download(scale = 50, type = "land", category = "physical", returnclass = "sf")
gulf_bbox <- st_bbox(c(xmin = -98, xmax = -80, ymin = 18, ymax = 31), crs = st_crs(onshore))
onshore_gulf <- st_crop(onshore, gulf_bbox)

# FILTRAGE DES NAVIRES À L'ARRÊT
df_sf <- df_sf %>% mutate(is_stopped = SOG <= 0.5)

# COMPARAISON POUR VOIR SI UN NAVIRE EST SUR LA TERRE
df_sf$on_land <- lengths(st_within(df_sf, onshore_gulf)) > 0

# ON TRANSFORME LES OBJETS EN PROJECTION IDENTIQUE POUR COMPARER
onshore_projection <- st_transform(onshore_gulf, crs = 3857)
df_projection <- st_transform(df_sf, crs = 3857)

# CALCUL DE LA DISTANCE MIN À LA CÔTE
df_sf$distance_to_coast_m <- st_distance(df_projection, onshore_projection) %>% apply(1, min)

# MARQUAGE DES VESSELS PROBABLEMENT DANS UN PORT
df_sf <- df_sf %>% mutate(likely_docked = is_stopped & (on_land | distance_to_coast_m <= 1500))

# FONCTION POUR CALCULER LE DELTA DRAFT LORS DES ARRÊTS/REDÉMARRAGES
calculate_draft_delta <- function(data) {
  data <- data %>%
    arrange(BaseDateTime) %>% # ON ORDONNE PAR TEMPS
    mutate( 
      period_start_draft = first(Draft), # ON PREND LE PREMIER DRAFT DE LA PÉRIODE
      period_end_draft = last(Draft), # ON PREND LE DERNIER DRAFT DE LA PÉRIODE
      period_duration = as.numeric(difftime(last(BaseDateTime), first(BaseDateTime), units = "hours")) # DURÉE DE LA PÉRIODE CONCERNÉE
    ) %>%
    ungroup() %>% # ON RÉORDONNE COMME AVANT
    mutate(
      prev_period_end_draft = lag(period_end_draft), # DRAFT DE LA PÉRIODE PRÉCÉDENTE
      draft_delta = abs(period_start_draft - prev_period_end_draft), # ON SOUSTRAIT LE DRAFT DE LA PÉRIODE PRÉCÉDENTE DU DRAFT DE LA PÉRIODE COURANTE
      significant_draft_change = !is.na(draft_delta) & draft_delta > 0.5 # ON CONSIDÈRE UN CHANGEMENT SIGNIFICATIF SI LE DELTA EST SUPÉRIEUR À 0.5 MÈTRES 
    )
  return(data)
}

# CALCUL DELTA DRAFT
df_sf <- df_sf %>% group_by(MMSI) %>% do(calculate_draft_delta(.)) %>% ungroup()

# ON MUTATE UNE COLONNE QUI VALIDE OU NON LES CONDITIONS SUIVANTES :
#   - BATEAU ARRÊTÉ (SOUS 1 KNOTS DE VITESSE)
#   - PAS SUR LA TERRE FERME (SELON LES LIBS CROISÉS QUI SONT CENSÉ SHAPE LÀ OÙ IL Y A DE LA TERRE DANS LE GOLFE)
#   - DISTANCE DE LA CÔTE SUPÉRIEUR À 15KM (OBLIGÉ DE METTRE UNE DISTANCE SUFFISAMENT IMPORTANTE POUR FILTRER L'INCERTITUDE DE LA LIBRAIRE QUI APRÈS CONSTAT DES RÉSULTATS PEUT FAIRE QUELQUES ERREURS DE JUGEMENT on_land)
#   - DRAFT SIGNIFICATIF -> VOIR MÉTHODE DE CALCUL PLUS HAUT

seuil_distance_offshore <- 15000

df_sf <- df_sf %>% mutate(likely_offshore_docked = is_stopped & (!(on_land)) & distance_to_coast_m > seuil_distance_offshore & map_VesselType(df_sf$VesselType) == "Tanker" & significant_draft_change)

df_sf <- df_sf[, !names(df_sf) %in% c("geometry")]

# ON ÉCHANTILLONNE POUR AFFICHAGE
sample_results <- df_sf %>% filter(likely_offshore_docked) %>% select(MMSI, BaseDateTime, LON, LAT, SOG, Draft, draft_delta, distance_to_coast_m, period_duration) %>% arrange(desc(draft_delta)) 

cat("\n -> ÉCHANTILLON DES 10 PREMIERS RÉSULTATS :\n")
print(sample_results)

df_export <- df_sf %>% st_drop_geometry() %>% filter(likely_offshore_docked) # SÉLECTION DES RÉSULTATS FINAUX

cat("\n -> RÉSULTATS FINAUX :\n")
cat("CAS DÉTÉCTÉS:", nrow(df_export), "\n")

# ON RÉALISE QUELQUES STATISTIQUES SUR LES CAS DÉTECTÉS POUR DEBUG ET COMPRENDRE
if(nrow(df_export) > 0) {
  final_stats <- df_export %>%
    summarise(
      navires_uniques = n_distinct(MMSI),
      draft_delta_moyen = mean(draft_delta, na.rm = TRUE),
      draft_delta_median = median(draft_delta, na.rm = TRUE),
      distance_moyenne = mean(distance_to_coast_m, na.rm = TRUE),
      duree_moyenne_h = mean(period_duration, na.rm = TRUE)
    )
  
  cat("\n -> STATISTIQUES DES CAS DÉTECTÉS :\n")
  print(final_stats)
}

# DIAGNOSTIC DÉTAILLÉ POUR COMPRENDRE POURQUOI PEU DE RÉSULTATS
cat(" -> DIAGNOSTIC DES CONDITIONS : \n")

total_tankers <- df_sf %>% filter(map_VesselType(VesselType) == "Tanker") %>% nrow()
cat("NOMBRE OBSERVATIONS TANKERS:", total_tankers, "\n")

stopped_tankers <- df_sf %>% filter(map_VesselType(VesselType) == "Tanker" & is_stopped) %>% nrow()
cat("NOMBRE OBSERVATIONS TANKERS ARRÊTÉS :", stopped_tankers, "\n")

offshore_tankers <- df_sf %>% filter(map_VesselType(VesselType) == "Tanker" & is_stopped & distance_to_coast_m > seuil_distance_offshore) %>% nrow()
cat("NOMBRE OBSERVATIONS TANKERS OFFSHORE (>15km):", offshore_tankers, "\n")

tankers_with_draft <- df_sf %>% filter(map_VesselType(VesselType) == "Tanker" & !is.na(Draft) & !is.na(draft_delta)) %>% nrow()
cat("NOMBRE OBSERVATIONS TANKER VALIDÉES PAR LES CONDITIONS CHOISIES:", tankers_with_draft, "\n")

# 5. DISTRIBUTION DES DELTAS DE DRAFT POUR LES TANKERS
draft_distribution <- df_sf  %>% filter(map_VesselType(VesselType) == "Tanker" & !is.na(draft_delta))  %>% summarise(
    count = n(),
    min_delta = min(draft_delta, na.rm = TRUE),
    q25_delta = quantile(draft_delta, 0.25, na.rm = TRUE),
    median_delta = median(draft_delta, na.rm = TRUE),
    q75_delta = quantile(draft_delta, 0.75, na.rm = TRUE),
    max_delta = max(draft_delta, na.rm = TRUE),
    count_over_1m = sum(draft_delta > 1.0, na.rm = TRUE),
    count_over_0_5m = sum(draft_delta > 0.5, na.rm = TRUE),
    count_over_0_3m = sum(draft_delta > 0.3, na.rm = TRUE)
  )

cat("\n -> DISTRIBUTION DES DELTAS DE DRAFT :\n")
print(draft_distribution)

# 6. VÉRIFICATION DES CONDITIONS POUR LES TANKERS
conditions <- df_sf  %>% filter(map_VesselType(VesselType) == "Tanker")  %>% summarise(
    total = n(),
    stopped = sum(is_stopped, na.rm = TRUE),
    not_on_land = sum(!on_land, na.rm = TRUE),
    offshore = sum(distance_to_coast_m > seuil_distance_offshore, na.rm = TRUE),
    stopped_offshore = sum(is_stopped & !on_land & distance_to_coast_m > seuil_distance_offshore, na.rm = TRUE),
    has_draft_change = sum(significant_draft_change, na.rm = TRUE),
    all_conditions = sum(is_stopped & !on_land & distance_to_coast_m > seuil_distance_offshore & significant_draft_change, na.rm = TRUE)
  )

cat("\n -> CONDITIONS :\n")
print(conditions)

# STATISTIQUES SUR LES DELTAS DE DRAFT POUR VALIDATION
draft_stats <- df_sf  %>% filter(!is.na(draft_delta) & map_VesselType(VesselType) == "Tanker")  %>% summarise(
    mean_draft_delta = mean(draft_delta, na.rm = TRUE),
    median_draft_delta = median(draft_delta, na.rm = TRUE),
    max_draft_delta = max(draft_delta, na.rm = TRUE),
    count_significant_changes = sum(significant_draft_change, na.rm = TRUE),
    total_observations = n()
  )

cat("\n -> STATISTIQUES DRAFT DELTA :\n")
print(draft_stats)

write.csv(df_export, "vessels_offshore_docked.csv", row.names = FALSE)

plot(st_geometry(onshore_gulf), col = "lightgreen", border = "darkgreen")
