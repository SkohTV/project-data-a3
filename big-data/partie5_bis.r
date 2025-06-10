library(dplyr)
library(nnet)
library(combinat)

df <- read.csv("result/export_IA.csv")

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

list_columns <- names(df)

columns_to_exclude <- c("id", "MMSI", "VesselName", "IMO", "CallSign")

df_excluded <- df %>% select(-(columns_to_exclude))
print(head(df))
