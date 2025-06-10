library(randomForest)

df <- read.csv("sujet/vessel-total-clean.csv")
# df <- read.csv("~/AIS_2024_01_01.csv")

summary(df)

colonnes_numeriques <- c("LAT", "LON", "SOG", "COG", "Heading", "VesselType", "Status", "Length", "Width", "Draft", "Cargo") # LISTING DES VARIABLES NUMÉRIQUES
df[df == "\\N"] <- NA # REMPLACEMENT DES VALEURS "\\N" PAR NA, CAR INITIALEMENT, LES VALEURS MANQUANTES ÉTAIENT MARQUÉES PAR "\\N"

# TRANSFORMATION DES COLONNES QUI SONT CENSÉES ÊTRE NUMÉRIQUE EN VALEUR NUMÉRIQUE
for(col in colonnes_numeriques) {
    if(col %in% names(df)) {
        df[[col]] <- as.numeric(df[[col]])
    }
}

# POUR CHAQUE VARIABLES, TRANSFORMER EN NA SI EN DEHORS DES CONDITIONS
df$LAT[df$LAT < -90 | df$LAT > 90] <- NA
df$LON[df$LON < -110 | df$LON > 110] <- NA
df$SOG[df$SOG > 27 | df$SOG < 0] <- NA
df$COG[df$COG > 360 | df$COG < 0] <- NA
df$Heading[df$Heading > 360 | df$Heading < 0] <- NA
df$Length[df$Length < 0 | df$Length > 500] <- NA
df$Width[df$Width < 0 | df$Width > 100] <- NA  
df$Draft[df$Draft < 0 | df$Draft > 30] <- NA
df$VesselType[df$VesselType == 0] <- NA
df$Status[df$Status == 15] <- NA
df$Cargo[df$Cargo == 0] <- NA

l <- nrow(df)
df <- df[!is.na(df$LON), ]
print(paste("Supprimé", l - nrow(df), "rows avec des LON incorrectes"))

# ON CONVERTIR STATUS EN FACTEUR CAR C'EST LA SEULE VALEUR NUMÉRIQUE À PRÉDIRE QU'IL FAUT CONSIDÉRER COMME UNE CLASSE CAR C'EST UN TYPECODE
df$Status <- as.factor(df$Status)

# ON S'ASSURE QUE LES COLONNES SONT BIEN PRÉSENTES ET ON SUPPRIME LES LIGNES EN DOUBLE (MÊME SI C'EST IMPROBABLE DANS CE CAS)
# df <- unique(df)

# ON REMPLACE LES VALEURS MANQUANTES DE CARGO ET VESSELTYPE EN SUIVANT LA LOGIQUE SUIVANTE:
# SI CARGO EST NA ET VESSELTYPE N'EST PAS NA, ON ATTRIBUE LE TYPECODE DE VESSELTYPE À CARGO CAR C'EST LA NATURE DES DEUX VALEURS SONT IDENTIQUES
# SI CARGO N'EST PAS NA ET VESSELTYPE EST NA, ON ATTRIBUE LE TYPECODE DE CARGO À VESSELTYPE CAR IDEM
# SI CARGO ET VESSELTYPE NE SONT PAS NA, ON GARDE CARGO SI VESSELTYPE EST UN MULTIPLE DE 10 (CAR C'EST UN TYPECODE) ET ON ATTRIBUE CARGO À VESSELTYPE CAR CARGO EST PLUS PRÉCIS # nolint: line_length_linter.
for (i in 1:nrow(df)){
    if (is.na(df$Cargo[i]) && !is.na(df$VesselType[i])){
        df$Cargo[i] <- df$VesselType[i]
    }
    else if (!is.na(df$Cargo[i]) && is.na(df$VesselType[i])){
        df$VesselType[i] <- df$Cargo[i]
    }
    else if (!is.na(df$Cargo[i]) && !is.na(df$VesselType[i])){
        if(df$VesselType[i] %% 10 == 0 && df$Cargo[i] %% 10 != 0){
            df$VesselType[i] <- df$Cargo[i]
        }
    }
}

variables <- list(
    "id", 
    "MMSI", 
    "BaseDateTime", 
    "LAT", 
    "LON", 
    "SOG", 
    "COG", 
    "Heading",
    "VesselName", 
    "IMO", 
    "CallSign", 
    "VesselType", 
    "Status", 
    "Length", 
    "Width", 
    "Draft", 
    "Cargo", 
    "TransceiverClass"
)

# LISTE DES CONTRAINTES À APPLIQUER POUR LES FUTURS VALEURS PRÉDITES COMBLANT LES NA
contraintes <- list(
    LAT = c(-90, 90),
    LON = c(-180, 180),
    SOG = c(0, 27),
    COG = c(0, 360),
    Heading = c(0, 360),
    VesselType = c(1, 99),
    Length = c(1, 500),
    Width = c(1, 100),
    Draft = c(0, 30),
    Cargo = c(1, 99)
)

appliquer_contraintes <- function(valeur, nom_variable) {
    # appliquer_contraintes : POUR LA VALEUR "valeur" ATTRIBUÉE À LA VARIABLE "nom_variable"
    # S'ASSURE QUE LA CONTRAINTE ASSIGNÉE DANS LE TABLEAU contraintes POUR CETTE VARIABLE, EST BIEN RESPECTÉE

    if(nom_variable %in% names(contraintes)) {
        limites <- contraintes[[nom_variable]]
        valeur <- pmax(valeur, limites[1])
        valeur <- pmin(valeur, limites[2])
    }
    return(valeur)
}

chercher_valeurs_manquantes <- function(df, taille_echantillon = 30000) {
    # chercher_valeurs_manquantes : FONCTION RÉALISANT L'ENSEMBLE DES PROCESS POUR CHERCHER QUELLES VALEURS ATTRIBUÉES À NA DU DATASET df
    # -> LE CHOIX DE L'ASSIGNATION DES VALEURS EST FAIT PAR RANDOM FOREST SUR UN BACKTEST DE taille_echantillon (30 000 PAR DEFAUT), MODÈLE D'IMPUTATION PAR ARBRE DE DÉCISION

    cat("\n *** DÉBUT DU PROCESS DE RECHERCHE DE VALEURS MANQUANTES AVEC RANDOM FOREST *** \n")

    na_count <- sapply(df, function(x) sum(is.na(x))) # COMPTAGE DU NOMBRE DE NA
    na_percent <- round(na_count / nrow(df) * 100, 2) # PART DE NA D'UNE VAR PAR RAPPORT AU TOTAL DE NA
    
    # DEBUG DU COMPTAGE DE NA
    cat("-> Part de valeurs manquantes par variable:\n")
    for(i in 1:length(na_percent)) {
        if(na_percent[i] > 0) {
            cat(sprintf("%s: %.2f%% (%d valeurs)\n", 
                names(na_percent)[i], na_percent[i], na_count[i]))
        }
    }

    set.seed(123) # SEED POUR REPRODUCTIBILITÉ DE L'ÉCHANTILLONNAGE

    if(nrow(df) > taille_echantillon) {
        if("VesselType" %in% names(df) && !all(is.na(df$VesselType))) {
            # SI VESSELTYPE EST PRÉSENT ET QU'IL N'EST PAS TOUT EN NA, ON VA ÉCHANTILLONNER PAR TYPE DE VESSELTYPE
            types_uniques <- unique(df$VesselType[!is.na(df$VesselType)]) # ON IDENTIFIE LES TYPESCODES POSSIBLES POUR UN VESSELTYPE
            echantillon_indices <- c() # ON INIT UNE LISTE QUI CONTIENDRA LES INDICES DES ÉCHANTILLONS APRÈS SAMPLE
            
            for(type in types_uniques) {
                indices_type <- which(df$VesselType == type) # OBTENTION DES INDICES POUR CHAQUE À LIGNE À CHAQUE TYPE
                n_type <- min(length(indices_type), max(1, round(taille_echantillon * length(indices_type) / nrow(df)))) # ON DÉTERMINE LE NOMBRE D'INDICES À ÉCHANTILLONNER POUR CHAQUE TYPE
                echantillon_indices <- c(echantillon_indices, sample(indices_type, n_type))
            }
            # SI LE NOMBRE D'INDICES ÉCHANTILLONNÉS EST MOINS QUE LA TAILLE DE L'ÉCHANTILLON, ON AJOUTE DES INDICES SUPPLÉMENTAIRES
            if(length(echantillon_indices) < taille_echantillon) {
                indices_restants <- setdiff(1:nrow(df), echantillon_indices)
                indices_supplementaires <- sample(indices_restants, min(length(indices_restants), taille_echantillon - length(echantillon_indices))) # ON ÉCHANTILLONNE DES INDICES RESTANTS POUR ATTEINDRE LA TAILLE DE L'ÉCHANTILLON
                echantillon_indices <- c(echantillon_indices, indices_supplementaires)
            }
            df_sample <- df[echantillon_indices, ] # ON CRÉE L'ÉCHANTILLON FINAL EN SÉLECTIONNANT LES LIGNES PAR LES INDICES ÉCHANTILLONNÉS

        }
    }
    
    cat(sprintf("Échantillon créé: %d lignes\n", nrow(df_sample)))
    
    variables_numeriques <- names(contraintes)
    
    cat("\n *** BACKTEST DU SAMPLE ***\n")
    
    # ON VA TRAITER CHAQUE VARIABLE NUMÉRIQUE POUR REMPLACER LES VALEURS MANQUANTES PAR RANDOM FOREST
    for(var_cible in variables_numeriques) { # FILTRE POUR TRAITER UNIQUEMENT LES VARIABLES NUMÉRIQUES POUR L'INSTANT
        
        if(sum(is.na(df[[var_cible]])) == 0) next # SI AUCUNE VALEUR MANQUANTE, ON SAUTE CETTE VARIABLE
        
        cat(sprintf("Traitement avec RF de %s...\n", var_cible))
        
        # DÉMARRAGE DU TRAITEMENT POUR LA VARIABLE var_cible
        predicteurs <- setdiff(variables_numeriques, var_cible) # ON VA UTILISER TOUTES LES AUTRES VARIABLES NUMÉRIQUES COMME PRÉDICTEURS
        predicteurs <- predicteurs[predicteurs %in% names(df_sample)] # ON NE GARDE QUE LES PRÉDICTEURS PRÉSENTS DANS L'ÉCHANTILLON
        # ON AFFICHE LES PRÉDICTEURS UTILISÉS
        cat(sprintf("  Prédicteurs utilisés: %s\n", paste(predicteurs, collapse = ", ")))
    
        # ON PRÉPARE LES DONNÉES D'ENTRAÎNEMENT EN NE GARDANT QUE LES LIGNES SANS NA POUR LA VARIABLE CIBLE ET LES PRÉDICTEURS
        colonnes_entrainement <- c(var_cible, predicteurs) # ON CRÉE UNE LISTE DES COLONNES À UTILISER POUR L'ENTRAÎNEMENT
        donnees_completes <- df_sample[complete.cases(df_sample[, colonnes_entrainement]), ] # ON NE GARDE QUE LES LIGNES SANS NA POUR LA VARIABLE CIBLE ET LES PRÉDICTEURS
        
        # ON POSE LA FORMULE POUR LE MODÈLE RANDOM FOREST & ON TRAIN LE MODÈLE PAR ARBRE DE DÉCISION
        formule <- as.formula(paste(var_cible, "~", paste(predicteurs, collapse = " + ")))
        rf_model <- randomForest(formule, data = donnees_completes, ntree = 100, na.action = na.omit, nodesize = 10)
        
        # MAINTENANT QUE LE MODÈLE EST ENTRAÎNÉ, ON VA PREDIRE LES VALEURS MANQUANTES
        cat(sprintf("  *** MODÈLE RANDOM FOREST ENTRAÎNÉ ***\n"))

        indices_na <- which(is.na(df[[var_cible]])) # ON IDENTIFIE LES LIGNES AVEC NA POUR LA VARIABLE CIBLE
        
        if(length(indices_na) > 0) {
            # SI DES LIGNES AVEC NA SONT TROUVÉES, ON VA PRÉPARER LES DONNÉES POUR LA PRÉDICTION
            donnees_prediction <- df[indices_na, predicteurs, drop = FALSE] # ON NE GARDE QUE LES PRÉDICTEURS POUR LES LIGNES AVEC NA
            
            # ON REMPLACE LES NA DANS LES PRÉDICTEURS PAR DES VALEURS PAR DÉFAUT
            for(pred in predicteurs) {
                na_pred <- is.na(donnees_prediction[[pred]])
                if(any(na_pred)) {
                    if(pred %in% c("VesselType", "Cargo", "Status")) {
                        # ON METS POUR L'INSTANT UN MODE POUR LES VARIABLES CATÉGORIELLES
                        mode_val <- as.numeric(names(sort(table(df[[pred]]), decreasing = TRUE))[1])
                    } else {
                        # ET ON MET UN MEDIAN POUR LES VARIABLES NUMÉRIQUES
                        mode_val <- median(df[[pred]], na.rm = TRUE)
                    }
                    if(pred != "Status") {
                        donnees_prediction[[pred]][na_pred] <- mode_val
                    }
                }
            }
                
            # ON FAIT LA PRÉDICTION AVEC LE MODÈLE RANDOM FOREST & ON APPLIQUE LES CONTRAINTES
            predictions <- predict(rf_model, donnees_prediction)
            predictions <- appliquer_contraintes(predictions, var_cible)
                
            # REMPLACEMENT DES VALEURS PRÉDITES DANS LE DATAFRAME ORIGINAL
            df[[var_cible]][indices_na] <- predictions
                
            cat(sprintf("  %s: %d valeurs imputées par Random Forest\n", var_cible, length(indices_na)))
            }
            
    }
    
    # POUR STATUS QUI EST LA SEULE VARIABLE CATÉGORIELLE À PRÉDIRE, ON VA UTILISER UN MODÈLE SPÉCIFIQUE, MALGRÉ QU'IL S'AGISSE D'UNE VARIABLE NUMÉRIQUE 
    # C'EST UN TYPECODE, DONC ON VA LE TRAITER COMME UNE VARIABLE CATÉGORIELLE
    if("Status" %in% names(df) && sum(is.na(df$Status)) > 0) {
        cat("Traitement de Status (variable catégorielle)...\n")
        
        # ON VA SÉLECTIONNER LES PRÉDICTEURS POUR STATUS
        predicteurs_status <- c("VesselType", "Cargo", "LAT", "LON", "SOG", "Length", "Width", "Draft")
        
        if(length(predicteurs_status) > 0) {

            colonnes_entrainement <- c("Status", predicteurs_status)
            donnees_completes <- df_sample[complete.cases(df_sample[, colonnes_entrainement]), ]
            
            if(nrow(donnees_completes) > 0) {
                # ON ENTRAÎNE CETTE FOIS-CI UN MODÈLE RANDOM FOREST DE CLASSIFICATION POUR STATUS
                formule <- as.formula(paste("Status", "~", paste(predicteurs_status, collapse = " + ")))
                rf_model_status <- randomForest(formule, data = donnees_completes, ntree = 100, na.action = na.omit)
                
                # ON PROCÈDE DE LA MÊME FAÇON QUE POUR LES AUTRES VARIABLES ENSUITE ...

                indices_na_status <- which(is.na(df$Status))
                
                if(length(indices_na_status) > 0) {
                    donnees_prediction_status <- df[indices_na_status, predicteurs_status, drop = FALSE]
                    
                    for(pred in predicteurs_status) {
                        na_pred <- is.na(donnees_prediction_status[[pred]])
                        if(any(na_pred)) {
                            if(pred %in% c("VesselType", "Cargo")) {
                                mode_val <- as.numeric(names(sort(table(df[[pred]]), decreasing = TRUE))[1])
                            } else {
                                mode_val <- median(df[[pred]], na.rm = TRUE)
                            }
                            donnees_prediction_status[[pred]][na_pred] <- mode_val
                        }
                    }
                    
                    predictions_status <- predict(rf_model_status, donnees_prediction_status)
                    
                    df$Status[indices_na_status] <- predictions_status
                    
                    cat(sprintf("  Status: %d valeurs imputées par Random Forest\n", length(indices_na_status)))
                }
            }
        }
    }
    
    # ON FAIT UN RAPPORT FINAL SUR LES VALEURS MANQUANTES POUR COMPARER AVANT ET APRÈS L'IMPUTATION
    cat("\n *** RAPPORT FINAL ***\n")
    na_count_final <- sapply(df, function(x) sum(is.na(x)))
    
    cat("Amélioration des valeurs manquantes:\n")
    for(var in names(na_count)) {
        if(na_count[var] > 0) {
            reduction <- na_count[var] - na_count_final[var]
            if(reduction > 0) {
                cat(sprintf("%s: %d → %d (-%d, %.1f%%)\n", 
                    var, na_count[var], na_count_final[var], 
                    reduction, (reduction/na_count[var])*100))
            }
        }
    }
    
    total_reduction <- sum(na_count) - sum(na_count_final)
    cat(sprintf("\nTotal des valeurs manquantes: %d → %d (-%d)\n", 
        sum(na_count), sum(na_count_final), total_reduction))
    cat(sprintf("Taux de complétion: %.1f%%\n", 
        (total_reduction/sum(na_count))*100))
    
    cat("\n *** PROCESSUS TERMINÉ ***\n")

    return(df)
}

df_complete <- chercher_valeurs_manquantes(df, taille_echantillon = 30000)

write.csv(df_complete, 'result/export_IA.csv', row.names = FALSE)
