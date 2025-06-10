

if (!dir.exists("result")) {
  dir.create("result")
}


df_golfe <- read.csv("result/export_IA.csv")



# classification des types de bateaux
df_golfe$Type_Simple <- "Autre"
df_golfe$Type_Simple[df_golfe$VesselType >= 60 & df_golfe$VesselType <= 69] <- "Passenger"
df_golfe$Type_Simple[df_golfe$VesselType >= 70 & df_golfe$VesselType <= 79] <- "Cargo"
df_golfe$Type_Simple[df_golfe$VesselType >= 80 & df_golfe$VesselType <= 89] <- "Tanker"

# print(paste( nrow(df_golfe), "observations dans le golf")) # nb de bateaux dans le golf du mexique
# on l'affiche dans les stats

# graph 1 : RÉPARTITION DES TYPES


type_counts <- aggregate(list(n = df_golfe$Type_Simple), # regroupe (aggregate) les lignes de "df_golfe" selon la colonne "Type_Simple" et pour chaque groupe, 
                                                         # elle compte le nombre d’occurrences (length) et stocke ce nombre dans la colonne n.
                         by = list(Type_Simple = df_golfe$Type_Simple), 
                         FUN = length)
type_counts <- type_counts[order(-type_counts$n), ] # tri décroissant selon le nombre d'occurence grace a la colonne n qu'on a créé au dessus
type_counts$pourcentage <- round(type_counts$n / sum(type_counts$n) * 100, 1) # ajoute la colonne pourcentage qui est le pourcentage du type dans la data frame



p1 <- ggplot(type_counts, aes(x = reorder(Type_Simple, n), y = n, fill = Type_Simple)) + # affiche le plot
  geom_col() +
  geom_text(aes(label = paste0(n, "\n(", pourcentage, "%)")), hjust = -0.1) +
  coord_flip() +
  labs(title = "Répartition des bateaux par type",
       x = "Type", y = "Nombre") +
  theme_minimal() +
  theme(legend.position = "none")



# print(p1)
ggsave("result/repartition_types.png", p1, width = 10, height = 6)


# graph 2 : HISTOGRAMME DES VITESSES


# échantillonnage pour pas avoir trop de points
df_vitesse <- df_golfe[!is.na(df_golfe$SOG) & df_golfe$SOG <= 25, ] # On garde que les lignes où la colonne SOG n'est pas NA et où SOG est inférieure ou égale à 25 noeuds (pas nécessaire grace a la partie 1 de robin mais au cas où...)



# le probleme qu'on a c'est qu'on a trop de points a afficher donc si le data frame > 5000 lignes on en prend 5000 aléatoire
# c'est pas dingue de faire ca, a voir si on modif par la suite
# filtrage?
if (nrow(df_vitesse) > 5000) { 
  df_vitesse <- df_vitesse[sample(nrow(df_vitesse), 5000), ]
}

p2 <- ggplot(df_vitesse, aes(x = SOG)) + # affichage du plot
  geom_histogram(bins = 20, fill = "steelblue", alpha = 0.7) +
  labs(title = "Distribution des vitesses",
       x = "Vitesse (noeuds)", y = "Nombre de bateaux") +
  theme_minimal()

# print(p2)
ggsave("result/histogramme_vitesses.png", p2, width = 10, height = 6)



# ce plot est le meme que avant sans les navires < 1 noeuds
df_vitesse2 <- df_vitesse[df_vitesse$SOG >= 1, ] # On garde que les lignes où la colonne SOG n'est pas NA et où SOG est inférieure ou égale à 25 noeuds (pas nécessaire grace a la partie 1 de robin mais au cas où...)

p2_temp <- ggplot(df_vitesse2, aes(x = SOG)) + # affichage du plot
  geom_histogram(bins = 20, fill = "steelblue", alpha = 0.7) +
  labs(title = "Distribution des vitesses",
       x = "Vitesse (noeuds)", y = "Nombre de bateaux") +
  theme_minimal()

# print(p2)
ggsave("result/histogramme_vitesses_sans_arret.png", p2_temp, width = 10, height = 6)


# graph 3 : VITESSE PAR TYPE (BOXPLOT)


# échantillonnage par type
df_boxplot <- df_golfe[!is.na(df_golfe$SOG) & df_golfe$SOG <= 25 & df_golfe$Type_Simple != "Autre", ] # on garde uniquement les SOG != NA, SOG <= 25 et Type_Simple != Autre



df_boxplot <- do.call(rbind, lapply(split(df_boxplot, df_boxplot$Type_Simple), function(x) { # on fait des sous groupe de Type_Simple (donc soit Tanker, Passager ou Cargo)
    x[sample(nrow(x), min(200, nrow(x))), ]                                                  # pour chaque groupe on selectionne 200 lignes aléatoirement
}))                                                                                          # on rassempble les 3 sous groupes dans 1 data frame
                                                                                             # donc on a un échantillon équillibré par type



p3 <- ggplot(df_boxplot, aes(x = Type_Simple, y = SOG, fill = Type_Simple)) + # plot 3
  geom_boxplot() +
  labs(title = "Vitesse par type de bateau",
       x = "Type", y = "Vitesse (noeuds)") +
  theme_minimal() +
  theme(legend.position = "none")

# print(p3)
ggsave("result/vitesse_par_type.png", p3, width = 10, height = 6)


# graph 4 : PORTS LES PLUS UTILISÉS


# on trouve les zones avec le plus de bateaux (= ports)
# grille simple 1 degres x 1 degres
lat_breaks <- seq(20, 35, by = 1) # lat de 20 a 35 degres
lon_breaks <- seq(-100, -80, by = 1) # long de -100 à -80 degres

ports_grid <- expand.grid( # on genere toutes les combinaisons possible des centres de chaque parties (comme la grille fait de 1 en 1, le centre est a +0.5)
  lat = lat_breaks[-length(lat_breaks)] + 0.5,
  lon = lon_breaks[-length(lon_breaks)] + 0.5
)

ports_grid$count <- 0
for (i in 1:nrow(ports_grid)) { #on compte le nombre de bateaux qui sont a +- 0.5 degres du centre d'un carré (défini au dessus)
  zone <- df_golfe[
    df_golfe$LAT >= ports_grid$lat[i] - 0.5 &
    df_golfe$LAT < ports_grid$lat[i] + 0.5 &
    df_golfe$LON >= ports_grid$lon[i] - 0.5 &
    df_golfe$LON < ports_grid$lon[i] + 0.5,
  ]
  ports_grid$count[i] <- nrow(zone)
}

# top 10 des zones les plus fréquentées
top_ports <- ports_grid[order(-ports_grid$count), ][1:10, ] # on extrait les 10 carrés les plus fréquentés car sinon on a trop de valeur
top_ports$zone <- paste0("Zone ", seq_len(nrow(top_ports))) # c'est aussi pour limiter le plot et ne pas qu'il bug avec trop de valeur (on veut quelque chose de clair et lisible) (grace a cette ligne on donne un nom a la zone (zonr 1, zone 2, ...... par la suite on pourra ajouter le nom des port directemeent))

p4 <- ggplot(top_ports, aes(x = reorder(zone, count), y = count)) + # plot 4
  geom_col(fill = "orange", alpha = 0.7) +
  geom_text(aes(label = count), hjust = -0.1) +
  coord_flip() +
  labs(title = "Top 10 des zones les plus fréquentées",
       x = "Zone géographique", y = "Nombre de bateaux") +
  theme_minimal()

# print(p4)
ggsave("result/zones_frequentees.png", p4, width = 10, height = 6)


# graph 5 : CARTE SIMPLE DES ZONES


# données cartographiques
world_map <- map_data("world")

# on garde que les zones avec du trafic significatif
# filtrage?
zones_actives <- ports_grid[ports_grid$count > 100, ] # on garde uniquement les zones où il y a + de 100 bateaux

p5 <- ggplot() + # plot 4 avec les polynomes (US, et Mexique) du module pour avoir un effet cart
  # fond de carte
  geom_polygon(data = world_map, 
               aes(x = long, y = lat, group = group), 
               fill = "lightgray", color = "white") +
  
  # zones colorées selon le trafic
  geom_tile(data = zones_actives, 
            aes(x = lon, y = lat, fill = count), 
            alpha = 0.8, width = 1, height = 1) +
  
  # top 5 des zones les plus fréquentées
  geom_text(data = head(zones_actives[order(-zones_actives$count), ], 5), # sur le TOP 5, on affiche le nombre de bateaux
            aes(x = lon, y = lat, label = count),                         # c'est assez illisible mais une aide pour voir dans quelle zone il est
            color = "white", fontface = "bold", size = 4) +
  
  scale_fill_gradient(name = "Trafic", 
                      low = "yellow", high = "red",
                      trans = "sqrt") +
  
  coord_fixed(xlim = c(-100, -80), ylim = c(20, 35)) +
  
  labs(title = "Carte des zones de trafic maritime",
       subtitle = "Golfe du Mexique - Zones les plus fréquentées",
       x = "Longitude", y = "Latitude") +
  
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 8),
    panel.background = element_rect(fill = "lightblue")
  )

# print(p5)
ggsave("result/carte_zones_trafic.png", p5, width = 12, height = 8)





# truc en plus pas nécessaire

# # on fait le pdf, (pas obligé)

# pdf("result/fonctionnalite_2.pdf", width = 12, height = 8)
# print(p1)
# print(p2)
# print(p2_temp)
# print(p3)
# print(p4)
# print(p5)
# dev.off()





# # stats globales

# cat("observations :", nrow(df_golfe), "\n\n")

# cat("repartitions par type :\n") # en pourcentage
# print(type_counts)

# cat("\nvitesse moyenne par type :\n")
# vitesse_stats <- aggregate(SOG ~ Type_Simple, # affichage des moyenne  de vitesse des Cargo, Passenger, Tanker
#                data = df_golfe[!is.na(df_golfe$SOG) & df_golfe$Type_Simple != "Autre", ], 
#                FUN = function(x) round(mean(x, na.rm = TRUE), 1))
# colnames(vitesse_stats)[2] <- "vitesse_moyenne"
# print(vitesse_stats)




