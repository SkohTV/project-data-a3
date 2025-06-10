library(dplyr)
library(lubridate)
library(readr)

df <- read.csv("../result/export_IA.csv")
df$BaseDateTime <- as.POSIXct(df$BaseDateTime, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")

get_mode <- function(x) {
  # OBTENTION DE LA VALEUR MODALE POUR LA VITESSE DE CROISIÈRE D'UN SEGMENT
  tmp <- na.omit(unique(x))
  tmp[which.max(tabulate(match(x, tmp)))]
}

df <- df %>%
  select(MMSI, BaseDateTime, LAT, LON, SOG)

df <- df %>%
  arrange(MMSI, BaseDateTime) %>%
  mutate(is_stopped = SOG <= 0.5)

# SEGMENTATION ENTRE SEGMENTS ARRÊTS ET SEGMENTS DÉPLACEMENTS
df <- df %>%
  group_by(MMSI) %>%
  mutate(
    state_change = is_stopped != lag(is_stopped, default = first(is_stopped)),
    segment_id = cumsum(state_change)
  ) %>%
  ungroup()

# FRAMING DU RÉSUMÉ DES INFORMATIONS VOULUS DE LA SEGMENTATION
segments <- df %>%
  group_by(MMSI, segment_id, is_stopped) %>%
  summarise(
    start_time = min(BaseDateTime),
    end_time = max(BaseDateTime),
    duration_min = as.numeric(difftime(end_time, start_time, units = "mins")),
    cruise_speed = get_mode(round(SOG,1)), # ARRONDI CAR DES DELTAS DE 0.1 KNOTS VONT BIAISÉ
    start_lat = first(LAT),
    start_lon = first(LON),
    end_lat = last(LAT),
    end_lon = last(LON),
    .groups = "drop"
  ) %>%
  group_by(MMSI) %>%
  mutate(IdSegment = row_number()) %>%
  ungroup()

write_csv(segments, "segments_vessels.csv")
