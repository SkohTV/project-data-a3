library(dplyr)
library(readr)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)

segments <- read_csv("segments_vessels.csv")
positions <- read_csv("../result/export_IA.csv")

segments$start_time <- as.POSIXct(segments$start_time)
segments$end_time <- as.POSIXct(segments$end_time)
positions$BaseDateTime <- as.POSIXct(positions$BaseDateTime)

onshore <- ne_download(scale = 50, type = "land", category = "physical", returnclass = "sf")
gulf_bbox <- st_bbox(c(xmin = -98, xmax = -80, ymin = 18, ymax = 31), crs = st_crs(onshore))
onshore_gulf <- st_crop(onshore, gulf_bbox)
onshore_projection <- st_transform(onshore_gulf, crs = 3857)

calculate_segment_distance <- function(mmsi, start_t, end_t) {
  seg_pos <- positions %>% 
    filter(MMSI == mmsi, BaseDateTime >= start_t, BaseDateTime <= end_t)
  
  if(nrow(seg_pos) == 0) {
    return(list(mean_distance = NA, n_positions = 0))
  }
  
  pos_sf <- st_as_sf(seg_pos, coords = c("LON", "LAT"), crs = 4326)
  pos_proj <- st_transform(pos_sf, crs = 3857)
  distances <- st_distance(pos_proj, onshore_projection) %>% apply(1, min)
  
  return(list(mean_distance = mean(distances, na.rm = TRUE), n_positions = nrow(seg_pos)))
}

segments_with_distance <- segments %>%
  rowwise() %>%
  mutate(
    result = list(calculate_segment_distance(MMSI, start_time, end_time)),
    mean_distance_to_coast = round(result$mean_distance,1),
    n_positions = result$n_positions
  ) %>%
  select(-result) %>%
  ungroup()

write_csv(segments_with_distance, "segments_with_coast_distance.csv")
