library(glm2)
library(leaflet)
library(htmlwidgets)

output_dir <- "/tmp/leaflet/"

# Import des big data
df <- read.csv("vessel-total-clean.csv")

df[df == "\\N"] <- NA

df$SOG[df$SOG > 27] <- NA
df$COG[df$COG > 360] <- NA
df$VesselType[df$VesselType == 0] <- NA
df$Status[df$Status == 15] <- NA
df$Cargo[df$Cargo == 0] <- NA
df <- df[df$LON > -110, ]

df <- unique(df)

# Check what is NA
colSums(is.na(df))

df$SOG[is.na(df$SOG)] <- mean(df$SOG)

vessel_type_map <- data.frame(
  VesselType = c(60, 70, 80),
  VesselCategory = c("Passenger", "Cargo", "Tanker")
)

df <- merge(df, vessel_type_map, by = "VesselType", all.x = TRUE)
df$VesselCategory[is.na(df$VesselCategory)] <- "Autre bato"


boats <- unique(df$MMSI)


find_correct_rows <- function(mmsi) {
  rows <- df[df$MMSI == mmsi, ]

  rows$BaseDateTime <- as.POSIXct(
    rows$BaseDateTime,
    tz = "",
    format = "%Y-%m-%d %H:%M:%S"
  )

  rows <- rows[order(rows$BaseDateTime), ]

  rows
}


add_boat_line <- function(map, mmsi, opacity, weight, color) {
  rows <- find_correct_rows(mmsi)
  first <- head(rows, 1)
  label <- first$VesselName
  popup <- paste(
    "MMSI: ", first$MMSI, "<br>",
    "Name: ", first$VesselName, "<br>",
    "IMO: ", first$IMO, "<br>",
    "CallSign: ", first$CallSign, "<br>",
    "VesselType/Cargo: ", first$VesselType, "-", first$Cargo, "<br>",
    "Status: ", first$Status, "<br>",
    "LxWxD: ", first$Length, "x", first$Width, "x", first$Draft, "<br>", # nolint: line_length_linter.
    "Transceiver: ", first$TransceiverClass, "<br>",
    sep = ""
  )

  addPolylines(
    map = map,
    lng = rows$LON,
    lat = rows$LAT,
    weight = weight,
    opacity = opacity,
    color = color,
    label = label,
    popup = popup,
    group = as.character(first$VesselName)
  )
}

add_boat_markers <- function(map, mmsi) {
  rows <- find_correct_rows(mmsi)
  first <- head(rows, 1)
  last <- tail(rows, 1)

  map %>%
    addMarkers(lng = first$LON, lat = first$LAT, group = first$VesselName) %>%
    addMarkers(lng = last$LON, lat = last$LAT, group = last$VesselName)
}




# For most used roads
most_used_roads <- function() {

  map <- leaflet() %>%
    addTiles()

  for (boat in boats) {
    map <- map %>%
      add_boat_line(boat, opacity = 0.1, weight = 30, color = "blue")
  }

  saveWidget(
    widget = map,
    file = paste(output_dir, "most_used_roads.html", sep = "")
  )
  print("Exported most_used_roads.html")
}

# For single boat
single_boat <- function() {

  map <- leaflet() %>%
    addTiles()

  vessel_names <- unique(df$VesselName)
  vessel_names <- vessel_names[order(vessel_names)]

  for (boat in boats) {
    map <- map %>%
      add_boat_line(boat, opacity = 0.8, weight = 2, color = "red") %>%
      add_boat_markers(boat)
  }

  map <- map %>% addLayersControl(
    baseGroups = vessel_names,
    options = layersControlOptions(collapsed = FALSE)
  )

  saveWidget(
    widget = map,
    file = paste(output_dir, "single_boat.html", sep = "")
  )
  print("Exported single_boat.html")
}

# All boats
all_boats <- function() {

  map <- leaflet() %>%
    addTiles()

  vessel_names <- unique(df$VesselName)
  vessel_names <- vessel_names[order(vessel_names)]

  for (boat in boats) {
    map <- map %>%
      add_boat_line(boat, opacity = 0.8, weight = 2, color = "blue")
  }

  saveWidget(
    widget = map,
    file = paste(output_dir, "all_boats.html", sep = "")
  )
  print("Exported all_boats.html")
}



# most_used_roads()
# single_boat()
all_boats()
