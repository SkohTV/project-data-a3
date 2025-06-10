# Importing big data
df <- read.csv("result/export_IA.csv")
boats <- unique(df$MMSI)

# Find the rows matching a boat MMSI
find_correct_rows <- function(mmsi) {
  rows <- df[df$MMSI == mmsi, ]

  # Used to convert string to timestamp...
  rows$BaseDateTime <- as.POSIXct(
    rows$BaseDateTime,
    tz = "",
    format = "%Y-%m-%d %H:%M:%S"
  )

  # ... that we can order
  rows <- rows[order(rows$BaseDateTime), ]

  rows
}


map_color <- function(vessel_code) {
  case_when(
    vessel_code >= 60 & vessel_code <= 69 ~ "#0000FF",
    vessel_code >= 70 & vessel_code <= 79 ~ "#00FF00",
    vessel_code >= 80 & vessel_code <= 89 ~ "#FF0000",
    .default = "#000000"
  )
}

map_VesselType <- function(vessel_code) {
  case_when(
    vessel_code >= 60 & vessel_code <= 69 ~ "Passenger",
    vessel_code >= 70 & vessel_code <= 79 ~ "Cargo",
    vessel_code >= 80 & vessel_code <= 89 ~ "Tanker",
    .default = "Autre"
  )
}

map_status <- function(code) {
  case_when(
    code == 0 ~ "Under way using engine",
    code == 1 ~ "At anchor",
    code == 2 ~ "Not under command",
    code == 3 ~ "Restricted manoeuverability",
    code == 4 ~ "Constrained by her draught",
    code == 5 ~ "Moored",
    code == 6 ~ "Aground",
    code == 7 ~ "Engaged in Fishing",
    code == 8 ~ "Under way sailing",
    code == 9 ~ "Reserved for future amendment of Navigational Status for HSC",
    code == 10 ~ "Reserved for future amendment of Navigational Status for WIG",
    code == 11 ~ "Reserved for future use",
    code == 12 ~ "Reserved for future use",
    code == 13 ~ "Reserved for future use",
    code == 14 ~ "AIS-SART is active",
    code == 15 ~ "Not defined (default)",
  )
}

make_popup <- function(first) {
  paste(
    "MMSI: ", first$MMSI, "<br>",
    "Name: ", first$VesselName, "<br>",
    "VesselType/Cargo: ", map_VesselType(first$VesselType), "<br>",
    "Status: ", map_status(first$Status), "<br>",
    "LxW: ", first$Length, "x", first$Width, "<br>",
    "Transceiver: ", first$TransceiverClass, "<br>",
    sep = ""
  )
}

# Create a polyline on the leaf_map for a given boat MMSI
add_boat_line <- function(leaf_map, mmsi, opacity, weight, grp_is_name = TRUE) {

  # Build values
  rows <- find_correct_rows(mmsi)
  first <- head(rows, 1)
  label <- first$VesselName
  popup <- make_popup(first)
  group <- if(grp_is_name == TRUE) as.character(first$VesselName) else map_VesselType(first$VesselType)

  # Create and return the polyline object
  addPolylines(
    map = leaf_map,
    lng = rows$LON,
    lat = rows$LAT,
    weight = weight,
    opacity = opacity,
    color = map_color(first$VesselType),
    label = label,
    popup = popup,
    group = group
  )
}

# Add markers at the start and end of the boats' journey
add_boat_markers <- function(leaf_map, mmsi) {
  rows <- find_correct_rows(mmsi)
  first <- head(rows, 1)
  last <- tail(rows, 1)
  
  label <- first$VesselName
  popup <- make_popup(first)

  l <- unique(rows$Draft)
  if (length(l) > 1 && all(round(l, 2) == l)) { # The all is to avoid "guessed" drafts
    l <- l[-1]
    changes <- rows[match(l, rows$Draft), ]
    for (i in 1:nrow(changes)) {
      item <- changes[i, ]
      leaf_map <- leaf_map %>% addMarkers(lng = item$LON, lat = item$LAT, group = first$VesselName)
    }
  }


  leaf_map %>%
    addMarkers(lng = first$LON, lat = first$LAT, group = first$VesselName, label = label, popup = popup) %>%
    addMarkers(lng = last$LON, lat = last$LAT, group = first$VesselName, label = label, popup = popup)
}


# For single boat
single_boat <- function() {

  leaf_map <- leaflet() %>%
    addTiles()

  vessel_names <- unique(df$VesselName)
  vessel_names <- vessel_names[order(vessel_names)]

  # Add a polyline and markers for each boat
  for (boat in boats) {
    leaf_map <- leaf_map %>%
      add_boat_line(boat, opacity = 0.8, weight = 2) %>%
      add_boat_markers(boat)
  }

  # Add toggle to show which journey
  leaf_map <- leaf_map %>% addLayersControl(
    baseGroups = vessel_names,
    options = layersControlOptions(collapsed = FALSE)
  )

  # Export the widget
  saveWidget(
    widget = leaf_map,
    file = "result/leaflet/single_boat.html"
  )
  print("Exported single_boat.html")
}

# All boats
all_boats <- function() {

  leaf_map <- leaflet() %>%
    addTiles()

  # Add a polyline for each boat (small and precise)
  for (boat in boats) {
    leaf_map <- leaf_map %>%
      add_boat_line(boat, opacity = 0.8, weight = 2, grp_is_name = FALSE)
  }

  leaf_map <- leaf_map %>%
    addHeatmap(lat = df$LAT, lng = df$LON, group = "Ports") %>%
    addHeatmap(lat = df[df$SOG > 15, ]$LAT, lng = df[df$SOG > 15, ]$LON, group = "Routes") %>%
    addLayersControl(
      baseGroups = c("Ports", "Routes"),
      overlayGroups = unique(sapply(unique(df$VesselType), map_VesselType)),
      options = layersControlOptions(collapsed = FALSE)
    )

  # Export the widget
  saveWidget(
    widget = leaf_map,
    file = "result/leaflet/all_boats.html"
  )
  print("Exported all_boats.html")
}


# Boat draft clusters
boat_draft_clusters <- function() {

  leaf_map <- leaflet() %>%
    addTiles()

  # Checks which boats have draft delta
  all_coords <- data.frame()
  for (boat in boats) {
    rows <- find_correct_rows(boat)
    l <- unique(rows$Draft)
    if (length(l) > 1 && all(round(l, 1) == l)) { # The all is to avoid "guessed" drafts
      changes <- rows[match(l, rows$Draft), ]
      for (i in 2:nrow(changes)) {
        ad <- changes[i, ]
        ad$oldDraft <- changes[i-1, ]$Draft
        if (abs(ad$oldDraft - ad$Draft) > 0.5) {
          all_coords <- rbind(all_coords, ad)
        }
      }
    }
  }

  # The cool CAH
  nb_clusters = 9
  distances <- dist(all_coords[c("LAT", "LON")])
  ward <- hclust(distances, method="ward.D2")
  groupes <- cutree(ward, k=nb_clusters)
  all_coords$grp <- groupes


  # Add a marker on each delta draft
  for (i in 1:nrow(all_coords)){
    item <- all_coords[i, ]

    # Build values
    label <- item$VesselName
    popup <- paste(
      make_popup(item),
      "Draft: ", item$oldDraft, " -> ", item$Draft, "<br>",
      sep = ""
    )

    leaf_map <- leaf_map %>% addMarkers(
      lng = item$LON,
      lat = item$LAT,
      label = label,
      popup = popup,
      group = paste("Cluster", item$grp)
    )
  }

# Compute all delta in clusters

ratios <- sapply(
    1:nb_clusters,
    function(x){
        mean(all_coords[all_coords$grp == x, ]$Draft) - mean(all_coords[all_coords$grp == x, ]$oldDraft)
    }
)


# Convert them to colors
  mx <- max(ratios)
  mn <- min(ratios)
  ratios <- sapply(ratios, FUN = function(x) { (x - mn) / (mx - mn)} )
  colors <- t(sapply(ratios, FUN = function(x) {colorRamp(c("green", "red"))(x)} ))
  colors <- apply(colors, 1, FUN = function(x) {rgb(x[1], x[2], x[3], maxColorValue=255)} )

  for (i in 1:nb_clusters) {
    mx_lat <- max(all_coords[all_coords$grp == i, ]$LAT)
    mn_lat <- min(all_coords[all_coords$grp == i, ]$LAT)
    mx_lon <- max(all_coords[all_coords$grp == i, ]$LON)
    mn_lon <- min(all_coords[all_coords$grp == i, ]$LON)

    leaf_map <- leaf_map %>% addRectangles(
      mx_lon + 0.1,
      mx_lat + 0.1,
      mn_lon - 0.1,
      mn_lat - 0.1,
      group = paste("Cluster", i),
      color = colors[i],
      stroke = FALSE
    )
  }

  # Add toggle to show which cluster
  leaf_map <- leaf_map %>% addLayersControl(
    overlayGroups = sapply(1:nb_clusters, function(x){paste("Cluster", x)} ),
    options = layersControlOptions(collapsed = FALSE)
  )

  saveWidget(
    widget = leaf_map,
    file = "result/leaflet/boat_draft_clusters.html"
  )
  print("Exported boat_draft_clusters.html")
}


all_boats()
single_boat()
boat_draft_clusters()
