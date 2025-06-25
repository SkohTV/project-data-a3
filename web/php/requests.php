<?php

  require_once('database.php');

  // Database connection.
  $db = dbConnect();
  if (!$db)
  {
    header('HTTP/1.1 503 Service Unavailable');
    exit;
  }

  $requestMethod = $_SERVER['REQUEST_METHOD'];
  $request = substr($_SERVER['PATH_INFO'], 1);
  $request = explode('/', $request);
  $requestRessource = array_shift($request);

  // for a route like -> POST php/add_boat?mmsi=XXX&vesselname=XXX&imo=XXX&callsign=XXX&transceiverclass=XXX&length=XXX&width=XXX : OUTPUT -> None

  if ($requestRessource == 'add_boat') {
    if ($requestMethod == 'POST') {
      $mmsi = $_POST['mmsi'] ?? null;
      $vesselname = $_POST['vesselname'] ?? null;
      $imo = $_POST['imo'] ?? null;
      $callsign = $_POST['callsign'] ?? null;
      $transceiverclass = $_POST['transceiverclass'] ?? null;
      $length = $_POST['length'] ?? null;
      $width = $_POST['width'] ?? null;

      if (!$mmsi || !$vesselname || !$imo || !$callsign || !$transceiverclass || !$length || !$width) {
        header('HTTP/1.1 400 Bad Request');
        exit;
      }

      $result = dbAddVessel($db, $mmsi, $vesselname, $imo, $callsign, $transceiverclass, $length, $width);
      if ($result === false) {
        header('HTTP/1.1 500 Internal Server Error');
        exit;
      }

      header('HTTP/1.1 201 Created');
      exit;
    } else {
      header('HTTP/1.1 405 Method Not Allowed');
      exit;
    }
  }

  // for a route like -> -> POST php/add_point_donnee?mmsi=XXX&base_date_time=XXX&latitude=XXX&longitude=XXX&sog=XXX&cog=XXX&heading=XXX&status_code=XXX&draft=XXX : OUTPUT -> None

  if ($requestRessource == 'add_point_donnee') {
    if ($requestMethod == 'POST') {
      if (!isset($_POST['mmsi']) || !isset($_POST['base_date_time']) || !isset($_POST['latitude']) || !isset($_POST['longitude']) || !isset($_POST['sog']) || !isset($_POST['cog']) || !isset($_POST['heading']) || !isset($_POST['status_code']) || !isset($_POST['draft'])) {
        header('HTTP/1.1 400 Bad Request');
        exit;
      }

      $result = dbAddPoint_donnee($db, $base_date_time, $mmsi, $latitude, $longitude, $sog, $cog, $heading, $status_code, $draft);
      if ($result === false) {
        header('HTTP/1.1 500 Internal Server Error');
        exit;
      }

      header('HTTP/1.1 201 Created');
      exit;
    } else {
      header('HTTP/1.1 405 Method Not Allowed');
      exit;
    }
  }
  

  // -> GET php/all_vessel_names : OUTPUT -> LIST all_vessels_names

  if ($requestRessource == 'all_vessel_names') {
    if ($requestMethod == 'GET') {
      $result = dbRequestVesselNames($db);
      if ($result === false) {
        header('HTTP/1.1 500 Internal Server Error');
        exit;
      }

      // Send the result as JSON.
      echo json_encode($result);
      exit;
    } else {
      header('HTTP/1.1 405 Method Not Allowed');
      exit;
    }
  }
  // -> GET php/all_transceiver_class

  if ($requestRessource == 'all_transceiver_class') {
    if ($requestMethod == 'GET') {
      $result = dbRequestAllTransceiverClass($db);
      if ($result === false) {
        header('HTTP/1.1 500 Internal Server Error');
        exit;
      }

      // Send the result as JSON.
      echo json_encode($result);
      exit;
    } else {
      header('HTTP/1.1 405 Method Not Allowed');
      exit;
    }
  }

  // -> GET php/all_mmsi

  if ($requestRessource == 'all_mmsi') {
    if ($requestMethod == 'GET') {
      $result = dbRequestAllMMSI($db);
      if ($result === false) {
        header('HTTP/1.1 500 Internal Server Error');
        exit;
      }

      // Send the result as JSON.
      echo json_encode($result);
      exit;
    } else {
      header('HTTP/1.1 405 Method Not Allowed');
      exit;
    }
  }

  // -> GET php/all_status_code

  if ($requestRessource == 'all_status_code') {
    if ($requestMethod == 'GET') {
      $result = dbRequestAllStatusCode($db);
      if ($result === false) {
        header('HTTP/1.1 500 Internal Server Error');
        exit;
      }

      // Send the result as JSON.
      echo json_encode($result);
      exit;
    } else {
      header('HTTP/1.1 405 Method Not Allowed');
      exit;
    }
  }

 // -> GET php/all_points_donnee?longueur_max=XXX&longueur_min=XXX&largeur_max=XXX&largeur_min=XXX&temps_max=XXX&temps_min=XXX&transceiver_class=XXX&status_code=XXX&mmsi=XXX : OUTPUT -> LIST points_donnee_in_filter
  
  if ($requestRessource == 'all_points_donnee') {
    if ($requestMethod == 'GET') {
      $longueur_max = $_GET['longueur_max'] ?? null;
      $longueur_min = $_GET['longueur_min'] ?? null;
      $largeur_max = $_GET['largeur_max'] ?? null;
      $largeur_min = $_GET['largeur_min'] ?? null;
      $temps_max = $_GET['temps_max'] ?? null;
      $temps_min = $_GET['temps_min'] ?? null;
      $transceiver_class = $_GET['transceiver_class'] ?? null;
      $status_code = $_GET['status_code'] ?? null;
      $mmsi = $_GET['mmsi'] ?? null;

      // Validate parameters.
      if (!is_numeric($longueur_max) || !is_numeric($longueur_min) || !is_numeric($largeur_max) || !is_numeric($largeur_min) ||
          !is_numeric($temps_max) || !is_numeric($temps_min) || !is_numeric($transceiver_class) || !is_numeric($status_code) || !is_numeric($mmsi)) {
        header('HTTP/1.1 400 Bad Request');
        exit;
      } 
      $result = dbRequestAllPoints_donnee($db, $longueur_max, $longueur_min, $largeur_max, $largeur_min, $temps_max, $temps_min, $transceiver_class, $status_code, $mmsi);
      if ($result === false) {
        header('HTTP/1.1 500 Internal Server Error');
        exit;
      }
      // Send the result as JSON.
      echo json_encode($result);
      exit;
    } else {
      header('HTTP/1.1 405 Method Not Allowed');
      exit;
    }
  }

  // -> GET php/get_tab?limits=XXX&page=XXX&longueur_max=XXX&longueur_min=XXX&largeur_max=XXX&largeur_min=XXX&temps_max=XXX&temps_min=XXX&transceiver_class=XXX&status_code=XXX&mmsi=XXX

  if ($requestRessource == 'get_tab') {
    if ($requestMethod == 'GET') {
      $limits = $_GET['limits'] ?? null;
      $page = $_GET['page'] ?? null;
      $longueur_max = $_GET['longueur_max'] ?? null;
      $longueur_min = $_GET['longueur_min'] ?? null;
      $largeur_max = $_GET['largeur_max'] ?? null;
      $largeur_min = $_GET['largeur_min'] ?? null;
      $temps_max = $_GET['temps_max'] ?? null;
      $temps_min = $_GET['temps_min'] ?? null;
      $transceiver_class = $_GET['transceiver_class'] ?? null;
      $status_code = $_GET['status_code'] ?? null;
      $mmsi = $_GET['mmsi'] ?? null;


      // Validate parameters.
      if (!is_numeric($limits) || !is_numeric($page) || !is_numeric($longueur_max) || !is_numeric($longueur_min) || !is_numeric($largeur_max) || !is_numeric($largeur_min)) {
        header('HTTP/1.1 400 Bad Request');
        exit;
      }



      if ($temps_max !== null && $temps_min !== null) {
        if (is_numeric($temps_max) && is_numeric($temps_min)) {
          $temps_max = date('Y-m-d H:i:s', (int)$temps_max);
          $temps_min = date('Y-m-d H:i:s', (int)$temps_min);
        } else {
          header('HTTP/1.1 400 Bad Request');
          exit;
        }
      }


      $result = dbRequestTab($db, $limits, $page, $longueur_max, $longueur_min, $largeur_max, $largeur_min, $temps_max, $temps_min, $transceiver_class, $status_code, $mmsi);
      if ($result === false) {
        header('HTTP/1.1 500 Internal Server Error');
        exit;
      } 

      // Send the result as JSON.
      echo json_encode($result);
      exit;
    } else {
      header('HTTP/1.1 405 Method Not Allowed');
      exit;
    }
  }
    
  // -> GET php/get_filter_values : OUTPUT -> json{"longueur": ["<max>", "<min>"], "largeur": ["<max>", "<min>"], "temps":["<max>", "<min>"], "status_code": ["<description_code_1>", ....], "transceiver": ["A", "B"]}

  if ($requestRessource == 'get_filter_values') {
    if ($requestMethod == 'GET') {
      $result = dbRequestFilterValues($db);
      if ($result === false) {
        header('HTTP/1.1 500 Internal Server Error');
        exit;
      }

      // Send the result as JSON.
      echo json_encode($result);
      exit;
    } else {
      header('HTTP/1.1 405 Method Not Allowed');
      exit;
    }
  }

  // -> GET php/all_clusters

  if ($requestRessource == 'all_clusters') {
    if ($requestMethod == 'GET') {
      $result = dbRequestAllClusters($db);
      if ($result === false) {
        header('HTTP/1.1 500 Internal Server Error');
        exit;
      }

      // Send the result as JSON.
      echo json_encode($result);
      exit;
    } else {
      header('HTTP/1.1 405 Method Not Allowed');
      exit;
    }
  }

  // -> GET php/all_points_donnee_cluster?cluster=XXX
  if ($requestRessource == 'all_points_donnee_cluster') {
    if ($requestMethod == 'GET') {
      $cluster = $_GET['cluster'] ?? null;

      // Validate parameters.
      if (!is_numeric($cluster)) {
        header('HTTP/1.1 400 Bad Request');
        exit;
      }

      $result = dbRequestAllPoints_donneeCluster($db, $cluster);
      if ($result === false) {
        header('HTTP/1.1 500 Internal Server Error');
        exit;
      }

      // Send the result as JSON.
      echo json_encode($result);
      exit;
    } else {
      header('HTTP/1.1 405 Method Not Allowed');
      exit;
    }
  }



// -> GET php/requests.php/predict_boat_cluster?latitude=XXX&longitude=XXX&sog=XXX&cog=XXX&heading=XXX : OUTPUT -> json{"<result_1>", "<result_2>", ...}


// http://etu0623.projets.isen-ouest.info/php/requests.php/predict_boat_cluster?latitude=48.123456&longitude=-4.123456&sog=10.5&cog=180&heading=90

if ($requestRessource == 'predict_boat_cluster') {
  if ($requestMethod == 'GET') {
    $latitude = $_GET['latitude'] ?? null;
    $longitude = $_GET['longitude'] ?? null;
    $sog = $_GET['sog'] ?? null;
    $cog = $_GET['cog'] ?? null;
    $heading = $_GET['heading'] ?? null;

    
    if (!is_numeric($latitude) || !is_numeric($longitude) || !is_numeric($sog) || !is_numeric($cog) || !is_numeric($heading)) {
      header('HTTP/1.1 400 Bad Request');
      exit;
    }

    $result = [];
    $return_var = 0;
    $cmd = "cd scripts && python3 besoin_client_1.py --lat " . escapeshellarg($latitude) . " --lon " . escapeshellarg($longitude) . " --sog " . escapeshellarg($sog) . " --cog " . escapeshellarg($cog) . " --heading " . escapeshellarg($heading) . " 2>&1";
    exec($cmd, $result, $return_var);

    if ($return_var !== 0) {
      error_log("Python script error: " . implode("\n", $result));
      header('HTTP/1.1 500 Internal Server Error');
      echo json_encode(['error' => 'Internal server error', 'details' => $result]);
      exit;
    }

    echo json_encode($result);
    exit;
  } else {
    header('HTTP/1.1 405 Method Not Allowed');
    exit;
  }
}


// -> GET php/requests.php/predict_boat_trajectory?latitude=XXX&longitude=XXX&sog=XXX&cog=XXX&heading=XXX&vesseltype=XXX&steps=XXX : OUTPUT -> [[LAT1, LON1], [LAT2, LON2], ...]
// http://etu0623.projets.isen-ouest.info/php/requests.php/predict_boat_trajectory?latitude=24.522600&longitude=-83.732800&sog=12.2&cog=113.1&heading=115&vesseltype=80&steps=15
if ($requestRessource == 'predict_boat_trajectory') {
  if ($requestMethod == 'GET') {
    $latitude = $_GET['latitude'] ?? null;
    $longitude = $_GET['longitude'] ?? null;
    $sog = $_GET['sog'] ?? null;
    $cog = $_GET['cog'] ?? null;
    $heading = $_GET['heading'] ?? null;
    $vesseltype = $_GET['vesseltype'] ?? null;
    $steps = $_GET['steps'] ?? null;

    
    if (!is_numeric($latitude) || !is_numeric($longitude) || !is_numeric($sog) || !is_numeric($cog) || !is_numeric($heading) || !is_numeric($vesseltype) || !is_numeric($steps)) {
      header('HTTP/1.1 400 Bad Request');
      exit;
    }

    $result = [];
    $return_var = 0;
    $json = [
        "LAT" => floatval($latitude),
        "LON" => floatval($longitude),
        "SOG" => floatval($sog),
        "COG" => floatval($cog),
        "Heading" => floatval($heading),
        "VesselType" => intval($vesseltype),
    ];
    $cmd = "cd scripts && python3.10 -O besoin_client_3.py --json '" . json_encode($json) . "' --steps " . escapeshellarg($steps) . " 2>&1";
    exec($cmd, $result, $return_var);

    if ($return_var !== 0) {
      error_log("Python script error: " . implode("\n", $result));
      header('HTTP/1.1 500 Internal Server Error');
      echo json_encode(['error' => 'Internal server error', 'details' => $result]);
      exit;
    }

    echo json_encode($result);
    exit;
  } else {
    header('HTTP/1.1 405 Method Not Allowed');
    exit;
  }
}

?>
