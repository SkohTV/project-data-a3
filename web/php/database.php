<?php

  require_once('constants.php');

  //----------------------------------------------------------------------------
  //--- dbConnect --------------------------------------------------------------
  //----------------------------------------------------------------------------
  // Create the connection to the database.
  // \return False on error and the database otherwise.
  function dbConnect()
  {
    try
    {
      $db = new PDO('mysql:host='.DB_SERVER.';port='.DB_PORT.';dbname='.DB_NAME, DB_USER, DB_PASSWORD);
      $db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    }
    catch (PDOException $exception)
    {
      error_log('Connection error: '.$exception->getMessage());
      return false;
    }
    return $db;
  }

  //----------------------------------------------------------------------------
  //--- dbRequestVessels --------------------------------------------------------
  //----------------------------------------------------------------------------
  // Request the list of vessels from the database.
  // \param db The connected database.
  function dbRequestVessels($db)
  {
    try
    {
      $request = 'SELECT * FROM vessel';
      $statement = $db->prepare($request);
      $statement->execute();
      $result = $statement->fetchAll(PDO::FETCH_ASSOC);
      // echo json_encode($result);
    }
    catch (PDOException $exception)
    {
      error_log('Request error: '.$exception->getMessage());
      return false;
    }
    return $result;
  }

  //----------------------------------------------------------------------------
  //--- dbRequestVessel --------------------------------------------------------
  //----------------------------------------------------------------------------
  // Request a vessel from the database.
  // \param db The connected database.
  // \param mmsi The MMSI of the vessel to request.
  function dbRequestVessel($db, $mmsi)
  {
    try
    {
      $request = 'SELECT * FROM vessel WHERE mmsi = :mmsi';
      $statement = $db->prepare($request);
      $statement->bindParam(':mmsi', $mmsi);
      $statement->execute();
      $result = $statement->fetch(PDO::FETCH_ASSOC);
    }
    catch (PDOException $exception)
    {
      error_log('Request error: '.$exception->getMessage());
      return false;
    }
    return $result;
  }

  //----------------------------------------------------------------------------
  //--- dbAddVessel ------------------------------------------------------------
  //----------------------------------------------------------------------------
  // Add a vessel to the database.
  // \param db The connected database.

  function dbAddVessel($db, $mmsi, $vesselname, $imo, $callsign, $transceiverclass, $length, $width) {
    try
    {
      // Get the transceiver class ID from the transceiver_class table.
      $transceiverclass = strtoupper($transceiverclass);
      $request = 'SELECT code_transceiver FROM transceiver_class WHERE class = :transceiverclass';
      $statement = $db->prepare($request);
      $statement->bindParam(':transceiverclass', $transceiverclass);
      $statement->execute();
      $transceiverclass = $statement->fetchColumn();
      if ($transceiverclass === false) {
        error_log('Transceiver class not found: '.$transceiverclass);
        return false;
      }
      // Check if the vessel already exists.
      $request = 'SELECT COUNT(*) FROM vessel WHERE mmsi = :mmsi';
      $statement = $db->prepare($request);
      $statement->bindParam(':mmsi', $mmsi);
      $statement->execute();
      $count = $statement->fetchColumn();
      if ($count > 0) {
        error_log('Vessel already exists with MMSI: '.$mmsi);
        return false;
      }
      // Insert the vessel into the vessel table.
      $request = 'INSERT INTO vessel (mmsi, vessel_name, imo_number, callsign, code_transceiver, length, width) VALUES (:mmsi, :vesselname, :imo, :callsign, :transceiverclass, :length, :width)';
      $statement = $db->prepare($request);
      $statement->bindParam(':mmsi', $mmsi);
      $statement->bindParam(':vesselname', $vesselname);
      $statement->bindParam(':imo', $imo);
      $statement->bindParam(':callsign', $callsign);
      $statement->bindParam(':transceiverclass', $transceiverclass);
      $statement->bindParam(':length', $length);
      $statement->bindParam(':width', $width);
      $statement->execute();
    }
    catch (PDOException $exception)
    {
      error_log('Insert error: '.$exception->getMessage());
      return false;
    }
    return true;
  }

  //----------------------------------------------------------------------------
  //--- dbRequestAllPoints_donnee --------------------------------------------------------
  //----------------------------------------------------------------------------
  // Request the list of point_donnee for a vessel from the database.
  // \param db The connected database.
  // \param mmsi The MMSI of the vessel to request.
  // \param longueur_max The maximum length of the vessel to request.
  // \param largeur_max The maximum width of the vessel to request.
  // \param longueur_min The minimum length of the vessel to request.
  // \param largeur_min The minimum width of the vessel to request.
  // \param temps_min The minimum time of the point_donnee to request. (YYYY-MM-DD HH:MM:SS format)
  // \param temps_max The maximum time of the point_donnee to request. (YYYY-MM-DD HH:MM:SS format)
  // \param transceiver_class The transceiver class of the vessel to request.
  // \param status_code The status code of the point_donnee to request.

  function dbRequestAllPoints_donnee($db, $mmsi, $longueur_max, $largeur_max, $longueur_min, $largeur_min, $temps_min, $temps_max, $transceiver_class, $status_code) {
    try
    {
      $request = 'SELECT * FROM point_donnee WHERE mmsi = :mmsi AND latitude BETWEEN :latitude_min AND :latitude_max AND longitude BETWEEN :longitude_min AND :longitude_max AND base_date_time BETWEEN :temps_min AND :temps_max';
      if ($transceiver_class !== null) {
        $request .= ' AND transceiver_class = :transceiver_class';
      }
      if ($status_code !== null) {
        $request .= ' AND status_code = :status_code';
      }
      $statement = $db->prepare($request);
      $statement->bindParam(':mmsi', $mmsi);
      $statement->bindParam(':latitude_min', $largeur_min);
      $statement->bindParam(':latitude_max', $largeur_max);
      $statement->bindParam(':longitude_min', $longueur_min);
      $statement->bindParam(':longitude_max', $longueur_max);
      $statement->bindParam(':temps_min', $temps_min);
      $statement->bindParam(':temps_max', $temps_max);
      if ($transceiver_class !== null) {
        $statement->bindParam(':transceiver_class', $transceiver_class);
      }
      if ($status_code !== null) {
        $statement->bindParam(':status_code', $status_code);
      }
      $statement->execute();
      return $statement->fetchAll(PDO::FETCH_ASSOC);
    }
    catch (PDOException $exception)
    {
      error_log('Request error: '.$exception->getMessage());
      return false;
    }
  }

  //----------------------------------------------------------------------------
  //--- dbAddPoint_donnee --------------------------------------------------------
  //----------------------------------------------------------------------------
  // Add a point_donnee to the database.
  // \param db The connected database.
  // \param base_date_time The base date time of the point_donnee. (YYYY-MM-DD HH:MM:SS format, PK point_donnee.base_date_time)
  // \param mmsi The MMSI of the vessel. (FK vessel.mmsi and PK point_donnee.mmsi)
  // \param latitude The latitude of the point_donnee.
  // \param longitude The longitude of the point_donnee.
  // \param sog The speed over ground of the point_donnee.
  // \param cog The course over ground of the point_donnee.
  // \param heading The heading of the point_donnee.
  // \param status_code The navigation status of the point_donnee. (FK point_donnee_status.code)
  // \param draft The draft of the point_donnee.

  function dbAddPoint_donnee($db, $base_date_time, $mmsi, $latitude, $longitude, $sog, $cog, $heading, $status_code, $draft) {
    try
    {
      $request = 'INSERT INTO point_donnee (base_date_time, mmsi, latitude, longitude, sog, cog, heading, status_code, draft) VALUES (:base_date_time, :mmsi, :latitude, :longitude, :sog, :cog, :heading, :status_code, :draft)';
      $statement = $db->prepare($request);
      $statement->bindParam(':base_date_time', $base_date_time);
      $statement->bindParam(':mmsi', $mmsi);
      $statement->bindParam(':latitude', $latitude);
      $statement->bindParam(':longitude', $longitude);
      $statement->bindParam(':sog', $sog);
      $statement->bindParam(':cog', $cog);
      $statement->bindParam(':heading', $heading);
      $statement->bindParam(':status_code', $status_code);
      $statement->bindParam(':draft', $draft);
      $statement->execute();
    }
    catch (PDOException $exception)
    {
      error_log('Insert error: '.$exception->getMessage());
      return false;
    }
    return true;
  }

  //----------------------------------------------------------------------------
  //--- dbRequestVesselNames --------------------------------------------------------
  //----------------------------------------------------------------------------
  // Request the list of vessel names from the database.
  // \param db The connected database.

  function dbRequestVesselNames($db)
  {
    try
    {
      $request = 'SELECT mmsi, vesselname FROM vessel ORDER BY vesselname';
      $statement = $db->prepare($request);
      $statement->execute();
      $result = $statement->fetchAll(PDO::FETCH_ASSOC);
    }
    catch (PDOException $exception)
    {
      error_log('Request error: '.$exception->getMessage());
      return false;
    }
    return $result;
  }

  //----------------------------------------------------------------------------
  //--- dbRequestAllTransceiverClass --------------------------------------------------------
  //----------------------------------------------------------------------------
  // Request the list of transceiver classes from the database.
  // \param db The connected database.
  function dbRequestAllTransceiverClass($db)
  {
    try
    {
      $request = 'SELECT * FROM transceiver_class ORDER BY class';
      $statement = $db->prepare($request);
      $statement->execute();
      $result = $statement->fetchAll(PDO::FETCH_ASSOC);
    }
    catch (PDOException $exception)
    {
      error_log('Request error: '.$exception->getMessage());
      return false;
    }
    return $result;
  }

  //----------------------------------------------------------------------------
  //--- dbRequestAllMMSI --------------------------------------------------------
  //----------------------------------------------------------------------------
  // Request the list of all MMSI from the database.
  // \param db The connected database.
  function dbRequestAllMMSI($db)
  {
    try
    {
      $request = 'SELECT mmsi FROM vessel ORDER BY mmsi';
      $statement = $db->prepare($request);
      $statement->execute();
      $result = $statement->fetchAll(PDO::FETCH_ASSOC);
    }
    catch (PDOException $exception)
    {
      error_log('Request error: '.$exception->getMessage());
      return false;
    }
    return $result;
  }

  //----------------------------------------------------------------------------
  //--- dbRequestAllStatusCode --------------------------------------------------------
  //----------------------------------------------------------------------------
  // Request the list of all status codes from the database.
  // \param db The connected database.
  function dbRequestAllStatusCode($db)
  {
    try
    {
      $request = 'SELECT code, description FROM status_code ORDER BY code';
      $statement = $db->prepare($request);
      $statement->execute();
      $result = $statement->fetchAll(PDO::FETCH_ASSOC);
    }
    catch (PDOException $exception)
    {
      error_log('Request error: '.$exception->getMessage());
      return false;
    }
    return $result;
  }

  //----------------------------------------------------------------------------
  //--- dbRequestTab --------------------------------------------------------
  //----------------------------------------------------------------------------
  // Request the list for the table in the visualization on website.
  // \param db The connected database.
  // \param mmsi The MMSI of the vessel to request.
  // \param limits The number of results to return.
  // \param page The page number to return.
  // \param longueur_max The maximum length of the vessel to request.
  // \param largeur_max The maximum width of the vessel to request.
  // \param longueur_min The minimum length of the vessel to request.
  // \param largeur_min The minimum width of the vessel to request.
  // \param temps_min The minimum time of the point_donnee to request. (YYYY-MM-DD HH:MM:SS format)
  // \param temps_max The maximum time of the point_donnee to request. (YYYY-MM-DD HH:MM:SS format)
  // \param transceiver_class The transceiver class of the vessel to request.
  // \param status_code The status code of the point_donnee to request.

function dbRequestTab($db, $limits, $page, $longueur_max, $longueur_min, $largeur_max, $largeur_min, $temps_max, $temps_min, $transceiver_class, $status_code, $mmsi) {
    try {
        $offset = ($page - 1) * $limits;
        
        $baseWhere = 'FROM point_donnee pd JOIN vessel v ON pd.mmsi = v.mmsi JOIN status_code sc ON sc.code_status = pd.code_status
                        WHERE v.length BETWEEN :longueur_min AND :longueur_max
                         AND v.width BETWEEN :largeur_min AND :largeur_max';
  
        // $baseWhere = 'FROM point_donnee pd 
        //              WHERE pd.mmsi IN (
        //                  SELECT v.mmsi FROM vessel v 
        //
        //                  WHERE v.length BETWEEN :longueur_min AND :longueur_max
        //                  AND v.width BETWEEN :largeur_min AND :largeur_max';

        $params = [
            ':longueur_min' => $longueur_min,
            ':longueur_max' => $longueur_max,
            ':largeur_min' => $largeur_min,
            ':largeur_max' => $largeur_max
        ];
        
        
        if ($transceiver_class !== null) {
            $baseWhere .= ' AND v.transceiverclass = :transceiver_class';
            $params[':transceiver_class'] = $transceiver_class;
        }
        
        // $baseWhere .= ')';
        
        
        if ($mmsi !== null) {
            $baseWhere .= ' AND pd.mmsi = :mmsi';
            $params[':mmsi'] = $mmsi;
        }
        
        if ($temps_min !== null && $temps_max !== null) {
            $baseWhere .= ' AND pd.base_date_time BETWEEN :temps_min AND :temps_max';
            $params[':temps_min'] = $temps_min;
            $params[':temps_max'] = $temps_max;
        } elseif ($temps_min !== null) {
            $baseWhere .= ' AND pd.base_date_time >= :temps_min';
            $params[':temps_min'] = $temps_min;
        } elseif ($temps_max !== null) {
            $baseWhere .= ' AND pd.base_date_time <= :temps_max';
            $params[':temps_max'] = $temps_max;
        }
        
        if ($status_code !== null) {
            $baseWhere .= ' AND sc.description = :status_code';
            $params[':status_code'] = $status_code;
        }
        
        
        $countQuery = 'SELECT COUNT(DISTINCT pd.id_point) as total ' . $baseWhere;
        $countStatement = $db->prepare($countQuery);
        foreach ($params as $key => $value) {
            $countStatement->bindValue($key, $value);
        }
        $countStatement->execute();
        $totalCount = $countStatement->fetch(PDO::FETCH_ASSOC)['total'];
        
        
        $dataQuery = 'SELECT pd.id_point, pd.base_date_time, pd.mmsi, pd.latitude, pd.longitude, 
                      pd.speed_over_ground as sog, pd.cap_over_ground as cog, pd.heading, 
                      pd.code_status as status_code, pd.draft, pd.id_cluster ' . 
                     $baseWhere . 
                     ' LIMIT :limits OFFSET :offset';
        
        $params[':limits'] = (int)$limits;
        $params[':offset'] = (int)$offset;
        
        $dataStatement = $db->prepare($dataQuery);
        
        
        foreach ($params as $key => $value) {
            if ($key === ':limits' || $key === ':offset') {
                $dataStatement->bindValue($key, $value, PDO::PARAM_INT);
            } else {
                $dataStatement->bindValue($key, $value);
            }
        }
        
        $dataStatement->execute();
        $data = $dataStatement->fetchAll(PDO::FETCH_ASSOC);
        
        
        $totalPages = ceil($totalCount / $limits);
        
        return [
            'data' => $data,
            'pagination' => [
                'current_page' => (int)$page,
                'total_pages' => $totalPages,
                'total_count' => (int)$totalCount,
                'per_page' => (int)$limits
            ]
        ];
        
    } catch (PDOException $exception) {
        error_log('Erreur de requête : ' . $exception->getMessage());
        return [
            'status' => 'error',
            'message' => 'Erreur lors de l\'exécution de la requête : ' . $exception->getMessage(),
            'data' => [],
            'pagination' => [
                'current_page' => 1,
                'total_pages' => 0,
                'total_count' => 0,
                'per_page' => $limits
            ]
        ];
    }
}


  //----------------------------------------------------------------------------
  //--- dbRequestFilterValues --------------------------------------------------------
  //----------------------------------------------------------------------------
  // Request the filter values for the table in the visualization on website.
  // -> GET php/get_filter_values : OUTPUT -> json{"longueur": ["<max>", "<min>"], "largeur": ["<max>", "<min>"], "temps":["<max>", "<min>"], "status_code": ["<description_code_1>", ....], "transceiver": ["A", "B"]}
  // \param db The connected database.
  function dbRequestFilterValues($db)
  {
    try
    {
      $result = array();

      // Request the maximum and minimum length.
      $request = 'SELECT MAX(length) AS max_length, MIN(length) AS min_length FROM vessel';
      $statement = $db->prepare($request);
      $statement->execute();
      $lengths = $statement->fetch(PDO::FETCH_ASSOC);
      $result['longueur'] = [$lengths['max_length'], $lengths['min_length']];

      // Request the maximum and minimum width.
      $request = 'SELECT MAX(width) AS max_width, MIN(width) AS min_width FROM vessel';
      $statement = $db->prepare($request);
      $statement->execute();
      $widths = $statement->fetch(PDO::FETCH_ASSOC);
      $result['largeur'] = [$widths['max_width'], $widths['min_width']];

      // Request the maximum and minimum time.
      $request = 'SELECT MAX(base_date_time) AS max_time, MIN(base_date_time) AS min_time FROM point_donnee';
      $statement = $db->prepare($request);
      $statement->execute();
      $times = $statement->fetch(PDO::FETCH_ASSOC);
      $result['temps'] = [$times['max_time'], $times['min_time']];

      // Request all status codes.
      $request = 'SELECT description FROM status_code ORDER BY code_status';
      $statement = $db->prepare($request);
      $statement->execute();
      $status_codes = $statement->fetchAll(PDO::FETCH_COLUMN);
      $result['status_code'] = array_values($status_codes);
    }
    catch (PDOException $exception)
    {
      error_log('Request error: '.$exception->getMessage());
      return false;
    }

    // Request all transceiver classes.
    $request = 'SELECT class FROM transceiver_class ORDER BY class';
    $statement = $db->prepare($request);
    $statement->execute();
    $transceivers = $statement->fetchAll(PDO::FETCH_COLUMN);
    $result['transceiver'] = array_values($transceivers);

    // Return the result.
    return $result;

  }

  //----------------------------------------------------------------------------
  //--- dbRequestAllClusters --------------------------------------------------------
  //----------------------------------------------------------------------------
  // Request the list of all clusters from the database.
  // \param db The connected database.
  function dbRequestAllClusters($db)
  {
    try
    {
      $request = 'SELECT * FROM cluster ORDER BY id';
      $statement = $db->prepare($request);
      $statement->execute();
      $result = $statement->fetchAll(PDO::FETCH_ASSOC);
    }
    catch (PDOException $exception)
    {
      error_log('Request error: '.$exception->getMessage());
      return false;
    }
    return $result;
  }

  //----------------------------------------------------------------------------
  //--- dbRequestAllPoints_donnee_Cluster --------------------------------------------------------
  //----------------------------------------------------------------------------
  // Request the list of point_donnee for a cluster from the database.
  // \param db The connected database.
  // \param cluster The ID of the cluster to request.
  function dbRequestAllPoints_donneeCluster($db, $cluster)
  {
    try
    {
      $request = 'SELECT * FROM point_donnee WHERE cluster_id = :cluster ORDER BY base_date_time';
      $statement = $db->prepare($request);
      $statement->bindParam(':cluster', $cluster, PDO::PARAM_INT);
      $statement->execute();
      $result = $statement->fetchAll(PDO::FETCH_ASSOC);
    }
    catch (PDOException $exception)
    {
      error_log('Request error: '.$exception->getMessage());
      return false;
    }
    return $result;
  }
