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
      $request = 'INSERT INTO vessel (mmsi, vesselname, imo, callsign, transceiverclass, length, width) VALUES (:mmsi, :vesselname, :imo, :callsign, :transceiverclass, :length, :width)';
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
  //--- dbRequestPoint_donnee --------------------------------------------------------
  //----------------------------------------------------------------------------
  // Request the list of point_donnee for a vessel from the database.
  // \param db The connected database.
  // \param mmsi The MMSI of the vessel to request.

  function dbRequestPoint_donnee($db, $mmsi)
  {
    try
    {
      $request = 'SELECT * FROM point_donnee WHERE mmsi = :mmsi ORDER BY base_date_time DESC';
      $statement = $db->prepare($request);
      $statement->bindParam(':mmsi', $mmsi);
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
  

