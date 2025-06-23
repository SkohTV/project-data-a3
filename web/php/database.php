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
      $db = new PDO('pgsql:host='.DB_SERVER.';port='.DB_PORT.';dbname='.DB_NAME, DB_USER, DB_PASSWORD);
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
  //--- dbRequestVessel --------------------------------------------------------
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
  

