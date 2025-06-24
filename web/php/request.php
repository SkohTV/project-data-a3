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

  // var_dump($requestMethod);
  // var_dump($request);
  // var_dump($requestRessource);
  // var_dump($request);

  // Now, request has the form:
  //   [0] = 'vessels'
  //   [1] = 'mmsi' (optional)
  
  echo "test";

  if($requestRessource == 'vessels'){
    // Request the list of vessels.
    if ($requestMethod == 'GET')
    {
      $result = dbRequestVessels($db);
      if (!$result)
      {
        header('HTTP/1.1 500 Internal Server Error');
        exit;
      }
      echo json_encode($result);
    }
    else
    {
      header('HTTP/1.1 405 Method Not Allowed');
      exit;
    }
  }

  // Send data to the client.
  header('Content-Type: application/json; charset=utf-8');
  header('Cache-control: no-store, no-cache, must-revalidate');
  header('Pragma: no-cache');
  header('HTTP/1.1 200 OK');
  exit;

?>