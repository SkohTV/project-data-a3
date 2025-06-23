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
  

  if($requestRessource == 'tweets'){
    if($requestMethod == 'GET'){
      
      $login = $_GET['login'];
      if($login == NULL){
      }else{
      }
    }

    if($requestMethod == 'POST'){
    }
  }

  // Send data to the client.
  header('Content-Type: application/json; charset=utf-8');
  header('Cache-control: no-store, no-cache, must-revalidate');
  header('Pragma: no-cache');
  header('HTTP/1.1 200 OK');
  exit;

?>