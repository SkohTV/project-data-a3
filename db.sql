CREATE TABLE vessel (
    mmsi INT PRIMARY KEY,
    vessel_name VARCHAR(100),
    imo VARCHAR(50),
    callsign VARCHAR(50),
    transceiver_class VARCHAR(50),
    length FLOAT,
    width FLOAT
);

CREATE TABLE point_donnee (
    base_date_time TIMESTAMP,
    mmsi INT,
    latitude FLOAT,
    longitude FLOAT,
    speed_over_ground FLOAT,
    cap_over_ground FLOAT,
    heading FLOAT,
    status INT,
    draft FLOAT,
    PRIMARY KEY (base_date_time, mmsi),
    FOREIGN KEY (mmsi) REFERENCES vessel(mmsi)
);
