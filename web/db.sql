USE etu0623;

CREATE TABLE vessel (
    mmsi INTEGER PRIMARY KEY,
    vessel_name VARCHAR(255),
    imo VARCHAR(50),
    callsign VARCHAR(50),
    transceiver_class VARCHAR(10),
    length FLOAT,
    width FLOAT
);

CREATE TABLE status_code (
    code INTEGER PRIMARY KEY,
    description VARCHAR(255)
);

CREATE TABLE point_donnee (
    base_date_time TIMESTAMP,
    mmsi INTEGER,
    latitude FLOAT,
    longitude FLOAT,
    speed_over_ground FLOAT,
    cap_over_ground FLOAT,
    heading FLOAT,
    draft FLOAT,
    status_code INTEGER,
    PRIMARY KEY (base_date_time, mmsi),
    FOREIGN KEY (mmsi) REFERENCES vessel(mmsi),
    FOREIGN KEY (status_code) REFERENCES status_code(code)
);
