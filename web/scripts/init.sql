USE etu0623;

CREATE TABLE cluster (
    id_cluster INT PRIMARY KEY AUTO_INCREMENT,
    cluster_name VARCHAR(255) NOT NULL,
    description TEXT
);

CREATE TABLE status_code (
    code_status INT PRIMARY KEY,
    description VARCHAR(255) NOT NULL
);

CREATE TABLE transceiver_class (
    code_transceiver INT PRIMARY KEY,
    class VARCHAR(255) NOT NULL
);

CREATE TABLE vessel (
    mmsi INT PRIMARY KEY,
    vessel_name VARCHAR(255),
    imo_number VARCHAR(20),
    callsign VARCHAR(20),
    length FLOAT,
    width FLOAT,
    code_transceiver INT,
    FOREIGN KEY (code_transceiver) REFERENCES transceiver_class(code_transceiver)
);

CREATE TABLE point_donnee (
    id_point INT PRIMARY KEY AUTO_INCREMENT,
    base_date_time DATETIME NOT NULL,
    mmsi INT NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    speed_over_ground FLOAT,
    cap_over_ground FLOAT,
    heading FLOAT,
    draft FLOAT,
    code_status INT,
    id_cluster INT,
    FOREIGN KEY (mmsi) REFERENCES vessel(mmsi),
    FOREIGN KEY (code_status) REFERENCES status_code(code_status),
    FOREIGN KEY (id_cluster) REFERENCES cluster(id_cluster)
);

CREATE INDEX idx_point_donnee_mmsi ON point_donnee(mmsi);
CREATE INDEX idx_point_donnee_datetime ON point_donnee(base_date_time);
CREATE INDEX idx_point_donnee_location ON point_donnee(latitude, longitude);

SET FOREIGN_KEY_CHECKS = 0;
TRUNCATE TABLE point_donnee;
TRUNCATE TABLE vessel;
TRUNCATE TABLE cluster;
TRUNCATE TABLE status_code;
TRUNCATE TABLE transceiver_class;
SET FOREIGN_KEY_CHECKS = 1;

INSERT INTO transceiver_class (code_transceiver, class) VALUES
(1, 'A'),
(2, 'B');

INSERT INTO status_code (code_status, description) VALUES
(0, 'Under way using engine'),
(1, 'At anchor'),
(2, 'Not under command'),
(3, 'Restricted manoeuvrability'),
(4, 'Constrained by her draught'),
(5, 'Moored'),
(6, 'Aground'),
(7, 'Engaged in fishing'),
(8, 'Under way sailing'),
(9, 'Reserved for future amendment (DG/HS/MP, HSC)'),
(10, 'Reserved for future amendment (DG/HS/MP, WIG)'),
(11, 'Reserved for future use'),
(12, 'Reserved for future use'),
(13, 'Reserved for future use'),
(14, 'Reserved for future use'),
(15, 'Not defined = default');

INSERT INTO cluster (cluster_name, description) VALUES
('Côtes de Floride', 'Est probablement dans le cluster des navires qui sont proches des cotes de Floride'),
('Ports Houston/Nouvelle-Orléans', 'Est probablement dans un port dans la zone Houston/Nouvelle-Orléans'),
('Transit Golfe Central', 'Est probablement en transit dans la zone centrale du Golfe du Mexique'),
('Corridor Commercial Côte Est', 'Est probablement dans un corridor commercial majeur de la côte Est'),
('Route Amérique Centrale', "Est probablement en déplacement commercial rapide vers l'Amérique Centrale"),
('Route Commerciale Ouest/Mexique', "Est probablement sur une route commerciale vers l'ouest/Mexique"),
('Arrêt Côtier Golfe du Mexique', "Est probablement à l'arrêt proche des côtes dans le golfe du Mexique");

