-- DDL for tables STATION, BIKE --

-- BIKE refers to STATION (bik_sta_ID - sta_ID)

---------------------
-- CREATE DATABASE --
---------------------

DROP DATABASE IF EXISTS BikeStations;
CREATE DATABASE IF NOT EXISTS BikeStations;
USE BikeStations;

-- Table structure for table 'STATION'

DROP TABLE IF EXISTS STATION;
CREATE TABLE IF NOT EXISTS STATION(
	sta_ID INT(10) NOT NULL AUTO_INCREMENT,
	sta_number INT(5) NOT NULL,
	sta_lat DECIMAL(20,15) NOT NULL,
	sta_lon DECIMAL(20,15) NOT NULL,
	sta_name VARCHAR(100) NOT NULL,
	sta_city VARCHAR(20) NOT NULL,
	sta_address VARCHAR(150) NOT NULL,
	sta_payment BOOLEAN,
	sta_bonus BOOLEAN,

	PRIMARY KEY (sta_ID)
	) ENGINE=InnoDB DEFAULT CHARSET=latin1
;

-- Table structure for table 'BIKE'

DROP TABLE IF EXISTS BIKE;
CREATE TABLE IF NOT EXISTS BIKE(
	bik_ID INT(10) NOT NULL AUTO_INCREMENT,
	bik_sta_ID INT(10),
	bik_timestamp DATETIME NOT NULL DEFAULT '0000-00-00 00:00:00',
	bik_status enum('CLOSED','OPEN') NOT NULL,
	bik_stands INT(4) NOT NULL,
	bik_available_stands INT(4) NOT NULL,
	bik_available INT(4) NOT NULL,

	PRIMARY KEY (bik_ID)
	) ENGINE=InnoDB DEFAULT CHARSET=latin1
;

-- Add FK constraint 

ALTER TABLE BIKE
	ADD CONSTRAINT BIKE_STATION_FK FOREIGN KEY (bik_sta_ID) REFERENCES STATION (sta_ID) ON DELETE SET NULL ON UPDATE CASCADE;


-- --------------------------------------------------------

----------------------
-- CREATE PROCEDURE --
----------------------

DELIMITER $$
DROP PROCEDURE IF EXISTS ADD_BIKE_STATION$$
CREATE PROCEDURE ADD_BIKE_STATION
	(IN v_name VARCHAR(100),
	IN v_lat DECIMAL(20,15),
	IN v_lon DECIMAL(20,15),
	IN v_address VARCHAR(150),
	IN v_available_bike_stands INT(4),
	IN v_available_bikes INT(4),
	IN v_banking BOOLEAN,
	IN v_bonus BOOLEAN,
	IN v_bike_stands INT(4),
	IN v_city VARCHAR(20),
	IN v_number INT(5),
	IN v_status enum('CLOSED','OPEN'),
	IN v_time DATETIME)
BEGIN
	DECLARE v_sta_ID INT;
	DECLARE v_bik_ID INT;
	SELECT sta_ID INTO v_sta_ID
	FROM STATION
	WHERE sta_city = v_city
	AND sta_number = v_number;
	IF v_sta_ID IS NULL THEN
	INSERT INTO STATION (sta_number, sta_lat, sta_lon, sta_name, sta_city, sta_address, sta_payment, sta_bonus)
	VALUES(v_number, v_lat, v_lon, v_name, v_city, v_address, v_banking, v_bonus);
	SELECT LAST_INSERT_ID() INTO v_sta_ID;
	END IF;
	SELECT bik_ID INTO v_bik_ID
	FROM BIKE
	WHERE bik_sta_ID = v_sta_ID
	AND bik_timestamp = v_time;
	IF v_bik_ID IS NULL THEN
	INSERT INTO BIKE (bik_sta_ID, bik_timestamp, bik_status, bik_stands, bik_available_stands, bik_available)
	VALUES(v_sta_ID, v_time, v_status, v_bike_stands, v_available_bike_stands, v_available_bikes);
	SELECT LAST_INSERT_ID() INTO v_bik_ID;
	END IF;
END$$
DELIMITER ;

-- --------------------------------------------------------

-------------------
-- ADD TEST DATA --
-------------------

INSERT INTO STATION VALUES(NULL,'3','10','10','station','ville','adresse',False,True);

INSERT INTO BIKE VALUES(NULL,'1','1984-09-10','CLOSED','5','2','3');
INSERT INTO BIKE (bik_sta_ID, bik_timestamp, bik_status, bik_stands, bik_available_stands, bik_available) VALUES
  ('1', '2018-01-25 20:49:46', 'OPEN', '15', '15', '3'),
  ('1', '2013-11-13 20:49:46', 'OPEN', '15', '15', '3'),
  ('1', '2016-05-21 20:49:46', 'CLOSED', '15', '15', '3');


