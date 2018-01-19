-- DDL for tables STATION, BIKE --


-- BIKE refers to STATION (bik_sta_ID - sta_ID)


-- --------------------------------------------------------


DROP DATABASE IF EXISTS BikeStations;
CREATE DATABASE IF NOT EXISTS BikeStations;
USE BikeStations;


-- --------------------------------------------------------


--
-- Table structure for table 'STATION'
--

DROP TABLE IF EXISTS STATION;
CREATE TABLE IF NOT EXISTS STATION(
	sta_ID INT(10) NOT NULL AUTO_INCREMENT,
	sta_number INT(5) NOT NULL,
	sta_lat NUMERIC(8) NOT NULL,
	sta_lon NUMERIC(8) NOT NULL,
	sta_name VARCHAR(50) NOT NULL,
	sta_city VARCHAR(20) NOT NULL,
	sta_address VARCHAR(100) NOT NULL,
	sta_payment BOOL,
	sta_bonus BOOL,

	PRIMARY KEY (sta_ID)
	) ENGINE=InnoDB DEFAULT CHARSET=latin1
;


--
-- Table structure for table 'BIKE'
--

DROP TABLE IF EXISTS BIKE;
CREATE TABLE IF NOT EXISTS BIKE(
	bik_ID INT(10) NOT NULL AUTO_INCREMENT,
	bik_sta_ID INT(10),
	bik_timestamp DATE NOT NULL,
	bik_status enum('CLOSED','OPEN') NOT NULL,
	bik_stands INT(4) NOT NULL,
	bik_available_stands INT(4) NOT NULL,
	bik_available INT(4) NOT NULL,

	PRIMARY KEY (bik_ID)
	) ENGINE=InnoDB DEFAULT CHARSET=latin1
;


-- --------------------------------------------------------


ALTER TABLE `BIKE`
	ADD CONSTRAINT BIKE_STATION_FK FOREIGN KEY (bik_sta_ID) REFERENCES STATION (sta_ID) ON DELETE SET NULL ON UPDATE CASCADE;

