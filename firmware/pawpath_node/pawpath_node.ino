/*
 * PawPath sensor node firmware
 * Target: ESP32 (Arduino core) + MPU6050 (I2C)
 *
 * Reads 3-axis accel + 3-axis gyro at SAMPLE_RATE_HZ (FR-1), streams each
 * sample over BLE as a CSV line (FR-7) and appends it to a CSV file on the
 * SD card so a session survives a dropped BLE connection.
 *
 * CSV schema per line: millis,ax,ay,az,gx,gy,gz
 *   ax/ay/az in g, gx/gy/gz in deg/s.
 *
 * Wiring (default ESP32 I2C pins): MPU6050 SDA->GPIO21, SCL->GPIO22.
 * SD card via SPI: CS->GPIO5, MOSI->GPIO23, MISO->GPIO19, SCK->GPIO18.
 */

#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <MPU6050.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ---- Configuration ----
static const uint32_t SAMPLE_RATE_HZ = 100;         // FR-1: >=100 Hz
static const uint32_t SAMPLE_PERIOD_MS = 1000 / SAMPLE_RATE_HZ;
static const int SD_CS_PIN = 5;
static const char *LOG_FILENAME = "/pawpath_log.csv";
static const char *DEVICE_NAME = "PawPath-Node";

#define SERVICE_UUID        "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define CHARACTERISTIC_UUID "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

MPU6050 imu;
File logFile;
BLECharacteristic *txCharacteristic;
bool bleClientConnected = false;
uint32_t lastSampleMs = 0;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override { bleClientConnected = true; }
  void onDisconnect(BLEServer *server) override {
    bleClientConnected = false;
    server->getAdvertising()->start();  // resume advertising for reconnect
  }
};

void setupBLE() {
  BLEDevice::init(DEVICE_NAME);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *service = server->createService(SERVICE_UUID);
  txCharacteristic = service->createCharacteristic(
      CHARACTERISTIC_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  txCharacteristic->addDescriptor(new BLE2902());

  service->start();
  server->getAdvertising()->addServiceUUID(SERVICE_UUID);
  server->getAdvertising()->start();
}

void setupSD() {
  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("SD init failed; continuing with BLE-only streaming.");
    return;
  }
  logFile = SD.open(LOG_FILENAME, FILE_APPEND);
  if (logFile) {
    logFile.println("millis,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
    logFile.flush();
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  imu.initialize();
  if (!imu.testConnection()) {
    Serial.println("MPU6050 connection failed. Check wiring.");
  }

  setupSD();
  setupBLE();

  lastSampleMs = millis();
}

void loop() {
  uint32_t now = millis();
  if (now - lastSampleMs < SAMPLE_PERIOD_MS) return;
  lastSampleMs = now;

  int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
  imu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);

  // Default MPU6050 full-scale ranges: accel +-2g, gyro +-250 dps.
  float ax = ax_raw / 16384.0f;
  float ay = ay_raw / 16384.0f;
  float az = az_raw / 16384.0f;
  float gx = gx_raw / 131.0f;
  float gy = gy_raw / 131.0f;
  float gz = gz_raw / 131.0f;

  char line[96];
  snprintf(line, sizeof(line), "%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
           (unsigned long)now, ax, ay, az, gx, gy, gz);

  if (bleClientConnected) {
    txCharacteristic->setValue((uint8_t *)line, strlen(line));
    txCharacteristic->notify();
  }

  if (logFile) {
    logFile.println(line);
    // Flush periodically rather than every line to limit SD wear/latency.
    if (now % 1000 < SAMPLE_PERIOD_MS) logFile.flush();
  }
}
