# PawPath firmware

Arduino sketch for the sensor node: ESP32 + MPU6050 IMU.

## Hardware

- ESP32 dev board (built-in BLE + Wi-Fi)
- MPU6050 (or MPU9250/BMX055 — swap the driver library accordingly)
- microSD card module (SPI) for onboard logging so a session survives a dropped BLE link
- LiPo battery, 500-1000 mAh

Wiring defaults (see comments in `pawpath_node/pawpath_node.ino` to change pins):

| Signal | ESP32 pin |
|--------|-----------|
| MPU6050 SDA | GPIO21 |
| MPU6050 SCL | GPIO22 |
| SD CS | GPIO5 |
| SD MOSI | GPIO23 |
| SD MISO | GPIO19 |
| SD SCK | GPIO18 |

## Arduino libraries required

Install via Library Manager:
- `I2Cdevlib-MPU6050` (electroniccats/jrowberg fork both work)
- `ESP32 BLE Arduino` (bundled with the ESP32 board package)
- `SD` (bundled with Arduino core)

## Behavior

- Samples accel + gyro at 100 Hz (FR-1).
- Streams each sample as a CSV line over a BLE UART-style notify characteristic (FR-7).
- Appends the same line to `/pawpath_log.csv` on the SD card, so BLE disconnects don't lose data.
- CSV schema: `millis,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps`.

## Next steps

- Add per-dog calibration (FR-9): capture a still-standing baseline on boot and subtract bias.
- Add a second/third node for per-limb placement once back/chest-only classification is validated (see literature review in `docs/PRD.md` section 3 — back/chest outperforms neck, and per-limb enables lameness localization).
- Add low-battery cutoff and a status LED.
