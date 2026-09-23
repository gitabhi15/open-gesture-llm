#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// --- WIFI CONFIGURATION (CHANGE THESE) ---
const char* ssid = "Abhiram's F17";
const char* password = "plxu3733";
const char* laptop_ip = "10.200.147.139";  // YOUR LAPTOP'S IP ADDRESS
const int udp_port = 5005;                // Port to send data to

WiFiUDP udp;
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

void setup() {
  Serial.begin(115200); 
  Wire.begin(21, 22);

  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWi-Fi connected!");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  if (!accel.begin()) {
    Serial.println("ADXL345 not detected");
    while (1);
  }
}

void loop() {
  sensors_event_t event;
  accel.getEvent(&event);

  // Format the data as a CSV string
  char payload[64];
  snprintf(payload, sizeof(payload), "%.2f,%.2f,%.2f\n", 
           event.acceleration.x, event.acceleration.y, event.acceleration.z);

  // Send over Wi-Fi (UDP)
  udp.beginPacket(laptop_ip, udp_port);
  udp.print(payload);
  udp.endPacket();

  // Also print to USB Serial for debugging just in case
  Serial.print(payload);

  delay(50);  // ~20 Hz
}