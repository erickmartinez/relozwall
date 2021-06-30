#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include "Adafruit_SCD30.h"
#include "arduino_secrets.h"


const char* ssid = SECRET_SSID;
const char* password = SECRET_PWD;

const char* http_username = SECRET_HTTP_USER; 
const char* http_password = SECRET_HTTP_PWD; 

// Web server running on port 80
WebServer server(80);

// Sensor
Adafruit_SCD30  scd30;
float temperature;
float humidity;
float co2;
bool sensor_detected = 0;

unsigned long lastTime = 0;  
unsigned long timerDelay = 1000;  // send readings timer

// JSON data buffer
StaticJsonDocument<300> jsonDocument;
char buffer[350];

bool ledState = 0;
const int ledPin = 2;

void init_scd30() {
  if (!scd30.begin()) {
    Serial.println("Failed to find SCD30 chip.");
    sensor_detected = 0;
  }
  Serial.println("SCD30 found!");
  Serial.print("Measurement Interval: ");
  Serial.print(scd30.getMeasurementInterval()); 
  Serial.println(" seconds");
  sensor_detected = 1;
}

void getSCD30Readings(){
  // Check whether new data is available.
  if (scd30.dataReady()) {
    if (!scd30.read()){ Serial.println("Error reading sensor data"); return; }
    
  }
  temperature = scd30.temperature;
  humidity = scd30.relative_humidity;
  co2 = scd30.CO2;
}

void connectToWifi() {
  /*Serial.print("MAC Address: ");
  Serial.println(WiFi.macAddress());*/
  /*esp_log_level_set("wifi", ESP_LOG_VERBOSE); 
  Serial.setDebugOutput(true);*/
  WiFi.begin(ssid, password);
  Serial.print("Connecting to ");
  Serial.println(ssid);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("Connected to the WiFi network.");
  Serial.println(WiFi.localIP());
  Serial.print("MAC Address: ");
  Serial.println(WiFi.macAddress());
}

void setup_routing() {
  server.on("/temperature", getTemperature);
  server.on("/humidity", getHumidity);     
  server.on("/env", getEnv);     
  server.on("/led", toggleLed);
  server.on("/mac", getMacAddress);
  server.on("/led-auth",[]() {
    if (!server.authenticate(http_username, http_password))
      return server.requestAuthentication();
    toggleLed();
  });
  server.on("/logout", []() {
    server.send(401,"text/plain", "Forbidden!");
  }); 

  // start server     
  server.begin();
  Serial.println("Setup restless server...");
}

void create_json(char *tag, float value, char *unit) {  
  jsonDocument.clear();  
  jsonDocument["type"] = tag;
  jsonDocument["value"] = value;
  jsonDocument["unit"] = unit;
  serializeJson(jsonDocument, buffer);
}
 
void add_json_object(char *tag, float value, char *unit) {
  JsonObject obj = jsonDocument.createNestedObject();
  obj["type"] = tag;
  obj["value"] = value;
  obj["unit"] = unit; 
}

void getTemperature() {
  Serial.println("Get temperature");
  create_json("temperature", temperature, "°C");
  server.send(200, "application/json", buffer);
}
 
void getHumidity() {
  Serial.println("Get humidity");
  create_json("humidity", humidity, "%");
  server.send(200, "application/json", buffer);
}
 
void getCO2() {
  Serial.println("Get CO2 content");
  create_json("CO2", co2, "ppm");
  server.send(200, "application/json", buffer);
}
 
void getEnv() {
  Serial.println("Get env");
  jsonDocument.clear();
  add_json_object("temperature", temperature, "°C");
  add_json_object("humidity", humidity, "%");
  add_json_object("CO2", co2, "ppm");
  serializeJson(jsonDocument, buffer);
  server.send(200, "application/json", buffer);
}

void toggleLed() {
  ledState = !ledState;
  Serial.println("Led state: ");
  Serial.print(ledState);
  Serial.println("");
  digitalWrite(ledPin, ledState);
  jsonDocument.clear();  
  jsonDocument["ledState"] = ledState;
  serializeJson(jsonDocument, buffer);
  server.send(200, "application/json", buffer);
}

void getMacAddress() {
  server.send(200, "application/text", WiFi.macAddress());
}


void setup() {
  Serial.begin(115200);
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);
  connectToWifi();
  // Init SCD30 sensor
  init_scd30();
  if (sensor_detected) {
    setup_routing(); 
  }
  
}

void loop() {
  if ((millis() - lastTime) > timerDelay && sensor_detected ) {
    getSCD30Readings();
    lastTime = millis();
  }
  if (sensor_detected) {
    server.handleClient();
  }
}
