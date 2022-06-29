#include <SPI.h>
#include <Wire.h>
#include "Adafruit_MAX31855.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include "HX711.h"
#include "arduino_secrets.h"

const char* ssid = SECRET_SSID;
const char* password = SECRET_PWD;

WiFiServer wifiServer(3001);
WiFiClient client;

IPAddress local_IP(192, 168, 4, 2);
IPAddress gateway(192, 168, 0, 1);
IPAddress subnet(255, 255, 240, 0);

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 32 // OLED display height, in pixels

#define MAXDO1 12
#define MAXCS1 13
#define MAXCLK1 15

#define MAXDO2 19
#define MAXCS2 23
#define MAXCLK2 5

#define I2C_SDA  26 //33
#define I2C_SCL  25 //32

#define POT_PIN 36

#define BUFF_SIZE 100

// HX711 circuit wiring
const int LOADCELL_DOUT_PIN = 33; //21
const int LOADCELL_SCK_PIN = 32; //22

HX711 scale;

TwoWire I2C_SSD1306 = TwoWire(0);

// Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
#define OLED_RESET     -1 // Reset pin # (or -1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3C ///< See datasheet for Address; 0x3D for 128x64, 0x3C for 128x32

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

//Initialize the Thermocouple
Adafruit_MAX31855 thermocouple1(MAXCLK1, MAXCS1, MAXDO1);
Adafruit_MAX31855 thermocouple2(MAXCLK2, MAXCS2, MAXDO2);

float calibrationFactor = 7.8764E+03;
long zeroFactor;
unsigned long lcdInterval, interval;
unsigned long lcdPreviousMillis, previousMillis, lcdWCPreviousMillis;
unsigned long currentMillis, currentWCMillis;

char responseBuffer[BUFF_SIZE];
int adcAverages = 5;
double inputCalibration;

float t1, t2, f;
long loadcellADC;
uint16_t potADC;

unsigned int columns = 5;
unsigned int outputBufferSize = 3*sizeof(float)+sizeof(long)+sizeof(uint16_t);

String input;
char rxChar;

typedef struct {
  float tc1;
  float tc2;
  float force;
  long loadcellADC;
  uint16_t potADC;
} readingData;

readingData binaryData = {0.0, 0.0, 0.0, 0, 0};

float adc2inches(int value) {
  return 3.327 + 0.01303 * value;
}

void lcdUpdate(float t1, float t2, float f, int pot) {
  char buff[60]; // Buffer big enough for 7-character float
  display.clearDisplay();
  display.setTextSize(1);      // Normal 1:1 pixel scale
  display.setTextColor(SSD1306_WHITE); // Draw white text
//  display.cp437(true);         // Use full 256 char 'Code Page 437' font
  display.setCursor(0, 0);     // Start at top-left corner
  snprintf(buff, sizeof(buff), "TC1: %6.2f \367C\nTC2: %6.2f \367C\nF:   %7.1f N\nD: %5.1f\"", t1, t2, f, adc2inches(pot));
  display.println(buff);
  display.display();
  //delay(1);
}

double readTC(uint8_t tcIndex) {
  double result = NAN;
  if (tcIndex == 1) {
    result = thermocouple1.readCelsius();
  } else if (tcIndex == 2) {
    result = thermocouple2.readCelsius();
  }
  if (isnan(result)) {
    delay(5);
    return readTC(tcIndex);
  }
  return result;
}

int savePotRead() {
  int adc = NAN;
  adc = analogRead(POT_PIN);
  if (isnan(adc)) {
    delay(2);
    return savePotRead();
  }
  return adc;
}

int adcAverage() {
  float num = 0;
  for (int i=0; i<adcAverages; ++i){
    num += savePotRead();
    //delay(1);
  }
  return (int) num / adcAverages;
}

void scanI2C() {
  byte error, address;
  int nDevices;
  Serial.println("Scanning...");
  nDevices = 0;
  for(address = 1; address < 127; address++ ) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    if (error == 0) {
      Serial.print("I2C device found at address 0x");
      if (address<16) {
        Serial.print("0");
      }
      Serial.println(address,HEX);
      nDevices++;
    }
    else if (error==4) {
      Serial.print("Unknow error at address 0x");
      if (address<16) {
        Serial.print("0");
      }
      Serial.println(address,HEX);
    }
  }
  if (nDevices == 0) {
    Serial.println("No I2C devices found\n");
  }
  else {
    Serial.println("done\n");
  }
}

void initWiFi() {
  WiFi.mode(WIFI_STA);
  // Configures static IP address
  if (!WiFi.config(local_IP, gateway, subnet)) {
    Serial.println("STA Failed to configure");
  }
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
  wifiServer.begin();
}

void setup() {
  Serial.begin(115200);
  initWiFi();
  // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  I2C_SSD1306.begin(I2C_SDA, I2C_SCL, 100000);
  Wire.begin(I2C_SDA, I2C_SCL, 100000);

  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  scale.set_scale(calibrationFactor);
  scale.tare(10); //Reset the scale to 0
  zeroFactor = scale.read_average(); //Get a baseline reading

  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;); // Don't proceed, loop forever
  }

  // Show initial display buffer contents on the screen --
  // the library initializes this with an Adafruit splash screen.
  display.display();
  delay(1000); // Pause for 1 seconds

  // Clear the buffer
  display.clearDisplay();

  // Draw a single pixel in white
  display.drawPixel(10, 10, WHITE);
  lcdInterval = 250;
  interval = 30000;
  lcdPreviousMillis = 0;
  previousMillis = 0;
  lcdWCPreviousMillis = 0;
}

void loop() {
   client = wifiServer.available();
   currentMillis = millis();

   if (client) {
    while (client.connected()) {
      currentWCMillis = millis();
      if ((unsigned long)(currentWCMillis - lcdWCPreviousMillis) >= lcdInterval) {
          t1 = readTC(1); //thermocouple1.readCelsius();
          t2 = readTC(2); //thermocouple2.readCelsius();
          f = scale.get_units(3);
          potADC = adcAverage();
          lcdUpdate(t1, t2, float(f), potADC);
          lcdWCPreviousMillis = currentWCMillis;
      }
      if (client.available()>0) {
        input = "";
        input = client.readStringUntil(0x0D);
        /*Serial.print(input);
        Serial.print('\n');
        Serial.flush();*/
        rxChar = input[0];
        switch (rxChar) {
          case 0x69: // i as in id
            client.print("EXTRUDER_READOUT\n");
            break;
          case 0x7a: // z as in zero
            scale.tare(20); //Reset the scale to 0
            zeroFactor = scale.read_average(); //Get a baseline reading
            snprintf(responseBuffer, BUFF_SIZE, "%lu", zeroFactor);
            client.print(responseBuffer);
            client.print("\n");
            break;
          case 0x72: // r as in read
            t1 = readTC(1); //thermocouple1.readCelsius();
            t2 = readTC(2); //thermocouple2.readCelsius();
            f = scale.get_units(1);
            loadcellADC = scale.read();
            potADC = adcAverage();
            binaryData = {t1, t2, f, loadcellADC, potADC};

            client.write((const uint8_t  *)&outputBufferSize, sizeof(unsigned int));
            client.write((const uint8_t  *)&columns, sizeof(unsigned int));
            client.write((const uint8_t  *)&binaryData, outputBufferSize);
            // snprintf(responseBuffer, BUFF_SIZE, "%6.2f,%6.2f,%5.1f,%6ld,%5ld", t1, t2, f, loadcellADC, potADC);
            // Serial.print(responseBuffer);
            // Serial.print("\n");
            break;
          case 0x63: // c as in calibration
            if (input[1] == 0x3F) { // If '?' found
              snprintf(responseBuffer, BUFF_SIZE, "%10.3E", calibrationFactor);
              client.print(responseBuffer);
              client.print("\n");
              break;
            }
            inputCalibration = input.substring(2).toFloat();
            if (abs(inputCalibration) > 0.0)
              calibrationFactor = inputCalibration;
              scale.set_scale(calibrationFactor);
            break;
          case 0x73: // s as in scan
            scanI2C();
            break;
          default:
            client.print("ERR_CMD");
            client.print("\n");
        }
      }
      //delay(10);
    }
    client.stop();
    Serial.println("Client disconnected");
  }

  // currentMillis = millis();
  if ((unsigned long)(currentMillis - lcdPreviousMillis) >= lcdInterval) {
    t1 = readTC(1); //thermocouple1.readCelsius();
    t2 = readTC(2); //thermocouple2.readCelsius();
    f = scale.get_units(3);
    potADC = adcAverage();
    lcdUpdate(t1, t2, float(f), potADC);
    lcdPreviousMillis = currentMillis;
  }

  if ((WiFi.status() != WL_CONNECTED) && (currentMillis - previousMillis >=interval)) {
    WiFi.disconnect();
    WiFi.reconnect();
    previousMillis = currentMillis;
  }

  if(Serial.available()) {
    input = Serial.readStringUntil(0x0D); // Read until line breaks
    rxChar = input[0];
    switch (rxChar) {
      case 0x69: // i as in id
        Serial.print("EXTRUDER_READOUT\n");
        break;
      default:
        Serial.print("ERR_CMD\n");
    }
  }
  
}