#include <SPI.h>
#include <Wire.h>
#include "Adafruit_MAX31855.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include "arduino_secrets.h"

const char* ssid = SECRET_SSID;
const char* password = SECRET_PWD;

WiFiServer wifiServer(3001);
WiFiClient client;

IPAddress local_IP(192, 168, 4, 3);
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

#define I2C_SDA 21 //33
#define I2C_SCL 22 //32

#define N_POINTS 2000

// Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
#define OLED_RESET     -1 // Reset pin # (or -1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3C ///< See datasheet for Address; 0x3D for 128x64, 0x3C for 128x32

TwoWire I2C_SSD1306 = TwoWire(0);
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

//Initialize the Thermocouple
Adafruit_MAX31855 thermocouple1(MAXCLK1, MAXCS1, MAXDO1);
Adafruit_MAX31855 thermocouple2(MAXCLK2, MAXCS2, MAXDO2);

char responseBuffer[50];
unsigned long dt = 5;  // 10 ms
unsigned long lcdInterval, wifiInterval;
unsigned long lcdPreviousMillis, wifiPreviousMillis, logPreviousMillis, logCurrentMillis, sensorsPreviousMillis;
unsigned long currentMillis;
unsigned long duration = 5000;
char tempBuffer[50];
float elapsedTime;
String input;
double estimatedDt;
char rxChar;


unsigned long logStartTime;
float logDataBinary[N_POINTS][3];
float readingData[3];
float t1, t2;
unsigned int logBufferSize, count, columns;
uint8_t failedAttemptsTC1 = 0;
uint8_t failedAttemptsTC2 = 0;

void readTC1() {
  double r = thermocouple1.readCelsius();
  if (!isnan(r)) {
    t1 = r;
    failedAttemptsTC1 = 0;
  } else {
    failedAttemptsTC1++;
    if (failedAttemptsTC1 <= 3) {
      delay(1);
      readTC1();
    }
  }
}

void readTC2() {
  double r = thermocouple2.readCelsius();
  if (!isnan(r)) {
    t2 = r;
    failedAttemptsTC2 = 0;
  } else {
    failedAttemptsTC2++;
    if (failedAttemptsTC2 <= 3) {
      delay(1);
      readTC2();
    }
  }
}

void lcdTemperature(double t1, double t2) {
  char buff[30]; // Buffer big enough for 7-character float
  display.clearDisplay();
  display.setTextSize(1);      // Normal 1:1 pixel scale
  display.setTextColor(SSD1306_WHITE); // Draw white text
//  display.cp437(true);         // Use full 256 char 'Code Page 437' font
  display.setCursor(0, 0);     // Start at top-left corner
  display.println("Laser Stand");
  display.println("--------------");
  snprintf(buff, sizeof(buff), "TC1: %6.2f \367C\nTC2: %6.2f \367C", t1, t2);
  display.println(buff);
  display.display();
  //delay(10);
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
  I2C_SSD1306.begin(I2C_SDA, I2C_SCL, 100000);
  Wire.begin(I2C_SDA, I2C_SCL, 100000);
  //I2CLCD.begin(I2C_SDA, I2C_SCL, 400000);
  // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  // delay(500);
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
  count = 0;
  lcdInterval = 250;
  lcdPreviousMillis = 0;
  wifiInterval = 30000;
  wifiPreviousMillis = 0;
  sensorsPreviousMillis = 0;
}


void loop() {
  client = wifiServer.available();
  currentMillis = millis();

  if ((WiFi.status() != WL_CONNECTED) && (currentMillis - wifiPreviousMillis >= wifiInterval)) {
    WiFi.disconnect();
    WiFi.reconnect();
    wifiPreviousMillis = currentMillis;
  }

  if ((!client) || (!client.connected())) {
    if ((unsigned long) (currentMillis - sensorsPreviousMillis) >= 2) {
      readTC1();
      readTC2();
      sensorsPreviousMillis = currentMillis;
    }

    if ((unsigned long)(currentMillis - lcdPreviousMillis) >= lcdInterval) {
        lcdTemperature(t1, t2);
        lcdPreviousMillis = currentMillis;
    }
  }


  if (client) {
    while (client.connected()) {
      currentMillis = millis();

      if ((unsigned long) (currentMillis - sensorsPreviousMillis) >= 2) {
        readTC1();
        readTC2();
        sensorsPreviousMillis = currentMillis;
      }

      if ((unsigned long)(currentMillis - lcdPreviousMillis) >= lcdInterval) {
          lcdTemperature(t1, t2);
          lcdPreviousMillis = currentMillis;
      }

      if (client.available()>0) {
        input = "";
        input = client.readStringUntil(0x0D);
        rxChar = input[0];
        switch (rxChar) {
          case 0x69: // i as in id
            client.print("TCLOGGER\n");
            break;
          case 0x6c: // l as in log
            elapsedTime = 0.0;
            count = 0;
            estimatedDt = (double) (duration + 500) / N_POINTS;
            dt = (unsigned long) max(estimatedDt, 5.0);
            // snprintf(tempBuffer, sizeof(tempBuffer), "dt: %lu ms\n",dt);
            // Serial.print(tempBuffer);
            logPreviousMillis = 0;
            elapsedTime = 0.0;
            logStartTime = millis();

            while ((elapsedTime < (duration + 500)/1000.0) && (count < N_POINTS)) {
              logCurrentMillis = millis();
              if ( (unsigned long) (logCurrentMillis - logPreviousMillis) >= dt) {
                if (client.available()>0) {
                  break;
                }
                readTC1();
                readTC2();
                elapsedTime = (float) (logCurrentMillis - logStartTime) / 1000.0;
                logDataBinary[count][0] = elapsedTime;
                logDataBinary[count][1] = t1;
                logDataBinary[count][2] = t2;
                ++count;
                logPreviousMillis = logCurrentMillis;
                //snprintf(tempBuffer, sizeof(tempBuffer), "%5.3f s %6.2f C,%6.2f C\n", elapsedTime, t1, t2);
                //Serial.print(tempBuffer);
              }
            }
            break;
          case 0x74: // t
            if (input[1] == 0x3F) { // If '?' found
              sprintf(responseBuffer, "%lu", duration);
              client.print(responseBuffer);
              client.print("\n");
              break;
            }
            duration = (unsigned long) atol(input.substring(2).c_str());
            break;
          case 0x72: // r as in read
            if (count > 0) {
              logBufferSize = N_POINTS*3*sizeof(float);
              columns = 3;
              client.write((const uint8_t *)&logBufferSize, sizeof(unsigned int));
              client.write((const uint8_t *)&columns, sizeof(unsigned int));
              client.write((const uint8_t *)&logDataBinary, N_POINTS*3*sizeof(float));
              count = 0;
              break;
            }
            readingData[0] = t1; // thermocouple1.readCelsius();
            readingData[1] = t2; // thermocouple2.readCelsius();
            logBufferSize = 2*sizeof(float);
            columns = 2;
            client.write((const uint8_t  *)&logBufferSize, sizeof(unsigned int));
            client.write((const uint8_t  *)&columns, sizeof(unsigned int));
            client.write((const uint8_t  *)&readingData, logBufferSize);
            break;
          default:
            client.print("ERR_CMD");
            client.print("\n");
        } // switch (rxChar)
      } // if (client.available()>0)
    } // if (client.connected())
    client.stop();
    // Serial.println("Client disconnected");
  } // if (client)

  if(Serial.available()) {
    input = Serial.readStringUntil(0x0D); // Read until line breaks
    rxChar = input[0];
    switch (rxChar) {
      case 0x69: // i as in id
        Serial.print("TCLOGGER\n");
        break;
      default:
        Serial.print("ERR_CMD\n");
    }
  }

}