#include <SPI.h>
#include <Wire.h>
#include "Adafruit_MAX31855.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 32 // OLED display height, in pixels

#define MAXDO1 12
#define MAXCS1 13
#define MAXCLK1 15

#define MAXDO2 19
#define MAXCS2 23
#define MAXCLK2 5

#define I2C_SDA 33
#define I2C_SCL 32

#define N_POINTS 2000

// Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
#define OLED_RESET     -1 // Reset pin # (or -1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3C ///< See datasheet for Address; 0x3D for 128x64, 0x3C for 128x32

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

//Initialize the Thermocouple
Adafruit_MAX31855 thermocouple1(MAXCLK1, MAXCS1, MAXDO1);
Adafruit_MAX31855 thermocouple2(MAXCLK2, MAXCS2, MAXDO2);

char responseBuffer[50];
unsigned long dt = 5;  // 10 ms
unsigned long lcdInterval;
unsigned long lcdPreviousMillis;
unsigned long duration = 5000;
char dataLineBuffer[20];
double elapsedTime;


bool logFlag;
bool hasLog;
String logData;
unsigned long logStartTime;

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
  delay(10);
}


void setup() {
  Serial.begin(115200);
  delay(500);
  //I2CLCD.begin(I2C_SDA, I2C_SCL, 400000);
  // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  delay(500);
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
  logFlag = false;
  hasLog = false;
  lcdInterval = 200;
  lcdPreviousMillis = 0;
}
void loop() {
   double t1;
   double t2;
   double estimatedDt;

   unsigned long currentMillis = millis();
   char tempBuffer[50];
   unsigned long interval;
   String input;
   char rxChar;
//   double c1 = thermocouple1.readCelsius();
//   double c2 = thermocouple2.readCelsius();

   if(Serial.available()) {
	   input = Serial.readStringUntil(0x0D); // Read until line breaks
	   rxChar = input[0];
	   switch (rxChar) {
      case 0x69: // i as in id
        Serial.print("TCLOGGER\n");
        break;
  		case 0x6c: // l as in log
        elapsedTime = 0.0;
        logStartTime = currentMillis;
        logData = "";
  			logFlag = true;
        hasLog = false;
  			break;
  		case 0x74: // t
        logFlag = false;
  			if (input[1] == 0x3F) { // If '?' found
  				sprintf(responseBuffer, "%lu", duration);
  				Serial.print(responseBuffer);
          Serial.print("\n");
  				break;
  			}
  			duration = (unsigned long) atol(input.substring(2).c_str());
        estimatedDt = (duration + 500) / N_POINTS;
        dt = (unsigned long) max(estimatedDt, 5.0);
  			/*sprintf(responseBuffer, "%lu", dt);
  			Serial.print(responseBuffer);
        Serial.print("\n");*/
  			break;
      case 0x72: // r as in read
        logFlag = false;
        if (hasLog) {
          Serial.print(logData);
          Serial.print("\n");
          hasLog = false;
          break;
        }
        t1 = thermocouple1.readCelsius();
        t2 = thermocouple2.readCelsius();
        snprintf(tempBuffer, sizeof(tempBuffer), "%6.2f,%6.2f", t1, t2);
        Serial.print(tempBuffer);
        Serial.print("\n");
        break;
  		default:
        logFlag = false;
  			Serial.print("ERR_CMD");
        Serial.print("\n");
	   }
   }

  interval = logFlag ? dt : lcdInterval;

  if ((unsigned long)(currentMillis - lcdPreviousMillis) >= interval) {
      t1 = thermocouple1.readCelsius();
      t2 = thermocouple2.readCelsius();
      if (logFlag) {
        elapsedTime = (double) (currentMillis - logStartTime) / 1000.0;
        snprintf(dataLineBuffer, sizeof(dataLineBuffer), "%6.4f,%6.2f,%6.2f", elapsedTime, t1, t2);
        logData = logData + dataLineBuffer + ";";
        hasLog = true;
        if (int(elapsedTime * 1000.0) >= duration + 500) {
          logFlag = false;
        }
      } else {
        lcdTemperature(t1, t2);
      }
      lcdPreviousMillis = currentMillis;
  }

}