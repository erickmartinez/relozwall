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

// Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
#define OLED_RESET     -1 // Reset pin # (or -1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3C ///< See datasheet for Address; 0x3D for 128x64, 0x3C for 128x32

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

//Initialize the Thermocouple
Adafruit_MAX31855 thermocouple1(MAXCLK1, MAXCS1, MAXDO1);
Adafruit_MAX31855 thermocouple2(MAXCLK2, MAXCS2, MAXDO2);

String timesStr;
unsigned long maxTime = 1000;
char responseBuffer[50];
unsigned long dt = 10;  // 10 ms
unsigned long lcdInterval = 300;

unsigned long lcdPreviousMillis = 0;

void lcdTemperature(double t1, double t2) {
  char buff[26]; // Buffer big enough for 7-character float
  display.clearDisplay();
  display.setTextSize(1);      // Normal 1:1 pixel scale
  display.setTextColor(SSD1306_WHITE); // Draw white text
  display.cp437(true);         // Use full 256 char 'Code Page 437' font
  display.setCursor(0, 0);     // Start at top-left corner
  snprintf(buff, sizeof(buff), "TC1: %5.1f C\nTC2: %5.1f C", t1, t2);
  display.println(buff);
  display.display();
  delay(1);
}

void logTemperature(void) {
	unsigned long currentMillis;
	unsigned long previousMillis = 0;
	unsigned long totalTime = 0;
	unsigned long startMillis = millis();
	char buff[50]; // Buffer big enough for 7-character float
	while (totalTime <= maxTime) {
    currentMillis = millis();
		if ((unsigned long)(currentMillis - previousMillis) >= dt) {
			double t1 = thermocouple1.readCelsius();
			double t2 = thermocouple2.readCelsius();
			snprintf(buff, sizeof(buff), "%5lu (ms), TC1: %5.1f °C, TC2: %5.1f °C", totalTime, t1, t2);
      previousMillis = currentMillis;
			Serial.println(buff);
		}
    totalTime = millis() - startMillis;
	}

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
  delay(2000); // Pause for 2 seconds

  // Clear the buffer
  display.clearDisplay();

  // Draw a single pixel in white
  display.drawPixel(10, 10, WHITE);
}
void loop() {
   double t1;
   double t2;

   unsigned long currentMillis = millis();
   char tempBuffer[50];

   String input;
   char rxChar;
//   double c1 = thermocouple1.readCelsius();
//   double c2 = thermocouple2.readCelsius();

   if(Serial.available()) {
	   input = Serial.readStringUntil(0x0D); // Read until line breaks
	   rxChar = input[0];
	   switch (rxChar) {
  		case 0x6c: // l as in log
  			logTemperature();
  			break;
  		case 0x74: // t
  			if (input[1] == 0x3F) { // If '?' found
  				sprintf(responseBuffer, "%lu", maxTime);
  				Serial.println(responseBuffer);
  				break;
  			}
  			maxTime = (unsigned long) atol(input.substring(2).c_str());
  			sprintf(responseBuffer, "%lu", maxTime);
  			Serial.println(responseBuffer);
  			break;
      case 0x72: // d as in display
        t1 = thermocouple1.readCelsius();
        t2 = thermocouple2.readCelsius();
        snprintf(tempBuffer, sizeof(tempBuffer), "%5.1f,%5.1f", t1, t2);
        Serial.println(tempBuffer);
        break;
  		default:
  			Serial.println("ERR_CMD");
	   }
   }

  if ((unsigned long)(currentMillis - lcdPreviousMillis) >= lcdInterval) {
      t1 = thermocouple1.readCelsius();
      t2 = thermocouple2.readCelsius();
      lcdTemperature(t1, t2);
      lcdPreviousMillis = currentMillis;
    }

}