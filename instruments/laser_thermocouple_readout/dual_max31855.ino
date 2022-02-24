#include <SPI.h>
#include <Wire.h>
#include "Adafruit_MAX31855.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 32 // OLED display height, in pixels

#define MAXDO 12
#define MAXCS 13
#define MAXCLK 15

#define MAXDO_2 19
#define MAXCS_2 23
#define MAXCLK_2 5

#define I2C_SDA 33
#define I2C_SCL 32

// Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
#define OLED_RESET     -1 // Reset pin # (or -1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3C ///< See datasheet for Address; 0x3D for 128x64, 0x3C for 128x32

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

//Initialize the Thermocouple
Adafruit_MAX31855 thermocouple(MAXCLK, MAXCS, MAXDO);
Adafruit_MAX31855 thermocouple2(MAXCLK_2, MAXCS_2, MAXDO_2);

String timesStr;
String tc1Str;
String tc2Str;



void lcdTemperature(double t1, double t2) {
  char buff[28]; // Buffer big enough for 7-character float
  display.clearDisplay();
  display.setTextSize(1);      // Normal 1:1 pixel scale
  display.setTextColor(SSD1306_WHITE); // Draw white text
  display.cp437(true);         // Use full 256 char 'Code Page 437' font
  display.setCursor(0, 0);     // Start at top-left corner
  snprintf(buff, sizeof(buff), "TC1: %4.1f C\nTC2: %4.1f C", t1, t2);
  display.println(buff);
  display.display();
  delay(10);
}

void setup() {
  Serial.begin(115200); 
  Serial.println("MAX31855 test");  //Wait for MAX chip to stabilize
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
   // Basic readout test, just print the current temp
   // Serial.print("Temperature:");
   double c = thermocouple.readCelsius();
   delay(10);
   double c2 = thermocouple2.readCelsius();
   
   if (isnan(c)) {
     Serial.println ("Something wrong with thermocouple!");
   } else {
     Serial.print("TCa: " + String(c) + " °C, ");
     //Serial.print("TCb: " + String(c2) + "C, ");
     Serial.println("TCb: " + String(c2) + " °C");
     // Serial.print(String(thermocouple.readFarenheit()) + "F,");
     //Serial.println("TCa: " + String(thermocouple.readInternal()) + "C (internal)");
     lcdTemperature(c, c2);
   }
   delay(1000);
}
