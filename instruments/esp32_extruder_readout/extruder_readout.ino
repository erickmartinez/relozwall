#include <SPI.h>
#include <Wire.h>
#include "Adafruit_MAX31855.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "HX711.h"

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

float calibrationFactor = 14651;
long zeroFactor;
unsigned long lcdInterval;
unsigned long lcdPreviousMillis;
char responseBuffer[BUFF_SIZE];
int adcAverages = 3;

double t1, t2, f;
long reading, potADC;

void lcdUpdate(float t1, float t2, float f, int pot) {
  char buff[60]; // Buffer big enough for 7-character float
  display.clearDisplay();
  display.setTextSize(1);      // Normal 1:1 pixel scale
  display.setTextColor(SSD1306_WHITE); // Draw white text
//  display.cp437(true);         // Use full 256 char 'Code Page 437' font
  display.setCursor(0, 0);     // Start at top-left corner
  snprintf(buff, sizeof(buff), "TC1: %6.2f \367C\nTC2: %6.2f \367C\nF:   %7.1f N\nPot: %5d", t1, t2, f, pot);
  display.println(buff);
  display.display();
  delay(10);
}


int adcAverage() {
  float num = 0;
  for (int i=0; i<adcAverages; ++i){
    num += analogRead(POT_PIN);
  }
  return int(num / adcAverages);
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


void setup() {
  Serial.begin(115200); 
  delay(500);
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
  lcdInterval = 200;
  lcdPreviousMillis = 0;
}
void loop() {
   double inputCalibration;
   unsigned long currentMillis = millis();
   String input;
   char rxChar;

   if(Serial.available()) {
     input = Serial.readStringUntil(0x0D); // Read until line breaks
     rxChar = input[0];
     switch (rxChar) {
      case 0x69: // i as in id
        Serial.print("EXTRUDER_READOUT\n");
        break;
      case 0x7a: // z as in zero
        scale.tare(20); //Reset the scale to 0
        zeroFactor = scale.read_average(); //Get a baseline reading
        snprintf(responseBuffer, BUFF_SIZE, "%lu", zeroFactor);
        Serial.print(responseBuffer);
        Serial.print("\n");
        break;
      case 0x72: // r as in read
        t1 = thermocouple1.readCelsius();
        delay(0.01);
        t2 = thermocouple2.readCelsius();
        delay(0.01);
        f = scale.get_units(3);
        delay(0.01);
        reading = scale.read();
        delay(0.01);
        potADC = adcAverage();
        snprintf(responseBuffer, BUFF_SIZE, "%6.2f,%6.2f,%5.1f,%6ld,%5ld", t1, t2, f, reading, potADC);
        Serial.print(responseBuffer);
        Serial.print("\n");
        break;
      case 0x63: // c as in calibration
        if (input[1] == 0x3F) { // If '?' found
          snprintf(responseBuffer, BUFF_SIZE, "%10.3E", calibrationFactor);
          Serial.print(responseBuffer);
          Serial.print("\n");
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
        Serial.print("ERR_CMD");
        Serial.print("\n");
     }
   }

  if ((unsigned long)(currentMillis - lcdPreviousMillis) >= lcdInterval) {
      t1 = thermocouple1.readCelsius();
      delay(0.01);
      t2 = thermocouple2.readCelsius();
      delay(0.01);
      f = scale.get_units(3);
      delay(0.01);
      potADC = adcAverage();
      lcdUpdate(t1, t2, float(f), potADC);
      lcdPreviousMillis = currentMillis;
  }
  
}