/*
cjones@ucsd.edu, erm013@ucsd.edu
Linear Displacement
*/

int posPin = A0;

// store read results for sending to the LCD and data storage
unsigned int adcVal, prevAdcVal;
unsigned long t0, currentMillis, previousMillis;

String input;
char rxChar;

typedef struct {
  unsigned long elapsedTime;
  unsigned int adcVal;
} readingStruct;

readingStruct reading = {0.0, 0.0};
unsigned int outputBufferSize = sizeof(reading);


void setup() {
  Serial.begin(57600);
  prevAdcVal = 0;
  previousMillis = 0;
  t0 = millis();
  /*while(!Serial);

  Serial.println("Time(ms), ADC raw");*/
}

void loop() {
  currentMillis = millis();
  if ((unsigned long) (currentMillis - previousMillis) >= 10 ){
    adcVal = analogRead(posPin);
  }

  if(Serial.available()) {
    input = Serial.readStringUntil(0x0D); // Read until line breaks
    rxChar = input[0];
    switch (rxChar) {
      case 0x69: // i as in id
        Serial.print("DEFLECTION_POT\n");
        break;
      case 0x7a: // z as in zero
        t0 = millis();
        break;
      case 0x72: // r as in read
        reading.elapsedTime = currentMillis - t0;
        reading.adcVal = adcVal;
        byte *b = (byte *) &outputBufferSize;
        Serial.write(b, 4);
        byte *d = (byte *) &reading;
        Serial.write(d, outputBufferSize);
        break;
      default:
        Serial.print("ERR_CMD\n");
    }
  }
}