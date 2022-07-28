int posPin = A0;

// store read results for sending to the LCD and data storage
unsigned int adcVal;
unsigned long currentMillis, previousMillis;

String input;
char rxChar;

unsigned int outputBufferSize = sizeof(adcVal);

void setup() {
  Serial.begin(19200);
  previousMillis = 0;
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
      case 0x72: // r as in read
        byte *d = (byte *) &adcVal;
        Serial.write(d, outputBufferSize);
        break;
      default:
        Serial.print("ERR_CMD\n");
    }
  }
}