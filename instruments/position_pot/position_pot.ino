int posPin = A0;

unsigned int adcVal;

String input;
char rxChar;

unsigned int outputBufferSize = sizeof(adcVal);

void setup() {
  Serial.begin(115200, SERIAL_8N1 );
  // previousMillis = 0;
}

void loop() {
  if(Serial.available()) {
    input = Serial.readStringUntil(0x0D); // Read until line breaks
    rxChar = input[0];
    switch (rxChar) {
      case 0x69: // i as in id
        Serial.print("DEFLECTION_POT\n");
        Serial.flush();
        break;
      case 0x72: // r as in read
        adcVal = analogRead(posPin);
        //adcVal = (adcVal > 0) ? adcVal : 0;
        byte *d = (byte *) &adcVal;
        Serial.write(d, outputBufferSize);
        Serial.flush();
        // Serial.write((uint8_t *) &adcVal, outputBufferSize);
        break;
      default:
        Serial.print("ERR_CMD\n");
        Serial.flush();
    }
  }
}
