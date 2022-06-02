#define INTERVAL_MAX 10000
#define SERIAL_SPEED 115200
const unsigned int BUFFER_LENGTH = 80;

const int outPin = 15;
unsigned long intervalMS = 1000;
unsigned long previousMillis = 0;
char buffer[BUFFER_LENGTH];
int pinState = LOW;

void fireLaser() {
  unsigned long intervalUS = intervalMS * 1000;
  // Wait 1 second
  delay(500);
  digitalWrite(outPin, HIGH);
//  Serial.print("Activating pulse\n");
  delayMicroseconds(intervalUS);
  digitalWrite(outPin, LOW);
//  Serial.print("De-activating pulse\n");
}

void changeInterval(unsigned long value) {
  if (value <= INTERVAL_MAX) {
    intervalMS = value;
  }
}

void setup() {
  Serial.begin(SERIAL_SPEED); //, SERIAL_8N1);
  pinMode(outPin, OUTPUT);
  digitalWrite(outPin, LOW);
}

void loop() {
  String input;
  char rxChar;
  unsigned long inputExposureTime;
  //unsigned long currentMillis = millis();
  unsigned long intervalUS = intervalMS * 1000;

  if(Serial.available()) {
    input = Serial.readStringUntil(0x0D);
    rxChar = input[0];
    switch (rxChar) {
      case 0x69: // i as in id
        Serial.print("TRIGGER\n");
        break;
      case 0x66: // 'f': Fire the laser
          fireLaser();
        break;
      case 0x74: // 't': Change the exposure time
        if (input[1] == 0x3F) {
          sprintf(buffer, "%lu", intervalMS);
          Serial.print(buffer);
          Serial.print('\n');
          break;
        }
        inputExposureTime = (unsigned long) (unsigned int) input.substring(2).toInt();
        sprintf(buffer, "%lu", inputExposureTime);
        if (inputExposureTime > INTERVAL_MAX) {
          Serial.print("ERR_INTERVAL");
          Serial.print('\n');
        } else {
          changeInterval(inputExposureTime);
          Serial.print("PERIOD: ");
          Serial.print(buffer);
          Serial.print('\n');
        }
        break;
      default:
        Serial.print("ERR_CMD");
        Serial.print('\n');
    }
  }
}