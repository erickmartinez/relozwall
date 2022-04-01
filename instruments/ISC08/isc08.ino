/* Arduino sketch for the ARPA-E extruder motor
*  Controller is ISC08 from stepperonline
*  9/30/2021
*  Update 4/1/2022
*   Added STOP command and limit switch logic
*/


/* Commands
*    Read manual.txt!!
*/

#define SERIAL_SPEED 57600
#define SERIAL_TIMEOUT 5000
// Speed presets, speed can be set from 0 - 100
#define Q_SPEED 60
#define Q_TIME 50

// Names for IO pins
#define ENABLE 4
#define SPD 5  // speed control (PWM needs LP filter)
#define HILO 6 // high and low speed
#define DIRECTION 7
#define ILS 2  //inner limit switch
#define OLS 3  //outer limit switch

// Flags
volatile bool ilsFlag = true;
volatile bool olsFlag = true;

void setup()
{
  Serial.setTimeout(SERIAL_TIMEOUT);
  Serial.begin(SERIAL_SPEED, SERIAL_8N1);
  
  // digital pins
  pinMode(ENABLE, OUTPUT);       // HIGH is RUN / LOW is free
  pinMode(SPD, OUTPUT);   
  pinMode(HILO, OUTPUT);   
  pinMode(DIRECTION, OUTPUT); // HIGH is IN
  pinMode(ILS, INPUT_PULLUP);
  pinMode(OLS, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ILS), ilsFun, RISING);
  attachInterrupt(digitalPinToInterrupt(OLS), olsFun, RISING);

  // Make sure the motor is disabled on startup
  digitalWrite(ENABLE, LOW);
}

/* Interrupt Service Routines */
void ilsFun()
{
  digitalWrite(ENABLE, LOW); // stop motor
  ilsFlag = true;
}
void olsFun()
{
  digitalWrite(ENABLE, LOW); // stop motor
  olsFlag = true;
}

void run(unsigned char s, unsigned long t)
{
  if(s < 50)
  {
    digitalWrite(HILO, LOW);
  }
  else
  {
    digitalWrite(HILO, HIGH);
    s = s % 50;
  }
  
  /* The ISC08 speed is set with a voltage between 0.5 and 4.5VDC
   * The Arduino's analog out is a PWM that can be set from 0 to 255
   * With the Low Pass filter on the analog out pin 0.5 to 4.5 was
   *  achieved with values from 25 to 235 */
  s = 4.286 * s + 25;
  t = t*100;

  analogWrite(SPD, s); // set PWM on analog out 5
  delay(50); // allow time for voltage to settle

  unsigned long start = millis();
  digitalWrite(ENABLE, HIGH);
  while((millis() - start) < t)
  {
    if(Serial.available())
      break;
  }

  digitalWrite(ENABLE, LOW);
  // Leave motor inputs at a low setting
  digitalWrite(HILO, LOW);
  digitalWrite(SPD, 0);
}

void loop()
{
  // reset limit switches
  if(!digitalRead(ILS))
    ilsFlag = false;
  if(!digitalRead(OLS))
    olsFlag = false;

  if(Serial.available())
  {
    /* readStringUntil will take the entire string up to but excluding
     * the designated characther. 0x0D is <CR> and is sent by the RETURN
     * or ENTER key of many terminal programs */

    String input = Serial.readStringUntil(0x0D);
    char dir = input[0];
    unsigned int speed = input.substring(1,3).toInt();
    unsigned long run_time = input.substring(3).toInt();

    switch(dir)
    {
      case 0x69: // i as in id
        Serial.print("TRANSLATOR\n");
        break;
      case 0x66: // 'f' forward or in
        if(ilsFlag){
          Serial.println("ERROR_MOVE_IN");
          break;
        }
        digitalWrite(DIRECTION, HIGH);
        run(speed, run_time);
        break;
      case 0x72: // 'r' reverse or out
        if(olsFlag){
          Serial.println("ERROR_MOVE_OUT");
          break;
        }
        digitalWrite(DIRECTION, LOW);
        run(speed, run_time);
        break;
      case 0x71: // 'q' quick out
        digitalWrite(DIRECTION, LOW);
        run(Q_SPEED, Q_TIME);
        break;
      case 0x73: // 's' STOP
        // We don't need to do anything here, just skip default
        break;
      default:
        Serial.println("ERROR_CMD");
        break;
    }
  }
}
