/* Arduino sketch for the ARPA-E extruder motor
*  Controller is ISC08 from stepperonline
*  9/30/2021
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
  attachInterrupt(digitalPinToInterrupt(ILS), limit_switch, LOW);
  attachInterrupt(digitalPinToInterrupt(OLS), limit_switch, LOW);

  // Make sure the motor is disabled on startup
  digitalWrite(ENABLE, LOW);
}

/* Interrupt Service Routine */
void limit_switch()
{
  digitalWrite(ENABLE, LOW);
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
  digitalWrite(ENABLE, HIGH);
  delay(t);
  digitalWrite(ENABLE, LOW);

  // Leave motor inputs at a low setting
  digitalWrite(HILO, LOW);
  digitalWrite(SPD, 0);
}

void loop()
{
  String input;
  char dir;
  char speed;
  unsigned long run_time;

  if(Serial.available())
  {
    /* readStringUntil will take the entire string up to but excluding
     * the designated characther. 0x0D is <CR> and is sent by the RETURN
     * or ENTER key of many terminal programs */
    input = Serial.readStringUntil(0x0D);
    dir = input[0];
    speed = input.substring(1,3).toInt();
    run_time = input.substring(3).toInt();

    switch(dir)
    {
      case 0x66: // 'f' forward or in
        digitalWrite(DIRECTION, HIGH);
        run(speed, run_time);
        break;
      case 0x72: // 'r' reverse or out
        digitalWrite(DIRECTION, LOW);
        run(speed, run_time);
        break;
      case 0x71: // 'q' quick out
        digitalWrite(DIRECTION, LOW);
        run(Q_SPEED, Q_TIME);
        break;
      default:
        Serial.println("Input Not Valid");
        break;
    }
  }
}
