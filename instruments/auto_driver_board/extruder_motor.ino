/* Arduino sketch for the ARPA-E extruder motor using a SparkFun AutoDriver
*  Board with the ST Micro L6470 stepper driver
*  Christopher Jones
*  cjjones@ucsd.edu
*  9/8/2021
*/

//#include <Adafruit_LiquidCrystal.h>
#include <SparkFunAutoDriver.h>
#include <SPI.h>

#define MIN_SPEED 0
#define MAX_SPEED 700
#define FULL_SPEED 1400
#define ACCELERATION 10000
#define DECELERATION 10000
#define SERIAL_SPEED 57600
#define SERIAL_TIMEOUT 100

AutoDriver translator(0, 10, 7, 8); //(motor position, CS, reset, busy)
int status;
unsigned int runspeed = 700;

void dSPINConfig(void)
{
  // Configure board
  translator.SPIPortConnect(&SPI);
  translator.configSyncPin(BUSY_PIN, 0x00); // busy pin could be repurposed
  translator.configStepMode(STEP_FS_128); // STEP_FS_2 to STEP_FS_128
  translator.setMinSpeed(MIN_SPEED); // steps per second
  translator.setMaxSpeed(MAX_SPEED); // steps per second
  translator.setFullSpeed(FULL_SPEED); // only matters at high speeds
  translator.setAcc(ACCELERATION); // steps per second per second
  translator.setDec(DECELERATION); // steps per second per second
  translator.setOCThreshold(OC_6000mA); // 6A is the highest setting
  translator.setPWMFreq(PWM_DIV_2, PWM_MUL_2); // 31.25kHz PWM freq
  translator.setSlewRate(SR_530V_us); // 180 and 290 are the other options
  translator.setOCShutdown(OC_SD_DISABLE); // don't shutdown on OC
  translator.setVoltageComp(VS_COMP_ENABLE); // compensate for motor V
  translator.setSwitchMode(SW_HARD_STOP); // CONFIG_SW_USER is other option
  

  translator.setRunKVAL(255); // sets max coil voltage, 255 = Vin
  translator.setAccKVAL(64); // three quarterts voltage acceleration
  translator.setDecKVAL(64); // three quarterts voltage deceleration
  translator.setHoldKVAL(12); // one twentieth holding voltage (PWM on coil 1)


  // Reset ABS_POS which counts steps from start or reset. Use for HOME
  //translator.resetPos();

  translator.softHiZ(); // start with no holding current
}

void setup()
{
  Serial.setTimeout(SERIAL_TIMEOUT); // allow 5 seconds for serial input
  Serial.begin(SERIAL_SPEED, SERIAL_8N1);
  
  // SPI and digi pins
  pinMode(7, OUTPUT);    //reset
  pinMode(8, INPUT);     //busy
  pinMode(10, OUTPUT);   // chip select
  pinMode(MOSI, OUTPUT); // Pin 11
  pinMode(MISO, INPUT);  // Pin 12
  pinMode(13, OUTPUT);   // spi clock

  // set chip select
  digitalWrite(10, HIGH);
  // reset the L6470
  digitalWrite(7, LOW);
  digitalWrite(7, HIGH);

  SPI.begin();
  SPI.setDataMode(SPI_MODE3);
  
  //Serial.println("\nStatus before board config");
  //status = translator.getStatus();
  //Serial.print("Status: ");
  //Serial.print(status, HEX);
  //Serial.print('\n');

  dSPINConfig();

  //Serial.println("\nSetup Complete");
  
  //Serial.println("\nStatus after board config");
  //status = translator.getStatus();
  //Serial.print("Status: ");
  //Serial.print(status, HEX);
  //Serial.print('\n');
}

void clearMessages(void)
{
  translator.getStatus();
  translator.getStatus();
}

void loop()
{
  char rxChar = 0;
  char dir;
  long curPos;
  String input;
  long moveSteps;
  long input_speed;

  if(Serial.available())
  {
    input = Serial.readStringUntil(0x0D);
    rxChar = input[0]; //Serial.read();
    switch(rxChar)
    {
      case 0x66: // 'f'
        translator.run(FWD, runspeed); // forward steps/s
        Serial.print("Forward: ");
        Serial.print(runspeed);
        Serial.print('\n');
        break;
      case 0x72: // 'r'
        translator.run(REV, runspeed); // reverse steps/s
        Serial.print("Reverse: ");
        Serial.print(runspeed);
        Serial.print('\n');
        break;
      case 0x20: // 'space'
        translator.softHiZ(); // stop w/ no holding current
        break;
/*      case 0x68: // 'h'
        translator.goHome(); // position to 0
        break; */
      case 0x76: // 'v'
        input_speed = input.substring(2).toInt();
        if (input_speed > 0 && input_speed < 1400) {
          runspeed = input_speed;
        }
        Serial.print(runspeed);
        Serial.print('\n');
        break;
      case 0x6D: // 'm' input distance and direction then execute
        moveSteps = input.substring(2).toInt();
        if(moveSteps != 0)
        {
          if(moveSteps < 0)
            dir = REV;
          else
            dir = FWD;
          moveSteps = abs(moveSteps) * 128; // correction for microsteps
          Serial.print("Moving ");
          if(dir)
            Serial.print("in ");
          else
            Serial.print("out ");
          Serial.print(moveSteps/128, DEC);
          Serial.print(" steps\n");
          translator.move(dir, moveSteps);
        }
        break;
      case 0x73: // 's' the status of the system
        status = translator.getStatus();
        Serial.print(status, HEX);
        Serial.print('\n');
        break;
      case 0x70: // 'p' The position of the system
        curPos = translator.getPos();
        Serial.print(curPos);
        Serial.print('\n');
        break;
      default:
        Serial.print("Command '");
        Serial.print(rxChar);
        Serial.print("' not found!\n");
    }
    /* curPos = translator.getPos(); */
    /*Serial.print("Position: ");
    Serial.print(curPos);*/
    /* clearMessages(); */
    /*Serial.print('\n');
    Serial.print("Status: ");
    Serial.print(status, HEX);
    Serial.print('\n');*/
  }
}
