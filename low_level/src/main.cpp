#include <Arduino.h>

#define MICROSTEP_MODE 4

const int dirPin = 8;
const int stepPin = 9;
const int stepsPerRevolution = 200 * MICROSTEP_MODE;  // Typically 200 steps for a 1.8° stepper motor

// Variables for angle control
float currentAngle = 0.0;
String inputString = "";
boolean stringComplete = false;

void processCommand();
void serialEvent();

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Initialize motor pins
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  
  // Reserve memory for input string
  inputString.reserve(32);
  
  // Send ready message
  Serial.println("Stepper motor controller ready!");
}

void loop() {
  // Process command when complete string is received
  if (stringComplete) {
    processCommand();
    inputString = "";
    stringComplete = false;
  }
}

// Process incoming serial data
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}

// Function to process commands
void processCommand() {
  // Command format: "ANGLE:123.45" - rotate to specific angle
  if (inputString.startsWith("ANGLE:")) {
    String angleStr = inputString.substring(6);
    float targetAngle = angleStr.toFloat();
    
    // Calculate the number of steps and direction
    float angleDiff = targetAngle - currentAngle;
    int steps = abs(angleDiff) * stepsPerRevolution / 360.0;
    
    // Set direction
    if (angleDiff > 0) {
      digitalWrite(dirPin, HIGH);  // Clockwise
    } else {
      digitalWrite(dirPin, LOW);   // Counter-clockwise
    }
    
    // Move motor
    for (int i = 0; i < steps; i++) {
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(800);
      digitalWrite(stepPin, LOW);
      delayMicroseconds(800);
    }
    
    // Update current angle
    currentAngle = targetAngle;
    
    // Send feedback
    Serial.print("DONE:");
    Serial.println(currentAngle);
  }
  else {
    Serial.println("ERROR:Invalid command format. Use ANGLE:xxx.xx");
  }
}