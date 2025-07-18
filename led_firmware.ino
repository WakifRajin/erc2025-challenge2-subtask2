#include <FastLED.h>

// Hardware Configuration
#define LED_DATA_PIN 2
#define LED_ENABLE_REQUEST_PIN 4
#define MATRIX_WIDTH 16
#define MATRIX_HEIGHT 16
#define NUM_LEDS (MATRIX_WIDTH * MATRIX_HEIGHT)

// Communication Configuration
#define SERIAL_BAUD 115200
#define FRAME_DURATION_MS 200  // Duration each frame is displayed
#define STARTUP_PATTERN_FRAMES 5  // Number of startup calibration frames

// Data Encoding Configuration
#define BITS_PER_CHAR 6  // A-Z,0-9 = 36 chars, need 6 bits
#define CHECKSUM_POLYNOMIAL 0x07  // CRC-8 polynomial

// LED Matrix
CRGB leds[NUM_LEDS];
String receivedPassword = "";
bool passwordReceived = false;
bool transmissionComplete = false;

// CRC-8 Checksum calculation
uint8_t calculateCRC8(const String& data) {
  uint8_t crc = 0;
  for (int i = 0; i < data.length(); i++) {
    crc ^= data[i];
    for (int j = 0; j < 8; j++) {
      if (crc & 0x80) {
        crc = (crc << 1) ^ CHECKSUM_POLYNOMIAL;
      } else {
        crc <<= 1;
      }
    }
  }
  return crc;
}

// Convert character to 6-bit value (A-Z = 0-25, 0-9 = 26-35)
uint8_t charTo6Bit(char c) {
  if (c >= 'A' && c <= 'Z') {
    return c - 'A';
  } else if (c >= '0' && c <= '9') {
    return c - '0' + 26;
  }
  return 0; // Invalid character defaults to 'A'
}

// Convert Z-pattern position to LED index
int getZPatternIndex(int x, int y) {
  if (y % 2 == 0) {
    // Even rows: left to right
    return y * MATRIX_WIDTH + x;
  } else {
    // Odd rows: right to left
    return y * MATRIX_WIDTH + (MATRIX_WIDTH - 1 - x);
  }
}

// Set pixel color in Z-pattern addressing
void setPixelZ(int x, int y, CRGB color) {
  if (x >= 0 && x < MATRIX_WIDTH && y >= 0 && y < MATRIX_HEIGHT) {
    int index = getZPatternIndex(x, y);
    leds[index] = color;
  }
}

// Clear all LEDs
void clearMatrix() {
  FastLED.clear();
  FastLED.show();
}

// Display startup calibration pattern
void displayStartupPattern(int frameNum) {
  clearMatrix();
  
  switch (frameNum) {
    case 0:
      // Full white - brightness calibration
      fill_solid(leds, NUM_LEDS, CRGB::White);
      break;
      
    case 1:
      // Checkerboard pattern - pixel differentiation
      for (int y = 0; y < MATRIX_HEIGHT; y++) {
        for (int x = 0; x < MATRIX_WIDTH; x++) {
          CRGB color = ((x + y) % 2 == 0) ? CRGB::White : CRGB::Black;
          setPixelZ(x, y, color);
        }
      }
      break;
      
    case 2:
      // Corner markers - alignment check
      setPixelZ(0, 0, CRGB::Red);
      setPixelZ(MATRIX_WIDTH-1, 0, CRGB::Green);
      setPixelZ(0, MATRIX_HEIGHT-1, CRGB::Blue);
      setPixelZ(MATRIX_WIDTH-1, MATRIX_HEIGHT-1, CRGB::Yellow);
      break;
      
    case 3:
      // Quadrant test - segment identification
      for (int y = 0; y < MATRIX_HEIGHT/2; y++) {
        for (int x = 0; x < MATRIX_WIDTH/2; x++) {
          setPixelZ(x, y, CRGB::Red);  // Top-left
          setPixelZ(x + MATRIX_WIDTH/2, y, CRGB::Green);  // Top-right
          setPixelZ(x, y + MATRIX_HEIGHT/2, CRGB::Blue);  // Bottom-left
          setPixelZ(x + MATRIX_WIDTH/2, y + MATRIX_HEIGHT/2, CRGB::Yellow);  // Bottom-right
        }
      }
      break;
      
    case 4:
      // Sync pattern - timing reference
      for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = (i % 4 == 0) ? CRGB::White : CRGB::Black;
      }
      break;
  }
  
  FastLED.show();
}

// Display 6-bit value using quadrant encoding
void display6BitValue(uint8_t value) {
  clearMatrix();
  
  // Split 6 bits into 4 quadrants (2 bits each for top quadrants, 1 bit each for bottom)
  // Top-left: bits 5,4    Top-right: bits 3,2    Bottom-left: bit 1    Bottom-right: bit 0
  
  uint8_t topLeft = (value >> 4) & 0x03;    // Bits 5,4 (values 0-3)
  uint8_t topRight = (value >> 2) & 0x03;   // Bits 3,2 (values 0-3)
  uint8_t bottomLeft = (value >> 1) & 0x01; // Bit 1 (values 0-1)
  uint8_t bottomRight = value & 0x01;       // Bit 0 (values 0-1)
  
  // Set quadrant colors based on bit values
  CRGB colors[4] = {CRGB::Black, CRGB::Red, CRGB::Green, CRGB::Blue};
  
  // Fill quadrants
  for (int y = 0; y < MATRIX_HEIGHT/2; y++) {
    for (int x = 0; x < MATRIX_WIDTH/2; x++) {
      setPixelZ(x, y, colors[topLeft]);
      setPixelZ(x + MATRIX_WIDTH/2, y, colors[topRight]);
      setPixelZ(x, y + MATRIX_HEIGHT/2, bottomLeft ? CRGB::White : CRGB::Black);
      setPixelZ(x + MATRIX_WIDTH/2, y + MATRIX_HEIGHT/2, bottomRight ? CRGB::White : CRGB::Black);
    }
  }
  
  FastLED.show();
}

// Display checksum using simple binary pattern
void displayChecksum(uint8_t checksum) {
  clearMatrix();
  
  // Display checksum as 8-bit binary pattern across the matrix
  for (int i = 0; i < 8; i++) {
    bool bitValue = (checksum >> (7 - i)) & 0x01;
    CRGB color = bitValue ? CRGB::Yellow : CRGB::Black;
    
    // Fill columns based on bit values
    int startCol = i * 2;
    for (int y = 0; y < MATRIX_HEIGHT; y++) {
      for (int x = startCol; x < startCol + 2 && x < MATRIX_WIDTH; x++) {
        setPixelZ(x, y, color);
      }
    }
  }
  
  FastLED.show();
}

// Main transmission function
void transmitPassword() {
  if (!passwordReceived || transmissionComplete) {
    return;
  }
  
  // Enable LED matrix
  digitalWrite(LED_ENABLE_REQUEST_PIN, HIGH);
  delay(100); // Small delay to ensure pin state is registered
  
  // Display startup calibration patterns
  for (int i = 0; i < STARTUP_PATTERN_FRAMES; i++) {
    displayStartupPattern(i);
    delay(FRAME_DURATION_MS);
  }
  
  // Calculate checksum for the password
  uint8_t checksum = calculateCRC8(receivedPassword);
  
  // Transmit password length indicator (frames 1-3 = length in binary)
  uint8_t passwordLength = receivedPassword.length();
  for (int i = 0; i < 3; i++) {
    bool bitValue = (passwordLength >> (6 - i)) & 0x01;
    clearMatrix();
    fill_solid(leds, NUM_LEDS, bitValue ? CRGB::Cyan : CRGB::Black);
    FastLED.show();
    delay(FRAME_DURATION_MS);
  }
  
  // Transmit each character
  for (int i = 0; i < receivedPassword.length(); i++) {
    uint8_t charValue = charTo6Bit(receivedPassword[i]);
    display6BitValue(charValue);
    delay(FRAME_DURATION_MS);
  }
  
  // Transmit checksum
  displayChecksum(checksum);
  delay(FRAME_DURATION_MS * 2); // Display checksum longer
  
  // End transmission pattern
  clearMatrix();
  fill_solid(leds, NUM_LEDS, CRGB::Purple);
  FastLED.show();
  delay(FRAME_DURATION_MS);
  
  // Clear matrix and disable
  clearMatrix();
  digitalWrite(LED_ENABLE_REQUEST_PIN, LOW);
  
  transmissionComplete = true;
}

// Process serial communication
void processSerialData() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input.startsWith("ACK")) {
      // Control board acknowledged HELLO
      Serial.println("READY");
    } else if (input.startsWith("PASSWORD:")) {
      // Extract password from the message
      receivedPassword = input.substring(9); // Remove "PASSWORD:" prefix
      passwordReceived = true;
      
      // Pre-compute any needed data here
      // (Optional preprocessing can be done here)
      
      // Start transmission immediately
      transmitPassword();
    }
  }
}

void setup() {
  // Initialize serial communication
  Serial.begin(SERIAL_BAUD);
  
  // Initialize LED matrix
  FastLED.addLeds<WS2812B, LED_DATA_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(128); // 50% brightness to avoid overexposure
  FastLED.setMaxPowerInVoltsAndMilliamps(5, 2000); // Power limiting
  
  // Initialize control pins
  pinMode(LED_ENABLE_REQUEST_PIN, OUTPUT);
  digitalWrite(LED_ENABLE_REQUEST_PIN, LOW);
  
  // Clear matrix initially
  clearMatrix();
  
  // Wait for system to stabilize
  delay(1000);
  
  // Send HELLO to control board
  Serial.println("HELLO");
}

void loop() {
  // Process incoming serial data
  processSerialData();
  
  // Small delay to prevent overwhelming the system
  delay(10);
  
  // If transmission is complete, we can enter a low-power state
  if (transmissionComplete) {
    delay(1000); // Keep system responsive but reduce activity
  }
}
