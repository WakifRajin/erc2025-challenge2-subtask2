#include <FastLED.h>

#define NUM_LEDS 256
#define DATA_PIN 4
#define LED_ENABLE_REQUEST_PIN 5
#define UART_BAUD_RATE 115200

CRGB leds[NUM_LEDS];
CRGB calib1_pattern[NUM_LEDS];
CRGB calib2_pattern[NUM_LEDS];
CRGB frame0_pattern[NUM_LEDS];
CRGB frame1_pattern[NUM_LEDS];
CRGB frame2_pattern[NUM_LEDS];

bool bitArray[616];
bool transmitting = false;
unsigned long transmissionStartTime;
int transmissionState = 0;

void showPattern(CRGB *pattern) {
    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = pattern[i];
    }
    FastLED.show();
}

void setAllBlack() {
    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = CRGB::Black;
    }
}

uint16_t computeCRC16CCITT(uint8_t *data, int length) {
    uint16_t crc = 0xFFFF;
    for (int i = 0; i < length; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc = crc << 1;
            }
        }
    }
    return crc & 0xFFFF;
}

void precomputePatterns(String password) {
    for (int i = 0; i < 616; i++) {
        bitArray[i] = 0;
    }

    int bitIndex = 0;
    for (int i = 0; i < 100; i++) {
        char c = password[i];
        int value;
        if (c >= '0' && c <= '9') {
            value = c - '0';
        } else if (c >= 'A' && c <= 'Z') {
            value = 10 + (c - 'A');
        } else {
            value = 0;
        }
        for (int j = 5; j >= 0; j--) {
            if (bitIndex < 600) {
                bitArray[bitIndex] = (value >> j) & 1;
                bitIndex++;
            }
        }
    }

    uint8_t dataBytes[75];
    for (int i = 0; i < 75; i++) {
        uint8_t b = 0;
        for (int j = 0; j < 8; j++) {
            b = (b << 1) | bitArray[i * 8 + j];
        }
        dataBytes[i] = b;
    }

    uint16_t crc = computeCRC16CCITT(dataBytes, 75);
    for (int j = 15; j >= 0; j--) {
        bitArray[600 + (15 - j)] = (crc >> j) & 1;
    }

    for (int i = 0; i < 256; i++) {
        frame0_pattern[i] = bitArray[i] ? CRGB::White : CRGB::Black;
    }
    for (int i = 0; i < 256; i++) {
        frame1_pattern[i] = bitArray[256 + i] ? CRGB::White : CRGB::Black;
    }
    for (int i = 0; i < 256; i++) {
        if (i < 104) {
            frame2_pattern[i] = bitArray[512 + i] ? CRGB::White : CRGB::Black;
        } else {
            frame2_pattern[i] = CRGB::Black;
        }
    }

    for (int i = 0; i < NUM_LEDS; i++) {
        calib1_pattern[i] = CRGB::White;
    }

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            int index = i * 16 + j;
            int visual_col = (i % 2 == 0) ? j : (15 - j);
            if ((i + visual_col) % 2 == 0) {
                calib2_pattern[index] = CRGB::White;
            } else {
                calib2_pattern[index] = CRGB::Black;
            }
        }
    }
}

void setup() {
    Serial.begin(UART_BAUD_RATE);
    FastLED.addLeds<WS2812B, DATA_PIN, GRB>(leds, NUM_LEDS);
    pinMode(LED_ENABLE_REQUEST_PIN, OUTPUT);
    digitalWrite(LED_ENABLE_REQUEST_PIN, LOW);
    setAllBlack();
    FastLED.show();

    Serial.println("HELLO");
    while (true) {
        if (Serial.available()) {
            String response = Serial.readStringUntil('\n');
            response.trim();
            if (response == "ACK") {
                break;
            }
        }
    }

    Serial.println("READY");
    String passwordLine = "";
    while (true) {
        if (Serial.available()) {
            char c = Serial.read();
            if (c == '\n') {
                break;
            }
            passwordLine += c;
        }
    }

    if (passwordLine.startsWith("PASSWORD:")) {
        String password = passwordLine.substring(9);
        precomputePatterns(password);
    }

    digitalWrite(LED_ENABLE_REQUEST_PIN, HIGH);
    transmitting = true;
    transmissionStartTime = millis();
    transmissionState = 0;
    showPattern(calib1_pattern);
}

void loop() {
    if (transmitting) {
        unsigned long now = millis();
        unsigned long elapsed = now - transmissionStartTime;

        if (elapsed < 1000) {
            if (transmissionState != 0) {
                transmissionState = 0;
            }
        } else if (elapsed < 2000) {
            if (transmissionState != 1) {
                showPattern(calib2_pattern);
                transmissionState = 1;
            }
        } else if (elapsed < 2500) {
            if (transmissionState != 2) {
                showPattern(frame0_pattern);
                transmissionState = 2;
            }
        } else if (elapsed < 3000) {
            if (transmissionState != 3) {
                showPattern(frame1_pattern);
                transmissionState = 3;
            }
        } else if (elapsed < 3500) {
            if (transmissionState != 4) {
                showPattern(frame2_pattern);
                transmissionState = 4;
            }
        } else {
            setAllBlack();
            FastLED.show();
            digitalWrite(LED_ENABLE_REQUEST_PIN, LOW);
            transmitting = false;
        }
    }
}
