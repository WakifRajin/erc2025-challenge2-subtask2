# Equipment Panel Challenge - Visual Communication System

## Overview

The Equipment Panel challenge requires us to develop a one-way visual data transmission system for remote robotics competitions. We'll build both transmitter firmware (ESP32 + LED matrix) and receiver software (robot camera + computer vision) to reliably communicate 100-character passwords through optical signals.

## Challenge Specification

### Hardware Setup
- **Transmitter**: ESP32 microcontroller with 16×16 WS2812B LED matrix (16cm × 16cm)
- **Receiver**: Robot-mounted camera with computer vision processing
- **Environment**: Indoor tent with controlled lighting, matt black background
- **Addressing**: Z-pattern pixel arrangement (top row L→R, second row R→L, alternating)

### Data Requirements
- **Password Format**: 100 characters, uppercase letters (A-Z) and digits (0-9) only
- **Transmission**: Single attempt per boot cycle, no retransmission allowed
- **Timing**: Transmission duration measured and scored
- **Backup**: QR code/printed password available if optical transmission fails

### Communication Protocol

We must follow this strict handshake protocol managed by the judge-controlled board:

```
1. ESP32 → Control Board: "HELLO\n"
2. Control Board → ESP32: "ACK\n"
3. [Team reports readiness, judges press START]
4. ESP32 → Control Board: "READY\n"
5. Control Board → ESP32: "PASSWORD:<100-char-password>\n"
6. ESP32 raises LED_ENABLE_REQUEST pin (timer starts)
7. ESP32 transmits password via LED matrix
8. ESP32 lowers LED_ENABLE_REQUEST pin (timer stops)
```

**Critical Requirements:**
- Password read only once per boot
- LED matrix enabled only once per boot
- No external communication except LED matrix
- Strict protocol adherence required

## Our Technical Implementation

### Encoding Strategy
We're implementing a 6-bit per character encoding since A-Z,0-9 requires only 36 values (6 bits vs 8 bits). This provides a 25% reduction in transmission time while maintaining reliability.

Our transmission structure:
```
┌─────────────────┬──────────────────┬─────────────────┬────────────┐
│ Startup Pattern │ Length Indicator │ Password Data   │ Checksum   │
│ (5 frames)      │ (3 frames)       │ (100 frames)    │ (2 frames) │
└─────────────────┴──────────────────┴─────────────────┴────────────┘
```

### Real-World Challenges We're Addressing
- **Variable lighting**: Indoor tent with ambient light interference
- **Camera exposure**: Risk of over/under-exposure of LED signals
- **Diffuser effects**: May reduce usable pixels to ~25% of matrix
- **Viewing angles**: Optimal reception may require non-perpendicular approach
- **Dynamic range**: Limited distinguishable colors in camera system

## Measurement & Scoring

### Primary Metrics
1. **Transmission Time**: Duration LED matrix is enabled (measured by control board)
2. **Accuracy**: Correctness of decoded password (100% required for full points)
3. **Reliability**: Success rate across different lighting conditions
4. **Protocol Compliance**: Adherence to communication handshake

### Scoring Framework
```
Base Score = 100 points
Time Bonus = max(0, 50 - transmission_time_seconds)
Accuracy Penalty = -50 points per character error
Protocol Violation = -25 points per deviation
```

### Test Scenarios
1. **Optimal Conditions**: Clear diffuser, optimal lighting
2. **Challenging Conditions**: Heavy diffuser, variable lighting
3. **Edge Cases**: Extreme viewing angles, high ambient light
4. **Stress Test**: Multiple consecutive attempts with different passwords

## Our Development Approach

### Transmitter Side (ESP32)
We're implementing:
- Startup calibration patterns for camera adaptation
- Adaptive brightness control to prevent overexposure
- Robust error detection using CRC checksums
- Optimized encoding for minimal transmission time

### Receiver Side (Robot)
We're developing:
- OpenCV-based LED detection pipeline
- Color calibration routines for varying conditions
- Temporal synchronization for frame capture
- Error correction algorithms for data integrity

### Testing Strategy
1. **Build test matrix**: Replicate competition hardware conditions
2. **Calibration firmware**: Test patterns for optimal settings
3. **Stress testing**: Various lighting and diffuser configurations
4. **Protocol validation**: Ensure strict compliance with handshake requirements

## Competition Context

This challenge represents a modern adaptation of traditional equipment panel tasks, designed for remote competition environments. Our success depends on expertise in embedded systems programming, computer vision, optical communication protocols, and real-time signal processing.

We have provided multiple codes for you to test and verify. For example, each of the folder contains a separate version of the firmware and the corresponding code for the receiver side to detect the password. Each version is tested and verified at different conditions from our side. 

The visual communication system serves as a critical component in the larger robotics challenge, where the decoded password enables access to subsequent tasks. We must balance transmission speed with reliability, as retry attempts require significant time penalties due to course navigation requirements.

Our approach prioritizes building a robust system that can handle real-world conditions while maintaining the speed necessary for competitive scoring. We're focusing on redundancy and error detection to ensure successful transmission on the first attempt.
