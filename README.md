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

## **We have provided multiple codes for you to test and verify. For example, each of the folder contains a separate version of the firmware and the corresponding code for the receiver side to detect the password. Each version is tested and verified at different conditions from our side.**

The visual communication system serves as a critical component in the larger robotics challenge, where the decoded password enables access to subsequent tasks. We must balance transmission speed with reliability, as retry attempts require significant time penalties due to course navigation requirements.

Our approach prioritizes building a robust system that can handle real-world conditions while maintaining the speed necessary for competitive scoring. We're focusing on redundancy and error detection to ensure successful transmission on the first attempt.

## Hardware Requirements

### Camera System
- USB webcam or integrated camera (minimum 720p recommended)
- Stable mounting system for consistent viewing angle
- Adequate lighting (avoid direct sunlight or harsh shadows)

### LED Matrix Setup
- 16x16 WS2812B LED matrix (16cm x 16cm)
- Arduino/ESP32 with transmitter firmware
- 4 corner markers for matrix boundary detection
- Matt black background recommended

### Positioning
- Camera distance: 30-100cm from matrix
- Matrix should occupy 20-60% of camera frame
- Avoid reflective surfaces and moving objects in background

## Software Requirements

- Python 3.7 or higher
- OpenCV 4.0 or higher
- NumPy 1.19 or higher
- A computer with sufficient processing power for real-time video processing

## Installation

### 1. Clone the Repository
```bash
mkdir led-matrix-receiver
cd led-matrix-receiver
git clone https://github.com/WakifRajin/erc2025-challenge2-subtask2.git
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python numpy
```

### 4. Verify Installation
```bash
python -c "import cv2, numpy; print('Installation successful!')"
```

## Quick Start

### 1. Basic Usage
```python
from led_matrix_receiver import LEDMatrixReceiver

# Create receiver with debug enabled
receiver = LEDMatrixReceiver(debug=True)

# Start capture (will run until transmission complete or timeout)
result = receiver.capture_and_decode(video_source=0, duration=60)

# Check result
if result:
    print(f"Password: {result['password']}")
    print(f"Valid: {result['checksum_valid']}")
else:
    print("No transmission received")
```

### 2. Command Line Usage
```bash
# Run with default settings
python led_matrix_receiver.py

# Run with specific camera
python led_matrix_receiver.py --camera 1

# Run with custom timeout
python led_matrix_receiver.py --timeout 120
```

## Configuration

### Camera Settings
```python
receiver = LEDMatrixReceiver(
    matrix_size=(16, 16),           # Matrix dimensions
    debug=True,                     # Enable debug visualization
    brightness_threshold=100,       # LED detection threshold
    color_similarity_threshold=30,  # Color classification tolerance
    frame_duration=0.2             # Expected frame duration (seconds)
)
```

### Detection Parameters
```python
# Adjust these based on your setup
receiver.min_contour_area = 100      # Minimum LED region size
receiver.max_contour_area = 50000    # Maximum LED region size
receiver.brightness_threshold = 100   # Brightness threshold for LED detection
```

## Usage

### Basic Operation

1. **Setup Hardware**
   - Position camera to view LED matrix
   - Ensure good lighting conditions
   - Start Arduino transmitter

2. **Run Receiver**
   ```bash
   python led_matrix_receiver.py
   ```

3. **Monitor Output**
   - Debug window shows detection process
   - Console displays transmission progress
   - Result saved to `received_password.json`

### Advanced Usage

#### Custom Video Source
```python
# Use specific camera
receiver.capture_and_decode(video_source=1)

# Use video file for testing
receiver.capture_and_decode(video_source='test_video.mp4')
```

#### Batch Processing
```python
# Process multiple transmissions
results = []
for i in range(3):
    print(f"Capture {i+1}/3 - Press Enter when ready...")
    input()
    result = receiver.capture_and_decode(duration=30)
    results.append(result)
```

#### Integration with Robot Control
```python
import rospy
from your_robot_msgs.msg import PasswordMsg

def robot_integration():
    receiver = LEDMatrixReceiver(debug=False)
    
    # Position robot in front of matrix
    # ... robot positioning code ...
    
    # Capture password
    result = receiver.capture_and_decode(duration=45)
    
    if result and result['checksum_valid']:
        # Send password to robot controller
        pub = rospy.Publisher('/password', PasswordMsg, queue_size=1)
        msg = PasswordMsg()
        msg.password = result['password']
        pub.publish(msg)
        return True
    
    return False
```

## Protocol Details

### Transmission Sequence
1. **Startup Patterns** (5 frames)
   - Full white (brightness calibration)
   - Checkerboard (pixel differentiation)
   - Corner markers (alignment)
   - Quadrant colors (segment identification)
   - Sync pattern (timing reference)

2. **Length Indicator** (3 frames)
   - Binary representation of password length
   - Cyan color for '1' bits, black for '0' bits

3. **Data Transmission** (100 frames for 100-char password)
   - Each character encoded as 6-bit value
   - Quadrant encoding: TL(2-bit), TR(2-bit), BL(1-bit), BR(1-bit)
   - Colors: Black=0, Red=1, Green=2, Blue=3, White=1, Black=0

4. **Checksum** (1 frame)
   - CRC-8 checksum displayed as 8-bit binary pattern
   - Yellow for '1' bits, black for '0' bits

5. **End Marker** (1 frame)
   - Solid purple indicating transmission complete

### Character Encoding
- A-Z: 0-25 (6-bit values)
- 0-9: 26-35 (6-bit values)
- Invalid characters default to 'A' (value 0)

### Error Detection
- CRC-8 with polynomial 0x07
- Matches Arduino implementation exactly
- Transmission marked invalid if checksum fails

## Troubleshooting

### Common Issues

#### "No matrix detected"
- **Cause**: Camera can't find LED matrix boundaries
- **Solution**: 
  - Ensure corner markers are visible and bright
  - Check camera focus and lighting
  - Adjust `brightness_threshold` parameter
  - Move camera closer or farther from matrix

#### "Checksum validation failed"
- **Cause**: Transmission errors or decoding issues
- **Solution**:
  - Check for interference or movement during transmission
  - Verify lighting is consistent
  - Ensure camera is stable
  - Try reducing ambient light

#### "Timeout reached"
- **Cause**: No transmission detected within time limit
- **Solution**:
  - Verify Arduino transmitter is running
  - Check serial communication on Arduino
  - Ensure matrix is powered and functioning
  - Increase timeout duration

#### Poor color detection
- **Cause**: Lighting conditions or camera settings
- **Solution**:
  - Adjust `color_similarity_threshold`
  - Use diffused lighting
  - Avoid direct sunlight
  - Calibrate camera exposure settings

### Debug Mode

Enable debug mode for detailed troubleshooting:
```python
receiver = LEDMatrixReceiver(debug=True)
```

Debug features:
- Real-time video window with detection overlay
- Frame-by-frame pattern analysis
- Console logging of detection process
- Visual feedback for color classification

### Performance Optimization

#### For Slower Systems
```python
# Reduce processing load
receiver = LEDMatrixReceiver(
    matrix_size=(16, 16),
    debug=False,  # Disable visualization
    brightness_threshold=120,  # Higher threshold
)

# Use lower camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

#### For Better Accuracy
```python
# Increase processing quality
receiver = LEDMatrixReceiver(
    brightness_threshold=80,        # Lower threshold
    color_similarity_threshold=20,  # Stricter color matching
    min_contour_area=50,           # Smaller minimum area
)

# Use higher camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
```

