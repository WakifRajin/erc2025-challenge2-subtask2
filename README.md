# Equipment Panel Challenge - Visual Communication System

## What We're Dealing With

The Equipment Panel challenge is basically a high-tech version of Morse code using LEDs. We need to build a system where an ESP32 controller talks to our robot through a 16×16 LED matrix, sending a 100-character password that we have to decode with our robot's camera. No pressure, but we only get one shot at this.

## The Setup We're Working With

### Hardware
- **Transmitter**: ESP32 microcontroller driving a 16×16 WS2812B LED matrix (16cm × 16cm)
- **Receiver**: Our robot's camera plus whatever computer vision magic we can code up
- **Environment**: Indoor tent with "controlled" lighting (we'll see about that)
- **LED Layout**: Z-pattern addressing because nothing can be simple

### What We Need to Transmit
- **Password**: 100 characters of A-Z and 0-9 only
- **Constraints**: One shot per boot, no do-overs
- **Timing**: They're literally timing how long our LEDs are on
- **Backup Plan**: There's a QR code somewhere if we completely fail

## The Protocol We Have to Follow

This is where things get strict. We have to follow this exact sequence or we get penalized:

```
1. Our ESP32 says "HELLO\n" to the control board
2. Control board responds "ACK\n"
3. We tell judges we're ready, they hit START
4. Our ESP32 says "READY\n"
5. Control board gives us "PASSWORD:<100-char-password>\n"
6. We pull up LED_ENABLE_REQUEST pin (timer starts)
7. We blast our password through the LED matrix
8. We pull down LED_ENABLE_REQUEST pin (timer stops)
```

**The Rules We Can't Break:**
- Read the password exactly once
- Turn on the LED matrix exactly once
- No cheating with external communication
- Follow the protocol to the letter

## Our Technical Approach

### How We're Encoding This
We're not falling for the 8-bit trap. Since we only have A-Z and 0-9 (36 characters), we can squeeze each character into 6 bits instead of 8. That's a 25% speed boost right there.

Our transmission plan:
```
┌─────────────────┬──────────────────┬─────────────────┬────────────┐
│ Startup Pattern │ Length Indicator │ Password Data   │ Checksum   │
│ (5 frames)      │ (3 frames)       │ (100 frames)    │ (2 frames) │
└─────────────────┴──────────────────┴─────────────────┴────────────┘
```

### The Real-World Problems We're Solving
- **Lighting**: "Controlled" lighting in a tent still means surprises
- **Camera Issues**: Our camera might decide to auto-expose at the worst moment
- **Diffuser Drama**: The LED diffuser might make only 25% of pixels usable
- **Viewing Angles**: Sometimes approaching at an angle works better than straight-on
- **Color Limits**: We can't just use a rainbow - the camera has limits

## How We're Getting Scored

### What They're Measuring
1. **Speed**: How fast we can transmit (shorter time = better score)
2. **Accuracy**: Did we get the password right? (100% or bust)
3. **Reliability**: Does our system work when conditions suck?
4. **Protocol Compliance**: Did we follow the rules exactly?

### The Scoring System
```
Base Score = 100 points
Time Bonus = max(0, 50 - transmission_time_seconds)
Accuracy Penalty = -50 points per wrong character
Protocol Violation = -25 points per screw-up
```

### Test Scenarios We're Preparing For
1. **Best Case**: Perfect lighting, clear diffuser, camera cooperates
2. **Worst Case**: Terrible diffuser, lighting changes mid-transmission
3. **Weird Cases**: Extreme angles, high ambient light, Murphy's Law
4. **Stress Test**: Back-to-back attempts when we're already stressed

## Our Development Strategy

### Transmitter Side (ESP32)
We're building in:
- Startup calibration patterns so our camera can adapt
- Smart brightness control that won't blind the camera
- Bulletproof error detection with CRC checksums
- Optimized encoding to minimize transmission time

### Receiver Side (Robot)
We're developing:
- OpenCV pipeline that can actually detect our LEDs
- Color calibration that adapts to conditions
- Frame synchronization that doesn't miss beats
- Error correction because stuff happens

### Our Testing Plan
1. **Build our own test rig**: We're not going in blind
2. **Calibration routine**: Test patterns to dial in our settings
3. **Stress testing**: Every lighting condition we can think of
4. **Protocol validation**: Make sure we don't get penalized for silly mistakes

## Why This Matters

This isn't just about blinking LEDs - it's about building a robust communication system under pressure. We're essentially creating a custom optical modem that has to work perfectly on the first try, in unknown conditions, with hardware we can't adjust once we submit our code.

The password we decode here unlocks the next part of the competition, so failure means we're stuck. No pressure at all.

Our approach balances speed with reliability because there's no point being the fastest team if we can't decode the password correctly. We're building redundancy into our system while keeping it simple enough that we can debug it when things go wrong.

Time to make some LEDs dance.
