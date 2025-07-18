import cv2
import numpy as np
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """Configuration for LED matrix detection"""
    matrix_size: int = 16
    expected_frames: int = 5  # calib1, calib2, frame0, frame1, frame2
    frame_duration: float = 0.5  # seconds
    brightness_threshold: int = 100
    min_contour_area: int = 50
    max_contour_area: int = 5000
    stabilization_frames: int = 3

class FrameBuffer:
    """Thread-safe frame buffer for storing captured frames"""
    def __init__(self, maxsize: int = 10):
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.Lock()
    
    def add_frame(self, frame: np.ndarray, timestamp: float):
        with self.lock:
            self.buffer.append((frame.copy(), timestamp))
    
    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        with self.lock:
            return self.buffer[-1] if self.buffer else None
    
    def get_frames_in_range(self, start_time: float, end_time: float) -> List[Tuple[np.ndarray, float]]:
        with self.lock:
            return [(frame, ts) for frame, ts in self.buffer if start_time <= ts <= end_time]

class LEDMatrixDetector:
    """Main detector class for LED matrix communication"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.frame_buffer = FrameBuffer()
        self.matrix_corners = None
        self.is_calibrated = False
        self.detected_frames = []
        self.transmission_start_time = None
        
    def find_matrix_corners(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Find the 4 corner markers of the LED matrix"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to find bright regions (potential markers)
        _, thresh = cv2.threshold(blurred, self.config.brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        marker_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.min_contour_area < area < self.config.max_contour_area:
                # Get bounding rectangle center
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w//2, y + h//2)
                marker_candidates.append(center)
        
        if len(marker_candidates) >= 4:
            # Sort by position to identify corners
            points = np.array(marker_candidates)
            
            # Find corner points by position
            # Top-left: min(x+y), Top-right: max(x-y), Bottom-left: min(x-y), Bottom-right: max(x+y)
            tl = points[np.argmin(points[:, 0] + points[:, 1])]
            tr = points[np.argmax(points[:, 0] - points[:, 1])]
            bl = points[np.argmin(points[:, 0] - points[:, 1])]
            br = points[np.argmax(points[:, 0] + points[:, 1])]
            
            corners = np.array([tl, tr, br, bl], dtype=np.float32)
            return corners
        
        return None
    
    def extract_matrix_region(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Extract and rectify the LED matrix region"""
        # Define target rectangle (square matrix)
        target_size = 320  # Larger size for better pixel detection
        target_corners = np.array([
            [0, 0],
            [target_size, 0],
            [target_size, target_size],
            [0, target_size]
        ], dtype=np.float32)
        
        # Compute perspective transform
        transform_matrix = cv2.getPerspectiveTransform(corners, target_corners)
        
        # Apply transform
        rectified = cv2.warpPerspective(frame, transform_matrix, (target_size, target_size))
        
        return rectified
    
    def detect_led_states(self, matrix_image: np.ndarray) -> np.ndarray:
        """Detect individual LED states in the matrix"""
        gray = cv2.cvtColor(matrix_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to handle LED bleed
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Divide into 16x16 grid
        height, width = blurred.shape
        cell_h = height // self.config.matrix_size
        cell_w = width // self.config.matrix_size
        
        led_states = np.zeros((self.config.matrix_size, self.config.matrix_size), dtype=bool)
        
        for row in range(self.config.matrix_size):
            for col in range(self.config.matrix_size):
                # Extract cell region
                y1 = row * cell_h
                y2 = (row + 1) * cell_h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w
                
                cell = blurred[y1:y2, x1:x2]
                
                # Calculate average brightness
                avg_brightness = np.mean(cell)
                
                # Threshold to determine LED state
                led_states[row, col] = avg_brightness > self.config.brightness_threshold
        
        return led_states
    
    def convert_to_z_pattern(self, led_states: np.ndarray) -> np.ndarray:
        """Convert 2D LED states to 1D array following Z-pattern"""
        z_pattern = np.zeros(self.config.matrix_size * self.config.matrix_size, dtype=bool)
        
        for row in range(self.config.matrix_size):
            for col in range(self.config.matrix_size):
                if row % 2 == 0:
                    # Even row: left to right
                    index = row * self.config.matrix_size + col
                else:
                    # Odd row: right to left
                    index = row * self.config.matrix_size + (self.config.matrix_size - 1 - col)
                
                z_pattern[index] = led_states[row, col]
        
        return z_pattern
    
    def detect_calibration_patterns(self, frame_data: List[Tuple[np.ndarray, float]]) -> bool:
        """Detect calibration patterns and establish timing"""
        if len(frame_data) < 2:
            return False
        
        # Look for calibration patterns
        # Pattern 1: All white (calib1_pattern)
        # Pattern 2: Checkerboard (calib2_pattern)
        
        calib1_detected = False
        calib2_detected = False
        
        for frame, timestamp in frame_data:
            if self.matrix_corners is None:
                corners = self.find_matrix_corners(frame)
                if corners is not None:
                    self.matrix_corners = corners
                    logger.info("Matrix corners detected")
            
            if self.matrix_corners is not None:
                matrix_region = self.extract_matrix_region(frame, self.matrix_corners)
                led_states = self.detect_led_states(matrix_region)
                
                # Check for all-white pattern (calib1)
                if np.sum(led_states) > 0.9 * self.config.matrix_size**2:
                    if not calib1_detected:
                        calib1_detected = True
                        logger.info(f"Calibration pattern 1 detected at {timestamp}")
                
                # Check for checkerboard pattern (calib2)
                elif self.is_checkerboard_pattern(led_states):
                    if not calib2_detected:
                        calib2_detected = True
                        self.transmission_start_time = timestamp
                        logger.info(f"Calibration pattern 2 detected at {timestamp}")
        
        return calib1_detected and calib2_detected
    
    def is_checkerboard_pattern(self, led_states: np.ndarray) -> bool:
        """Check if LED pattern is a checkerboard"""
        expected_checkerboard = np.zeros((self.config.matrix_size, self.config.matrix_size), dtype=bool)
        
        for row in range(self.config.matrix_size):
            for col in range(self.config.matrix_size):
                expected_checkerboard[row, col] = (row + col) % 2 == 0
        
        # Calculate similarity
        similarity = np.sum(led_states == expected_checkerboard) / (self.config.matrix_size**2)
        return similarity > 0.8
    
    def extract_data_frames(self, frame_data: List[Tuple[np.ndarray, float]]) -> List[np.ndarray]:
        """Extract the three data frames from the transmission"""
        if self.transmission_start_time is None:
            return []
        
        data_frames = []
        frame_times = [2.5, 3.0, 3.5]  # Expected times for frame0, frame1, frame2
        
        for frame_time in frame_times:
            target_time = self.transmission_start_time + frame_time
            
            # Find frame closest to target time
            closest_frame = None
            min_time_diff = float('inf')
            
            for frame, timestamp in frame_data:
                time_diff = abs(timestamp - target_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_frame = frame
            
            if closest_frame is not None and min_time_diff < 0.3:  # Within 300ms tolerance
                matrix_region = self.extract_matrix_region(closest_frame, self.matrix_corners)
                led_states = self.detect_led_states(matrix_region)
                z_pattern = self.convert_to_z_pattern(led_states)
                data_frames.append(z_pattern)
                logger.info(f"Data frame {len(data_frames)} extracted (time diff: {min_time_diff:.3f}s)")
        
        return data_frames
    
    def decode_password(self, data_frames: List[np.ndarray]) -> Optional[str]:
        """Decode password from the three data frames"""
        if len(data_frames) != 3:
            logger.error(f"Expected 3 data frames, got {len(data_frames)}")
            return None
        
        # Combine frames into bit array (616 bits total)
        all_bits = np.concatenate(data_frames)
        
        # Extract password bits (first 600 bits)
        password_bits = all_bits[:600]
        
        # Extract CRC bits (last 16 bits)
        crc_bits = all_bits[600:616]
        
        # Convert password bits to characters
        password = ""
        for i in range(100):  # 100 characters
            # Extract 6 bits for each character
            start_bit = i * 6
            end_bit = start_bit + 6
            char_bits = password_bits[start_bit:end_bit]
            
            # Convert bits to value
            value = 0
            for bit in char_bits:
                value = (value << 1) | int(bit)
            
            # Convert value to character
            if value < 10:
                char = str(value)
            elif value < 36:
                char = chr(ord('A') + value - 10)
            else:
                char = 'A'  # Default fallback
            
            password += char
        
        # Verify CRC
        if self.verify_crc(password_bits, crc_bits):
            logger.info("CRC verification passed")
            return password
        else:
            logger.error("CRC verification failed")
            return None
    
    def verify_crc(self, data_bits: np.ndarray, crc_bits: np.ndarray) -> bool:
        """Verify CRC16-CCITT checksum"""
        # Convert first 600 bits to 75 bytes
        data_bytes = []
        for i in range(75):
            byte_bits = data_bits[i*8:(i+1)*8]
            byte_value = 0
            for bit in byte_bits:
                byte_value = (byte_value << 1) | int(bit)
            data_bytes.append(byte_value)
        
        # Calculate CRC
        crc = 0xFFFF
        for byte in data_bytes:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc = crc << 1
        
        crc &= 0xFFFF
        
        # Convert received CRC bits to value
        received_crc = 0
        for bit in crc_bits:
            received_crc = (received_crc << 1) | int(bit)
        
        return crc == received_crc

class LEDMatrixReceiver:
    """Main receiver class that orchestrates the detection process"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.detector = LEDMatrixDetector(DetectionConfig())
        self.is_recording = False
        self.recording_thread = None
        
    def start_camera(self) -> bool:
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Camera initialized successfully")
        return True
    
    def stop_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def capture_frames(self, duration: float = 10.0):
        """Capture frames for specified duration"""
        if not self.cap:
            logger.error("Camera not initialized")
            return
        
        logger.info(f"Starting frame capture for {duration} seconds")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if ret:
                timestamp = time.time()
                self.detector.frame_buffer.add_frame(frame, timestamp)
            else:
                logger.warning("Failed to capture frame")
                break
        
        logger.info("Frame capture completed")
    
    def process_transmission(self) -> Optional[str]:
        """Process captured frames and decode password"""
        logger.info("Processing transmission...")
        
        # Get all captured frames
        latest_frame = self.detector.frame_buffer.get_latest_frame()
        if not latest_frame:
            logger.error("No frames captured")
            return None
        
        end_time = latest_frame[1]
        start_time = end_time - 10.0  # Look at last 10 seconds
        
        frame_data = self.detector.frame_buffer.get_frames_in_range(start_time, end_time)
        
        if len(frame_data) < 10:
            logger.error(f"Insufficient frames captured: {len(frame_data)}")
            return None
        
        # Detect calibration patterns
        if not self.detector.detect_calibration_patterns(frame_data):
            logger.error("Failed to detect calibration patterns")
            return None
        
        # Extract data frames
        data_frames = self.detector.extract_data_frames(frame_data)
        
        if len(data_frames) != 3:
            logger.error(f"Failed to extract data frames: {len(data_frames)}")
            return None
        
        # Decode password
        password = self.detector.decode_password(data_frames)
        
        if password:
            logger.info(f"Password decoded successfully: {password}")
        else:
            logger.error("Failed to decode password")
        
        return password
    
    def run_detection(self, capture_duration: float = 10.0) -> Optional[str]:
        """Run complete detection process"""
        if not self.start_camera():
            return None
        
        try:
            self.capture_frames(capture_duration)
            password = self.process_transmission()
            return password
        
        finally:
            self.stop_camera()

def main():
    """Main function for testing the receiver"""
    receiver = LEDMatrixReceiver(camera_index=0)
    
    print("LED Matrix Receiver starting...")
    print("Position your robot in front of the LED matrix and press Enter to start detection...")
    input()
    
    password = receiver.run_detection(capture_duration=15.0)
    
    if password:
        print(f"SUCCESS: Password decoded: {password}")
    else:
        print("FAILED: Unable to decode password")

if __name__ == "__main__":
    main()
