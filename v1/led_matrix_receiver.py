import cv2
import numpy as np
import time
import threading
import queue
from collections import deque
from typing import List, Tuple, Optional, Dict
import json

class LEDMatrixReceiver:
    def __init__(self, matrix_size=(16, 16), debug=False):
        self.matrix_size = matrix_size
        self.debug = debug
        
        # Detection parameters
        self.min_contour_area = 100
        self.max_contour_area = 50000
        self.brightness_threshold = 100
        self.color_similarity_threshold = 30
        
        # Protocol parameters
        self.frame_duration = 0.2  # 200ms per frame
        self.startup_frames = 5
        self.length_indicator_frames = 3
        
        # State tracking
        self.matrix_corners = None
        self.calibrated = False
        self.frame_buffer = deque(maxlen=1000)
        self.current_password = ""
        self.transmission_state = "idle"
        
        # Frame analysis
        self.last_frame_time = 0
        self.frame_count = 0
        
        # Color definitions matching Arduino code
        self.colors = {
            'black': (0, 0, 0),
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'white': (255, 255, 255),
            'yellow': (0, 255, 255),
            'cyan': (255, 255, 0),
            'purple': (255, 0, 255)
        }
        
    def detect_matrix_region(self, frame):
        """Detect the LED matrix region in the frame using corner markers"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to find bright regions
        _, thresh = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                filtered_contours.append(contour)
        
        if len(filtered_contours) < 4:
            return None
        
        # Find rectangular regions that could be the matrix
        for contour in filtered_contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                # Check if it's roughly square
                rect = cv2.boundingRect(approx)
                aspect_ratio = rect[2] / rect[3]
                if 0.8 < aspect_ratio < 1.2:  # Roughly square
                    return self.order_corners(approx.reshape(-1, 2))
        
        return None
    
    def order_corners(self, corners):
        """Order corners as [top-left, top-right, bottom-right, bottom-left]"""
        # Sort by y-coordinate
        corners = corners[corners[:, 1].argsort()]
        
        # Split into top and bottom
        top_corners = corners[:2]
        bottom_corners = corners[2:]
        
        # Sort top corners by x-coordinate
        top_corners = top_corners[top_corners[:, 0].argsort()]
        # Sort bottom corners by x-coordinate (reversed for bottom-right first)
        bottom_corners = bottom_corners[bottom_corners[:, 0].argsort()]
        
        return np.array([
            top_corners[0],    # top-left
            top_corners[1],    # top-right
            bottom_corners[1], # bottom-right
            bottom_corners[0]  # bottom-left
        ])
    
    def extract_matrix_data(self, frame):
        """Extract LED matrix data from the frame"""
        if self.matrix_corners is None:
            self.matrix_corners = self.detect_matrix_region(frame)
            if self.matrix_corners is None:
                return None
        
        # Create transformation matrix for perspective correction
        matrix_width = 160  # 16cm = 160 pixels at 1px/mm
        matrix_height = 160
        
        dst_corners = np.array([
            [0, 0],
            [matrix_width, 0],
            [matrix_width, matrix_height],
            [0, matrix_height]
        ], dtype=np.float32)
        
        transform_matrix = cv2.getPerspectiveTransform(
            self.matrix_corners.astype(np.float32), 
            dst_corners
        )
        
        # Apply perspective transformation
        corrected = cv2.warpPerspective(frame, transform_matrix, (matrix_width, matrix_height))
        
        # Extract individual pixel values
        pixel_data = np.zeros((self.matrix_size[1], self.matrix_size[0], 3), dtype=np.uint8)
        
        pixels_per_cell_x = matrix_width // self.matrix_size[0]
        pixels_per_cell_y = matrix_height // self.matrix_size[1]
        
        for y in range(self.matrix_size[1]):
            for x in range(self.matrix_size[0]):
                # Sample from center of each cell
                sample_x = x * pixels_per_cell_x + pixels_per_cell_x // 2
                sample_y = y * pixels_per_cell_y + pixels_per_cell_y // 2
                
                # Average over a small region to reduce noise
                region = corrected[
                    sample_y-2:sample_y+3,
                    sample_x-2:sample_x+3
                ]
                
                if region.size > 0:
                    pixel_data[y, x] = np.mean(region, axis=(0, 1))
        
        return pixel_data
    
    def classify_color(self, bgr_color):
        """Classify a BGR color to the nearest predefined color"""
        min_distance = float('inf')
        best_color = 'black'
        
        for color_name, color_bgr in self.colors.items():
            distance = np.sqrt(np.sum((np.array(bgr_color) - np.array(color_bgr))**2))
            if distance < min_distance:
                min_distance = distance
                best_color = color_name
        
        return best_color if min_distance < self.color_similarity_threshold else 'black'
    
    def analyze_frame_pattern(self, pixel_data):
        """Analyze the pattern in the current frame"""
        if pixel_data is None:
            return None
        
        # Convert to color classifications
        color_matrix = np.zeros((self.matrix_size[1], self.matrix_size[0]), dtype=object)
        
        for y in range(self.matrix_size[1]):
            for x in range(self.matrix_size[0]):
                color_matrix[y, x] = self.classify_color(pixel_data[y, x])
        
        # Analyze pattern type
        pattern_type = self.detect_pattern_type(color_matrix)
        
        return {
            'pattern_type': pattern_type,
            'color_matrix': color_matrix,
            'timestamp': time.time()
        }
    
    def detect_pattern_type(self, color_matrix):
        """Detect what type of pattern is being displayed"""
        # Count different colors
        unique_colors = set()
        for row in color_matrix:
            for color in row:
                unique_colors.add(color)
        
        # Check for specific patterns
        if len(unique_colors) == 1:
            if 'white' in unique_colors:
                return 'calibration_white'
            elif 'black' in unique_colors:
                return 'blank'
            elif 'cyan' in unique_colors:
                return 'length_indicator'
            elif 'purple' in unique_colors:
                return 'end_transmission'
        
        elif len(unique_colors) == 2 and 'white' in unique_colors and 'black' in unique_colors:
            return 'calibration_checkerboard'
        
        elif len(unique_colors) == 4 and all(c in unique_colors for c in ['red', 'green', 'blue', 'yellow']):
            return 'calibration_corners'
        
        elif len(unique_colors) <= 4 and any(c in unique_colors for c in ['red', 'green', 'blue', 'yellow']):
            return 'data_quadrant'
        
        elif 'yellow' in unique_colors:
            return 'checksum'
        
        return 'unknown'
    
    def decode_6bit_value(self, color_matrix):
        """Decode 6-bit value from quadrant pattern"""
        if color_matrix is None:
            return None
        
        # Split into quadrants
        mid_y = self.matrix_size[1] // 2
        mid_x = self.matrix_size[0] // 2
        
        # Sample from center of each quadrant
        top_left = color_matrix[mid_y//2, mid_x//2]
        top_right = color_matrix[mid_y//2, mid_x + mid_x//2]
        bottom_left = color_matrix[mid_y + mid_y//2, mid_x//2]
        bottom_right = color_matrix[mid_y + mid_y//2, mid_x + mid_x//2]
        
        # Map colors to bit values
        color_to_bits = {
            'black': 0,
            'red': 1,
            'green': 2,
            'blue': 3
        }
        
        # Decode quadrant values
        top_left_val = color_to_bits.get(top_left, 0)
        top_right_val = color_to_bits.get(top_right, 0)
        bottom_left_val = 1 if bottom_left == 'white' else 0
        bottom_right_val = 1 if bottom_right == 'white' else 0
        
        # Combine into 6-bit value
        # Top-left: bits 5,4  Top-right: bits 3,2  Bottom-left: bit 1  Bottom-right: bit 0
        value = (top_left_val << 4) | (top_right_val << 2) | (bottom_left_val << 1) | bottom_right_val
        
        return value
    
    def decode_checksum(self, color_matrix):
        """Decode checksum from binary pattern"""
        if color_matrix is None:
            return None
        
        checksum = 0
        for i in range(8):
            # Sample from each bit column
            col_start = i * 2
            if col_start < self.matrix_size[0]:
                # Sample from middle of the column
                sample_color = color_matrix[self.matrix_size[1]//2, col_start]
                bit_value = 1 if sample_color == 'yellow' else 0
                checksum |= (bit_value << (7 - i))
        
        return checksum
    
    def bit_value_to_char(self, value):
        """Convert 6-bit value back to character"""
        if value < 26:
            return chr(ord('A') + value)
        elif value < 36:
            return chr(ord('0') + value - 26)
        return 'A'  # Default fallback
    
    def calculate_crc8(self, data):
        """Calculate CRC-8 checksum (matching Arduino implementation)"""
        crc = 0
        polynomial = 0x07
        
        for char in data:
            crc ^= ord(char)
            for _ in range(8):
                if crc & 0x80:
                    crc = (crc << 1) ^ polynomial
                else:
                    crc <<= 1
                crc &= 0xFF
        
        return crc
    
    def process_transmission(self, frame_data_list):
        """Process a complete transmission sequence"""
        if len(frame_data_list) < self.startup_frames + self.length_indicator_frames + 1:
            return None
        
        # Skip startup frames
        data_frames = frame_data_list[self.startup_frames:]
        
        # Decode length indicator
        password_length = 0
        for i in range(self.length_indicator_frames):
            if i < len(data_frames):
                frame_data = data_frames[i]
                if frame_data['pattern_type'] == 'length_indicator':
                    # This is a simplified length extraction
                    # In practice, you'd need to decode the binary pattern
                    password_length = 100  # Default to expected length
        
        # Skip length indicator frames
        char_frames = data_frames[self.length_indicator_frames:]
        
        # Decode password characters
        password = ""
        checksum_frame = None
        
        for frame_data in char_frames:
            if frame_data['pattern_type'] == 'data_quadrant':
                char_value = self.decode_6bit_value(frame_data['color_matrix'])
                if char_value is not None:
                    password += self.bit_value_to_char(char_value)
            elif frame_data['pattern_type'] == 'checksum':
                checksum_frame = frame_data
                break
        
        # Verify checksum
        if checksum_frame:
            received_checksum = self.decode_checksum(checksum_frame['color_matrix'])
            calculated_checksum = self.calculate_crc8(password)
            
            if received_checksum == calculated_checksum:
                return {
                    'password': password,
                    'checksum_valid': True,
                    'length': len(password)
                }
            else:
                return {
                    'password': password,
                    'checksum_valid': False,
                    'received_checksum': received_checksum,
                    'calculated_checksum': calculated_checksum,
                    'length': len(password)
                }
        
        return {
            'password': password,
            'checksum_valid': None,
            'length': len(password)
        }
    
    def capture_and_decode(self, video_source=0, duration=30):
        """Main capture and decode function"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return None
        
        # Set camera properties for better detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_sequence = []
        start_time = time.time()
        transmission_started = False
        
        print("Starting capture... Press 'q' to quit early")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract matrix data
            pixel_data = self.extract_matrix_data(frame)
            frame_analysis = self.analyze_frame_pattern(pixel_data)
            
            if frame_analysis:
                current_time = time.time()
                
                # Check if this looks like start of transmission
                if not transmission_started and frame_analysis['pattern_type'] == 'calibration_white':
                    transmission_started = True
                    frame_sequence = []
                    print("Transmission started!")
                
                if transmission_started:
                    frame_sequence.append(frame_analysis)
                    
                    # Check for end of transmission
                    if frame_analysis['pattern_type'] == 'end_transmission':
                        print("Transmission ended!")
                        break
                
                # Debug output
                if self.debug:
                    print(f"Frame {len(frame_sequence)}: {frame_analysis['pattern_type']}")
            
            # Show frame if debugging
            if self.debug:
                display_frame = frame.copy()
                if self.matrix_corners is not None:
                    cv2.polylines(display_frame, [self.matrix_corners.astype(int)], True, (0, 255, 0), 2)
                
                cv2.imshow('LED Matrix Receiver', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Timeout check
            if time.time() - start_time > duration:
                print("Capture timeout reached")
                break
        
        cap.release()
        if self.debug:
            cv2.destroyAllWindows()
        
        # Process the complete transmission
        if len(frame_sequence) > 0:
            result = self.process_transmission(frame_sequence)
            return result
        
        return None

def main():
    # Create receiver instance
    receiver = LEDMatrixReceiver(debug=True)
    
    # Start capture and decode
    print("LED Matrix Password Receiver")
    print("Make sure the LED matrix is visible in the camera view")
    print("Starting capture in 3 seconds...")
    
    time.sleep(3)
    
    result = receiver.capture_and_decode(video_source=0, duration=60)
    
    if result:
        print("\n" + "="*50)
        print("TRANSMISSION RECEIVED!")
        print("="*50)
        print(f"Password: {result['password']}")
        print(f"Length: {result['length']}")
        print(f"Checksum Valid: {result['checksum_valid']}")
        
        if result['checksum_valid'] is False:
            print(f"Received Checksum: {result.get('received_checksum', 'N/A')}")
            print(f"Calculated Checksum: {result.get('calculated_checksum', 'N/A')}")
        
        # Save result to file
        with open('received_password.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print("\nResult saved to 'received_password.json'")
    else:
        print("No transmission received or failed to decode")

if __name__ == "__main__":
    main()
