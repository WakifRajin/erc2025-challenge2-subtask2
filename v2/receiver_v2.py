import cv2
import numpy as np
from scipy import ndimage
import os
import time

class LEDPanelDecoder:
    def __init__(self, video_source=0, display=False):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        self.display = display
        self.panel_bbox = None  # (x, y, w, h)
        self.cell_centers = None
        self.calib1_intensities = None
        self.calib2_expected = None
        self.thresholds = None
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # Default if FPS not available
        
    def detect_panel(self, frame):
        """Detect LED panel using background subtraction and contour analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Find largest rectangular contour
        max_area = 0
        best_rect = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:  # Minimum panel size
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            area = cv2.contourArea(box)
            if area > max_area:
                max_area = area
                best_rect = box
        
        if best_rect is None:
            return None
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(best_rect)
        return (x, y, w, h)
    
    def define_grid(self, bbox):
        """Create 16x16 grid within panel boundaries"""
        x, y, w, h = bbox
        cell_centers = np.zeros((16, 16, 2), dtype=np.float32)
        
        for i in range(16):
            for j in range(16):
                # Calculate center position with Z-pattern mapping
                if i % 2 == 0:  # Left to right
                    center_x = x + (j + 0.5) * (w / 16)
                else:  # Right to left
                    center_x = x + (15.5 - j) * (w / 16)
                center_y = y + (i + 0.5) * (h / 16)
                cell_centers[i, j] = [center_x, center_y]
                
        return cell_centers
    
    def get_led_intensity(self, frame, i, j, radius=5):
        """Get average intensity for a specific LED"""
        if self.cell_centers is None:
            return 0
            
        cx, cy = self.cell_centers[i, j]
        x0 = int(max(0, cx - radius))
        y0 = int(max(0, cy - radius))
        x1 = int(min(frame.shape[1], cx + radius))
        y1 = int(min(frame.shape[0], cy + radius))
        
        if x0 >= x1 or y0 >= y1:
            return 0
            
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return 0
            
        # Use green channel (most sensitive in most cameras)
        if len(roi.shape) == 3:
            roi = roi[:, :, 1]  # Green channel
            
        return np.mean(roi)
    
    def find_calibration_patterns(self):
        """Detect calibration frames in video stream"""
        calib1_frame = None
        calib2_frame = None
        
        # Find calib1 (all white)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if self.panel_bbox is None:
                self.panel_bbox = self.detect_panel(frame)
                if self.panel_bbox is None:
                    continue
                self.cell_centers = self.define_grid(self.panel_bbox)
            
            # Check panel intensity
            intensities = []
            for i in range(16):
                for j in range(16):
                    intensity = self.get_led_intensity(frame, i, j)
                    intensities.append(intensity)
            
            if np.mean(intensities) > 200:  # High intensity threshold
                calib1_frame = frame
                self.calib1_intensities = intensities
                break
        
        if calib1_frame is None:
            return False, "Calib1 not found"
        
        # Find calib2 (checkerboard pattern)
        start_time = time.time()
        while time.time() - start_time < 3:  # Timeout after 3 seconds
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Verify checkerboard pattern
            errors = 0
            expected = []
            for i in range(16):
                for j in range(16):
                    visual_col = j if i % 2 == 0 else 15 - j
                    expected_val = 1 if (i + visual_col) % 2 == 0 else 0
                    intensity = self.get_led_intensity(frame, i, j)
                    threshold = (self.calib1_intensities[i*16+j] * 0.5 + intensity * 0.5)
                    detected_val = 1 if intensity > threshold else 0
                    
                    if detected_val != expected_val:
                        errors += 1
                    expected.append(expected_val)
            
            if errors < 50:  # Allow some errors
                calib2_frame = frame
                self.calib2_expected = expected
                break
        
        if calib2_frame is None:
            return False, "Calib2 not found"
        
        # Calculate dynamic thresholds
        self.thresholds = np.zeros((16, 16))
        for i in range(16):
            for j in range(16):
                idx = i * 16 + j
                white_ref = self.calib1_intensities[idx]
                black_ref = self.get_led_intensity(calib2_frame, i, j)
                
                if self.calib2_expected[idx] == 1:  # White in calib2
                    self.thresholds[i, j] = (white_ref + black_ref) * 0.4
                else:  # Black in calib2
                    self.thresholds[i, j] = (white_ref + black_ref) * 0.6
        
        return True, "Calibration successful"
    
    def capture_data_frames(self):
        """Capture and process data frames"""
        frames = []
        for _ in range(3):  # Capture 3 data frames
            ret, frame = self.cap.read()
            if not ret:
                return None
            frames.append(frame)
            # Skip frames to account for 0.5s interval
            for _ in range(int(self.fps * 0.5) - 1):
                self.cap.read()
        return frames
    
    def extract_bits(self, frames):
        """Extract bits from data frames"""
        bits = []
        frame_bits = [256, 256, 104]  # Bits per frame
        
        for frame_idx, frame in enumerate(frames):
            bits.extend([0] * frame_bits[frame_idx])
            bit_count = 0
            
            for i in range(16):
                for j in range(16):
                    # Stop when frame bits are captured
                    if bit_count >= frame_bits[frame_idx]:
                        break
                    
                    intensity = self.get_led_intensity(frame, i, j)
                    threshold = self.thresholds[i, j]
                    bit = 1 if intensity > threshold else 0
                    
                    # Calculate position in Z-pattern
                    if i % 2 == 0:  # Left to right
                        pos = i * 16 + j
                    else:  # Right to left
                        pos = i * 16 + (15 - j)
                    
                    if frame_idx == 0:  # Frame 0: bits 0-255
                        bits[pos] = bit
                    elif frame_idx == 1:  # Frame 1: bits 256-511
                        bits[256 + pos] = bit
                    else:  # Frame 2: bits 512-615
                        if pos < 104:
                            bits[512 + pos] = bit
                    
                    bit_count += 1
        return bits
    
    def decode_password(self, bits):
        """Convert bits to password with CRC validation"""
        # Extract data and CRC
        data_bits = bits[:600]
        crc_bits = bits[600:616]
        received_crc = int(''.join(map(str, crc_bits)), 2)
        
        # Convert to bytes
        data_bytes = bytearray()
        for i in range(0, 600, 8):
            byte_str = ''.join(map(str, data_bits[i:i+8]))
            data_bytes.append(int(byte_str, 2))
        
        # Compute CRC
        computed_crc = self.compute_crc(data_bytes)
        if computed_crc != received_crc:
            return None, "CRC mismatch"
        
        # Decode password
        password = []
        for i in range(0, 600, 6):
            chunk = data_bits[i:i+6]
            if len(chunk) < 6:
                break
            value = int(''.join(map(str, chunk)), 2)
            if value < 10:
                password.append(chr(48 + value))  # 0-9
            elif value < 36:
                password.append(chr(65 + value - 10))  # A-Z
            else:
                password.append('?')  # Invalid
        
        return ''.join(password), None
    
    def compute_crc(self, data):
        """Compute CRC16-CCITT checksum"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc = crc << 1
                crc &= 0xFFFF
        return crc
    
    def process_video(self):
        """Main processing pipeline"""
        # Find calibration patterns
        success, msg = self.find_calibration_patterns()
        if not success:
            return None, msg
        
        # Capture data frames
        frames = self.capture_data_frames()
        if frames is None:
            return None, "Failed to capture data frames"
        
        # Extract bits
        bits = self.extract_bits(frames)
        if len(bits) != 616:
            return None, f"Expected 616 bits, got {len(bits)}"
        
        # Decode password
        password, error = self.decode_password(bits)
        if error:
            return None, error
        
        return password, "Success"
    
    def release(self):
        """Release resources"""
        if self.cap.isOpened():
            self.cap.release()

# Usage example
if __name__ == "__main__":
    decoder = LEDPanelDecoder(video_source="panel_video.mp4", display=True)
    password, status = decoder.process_video()
    decoder.release()
    
    if password:
        print(f"Decoded password: {password}")
    else:
        print(f"Error: {status}")
