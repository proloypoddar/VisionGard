import cv2
import numpy as np
import time
import threading
import subprocess
import platform
import os
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr
import queue

# ===== VISION GUARD SYSTEM =====
class VisionGuard:
    def __init__(self,
                 model_path='runs/train/yolov8_door_detection/weights/best.pt',
                 camera_id=0,
                 confidence=0.4,
                 door_class_names=None,
                 obstacle_class_names=None,
                 frame_width=800,
                 frame_height=600,
                 focal_length=800,
                 known_door_width=0.9):
        
        # Model initialization
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        
        # Class names
        self.door_class_names = door_class_names or ['door']
        self.obstacle_class_names = obstacle_class_names or []
        print(f"Door class names: {self.door_class_names}")
        print(f"Obstacle class names: {self.obstacle_class_names}")

        # Camera parameters
        self.camera_id = camera_id
        self.cap = None
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Detection parameters
        self.confidence = confidence

        # Distance estimation parameters
        self.focal_length = focal_length
        self.known_door_width = known_door_width

        # State variables
        self.running = False
        self.navigating = False
        self.door_detected = False
        self.door_distance = None
        self.door_bbox = None
        self.door_center_x = None
        self.obstacles = []

        # Navigation parameters
        self.last_guidance_time = 0
        self.guidance_interval = 2.0
        self.close_door_threshold = 1.0
        self.obstacle_warning_threshold = 1.5
        self.door_announced = False
        self.obstacle_announced = False

        # Frame processing control
        self.process_every_n_frames = 2
        self.frame_count = 0

        # Processing thread
        self.process_thread = None

        # For depth estimation
        self.depth_map = None

        # For smoother guidance
        self.direction_history = []
        self.direction_history_max = 5
        self.last_direction = None

        # For path planning
        self.path = []
        self.safety_margin = 30

        # For perspective views
        self.left_view = None
        self.right_view = None
        
        # Voice interface
        self.speech_engine = pyttsx3.init()
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self._speech_worker)
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
        # Voice recognition
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
            self.recognizer.adjust_for_ambient_noise(self.microphone)
            self.voice_thread = None
            self.listening = False
        except:
            print("Microphone not available. Voice commands disabled.")
            self.microphone = None
        
        # Display frames
        self.display_frames = {}
        
        # Initialize camera
        self.initialize_camera()

    # ===== CAMERA INITIALIZATION =====
    def initialize_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {self.camera_id}")
                
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.using_sample_images = False
            print(f"Camera initialized with resolution: {self.frame_width}x{self.frame_height}")
        except Exception as e:
            print(f"Warning: {e}")
            print("Using sample images instead...")
            self.setup_sample_images()

    # ===== DISTANCE ESTIMATION =====
    def estimate_distance(self, bbox_width):
        if bbox_width == 0:
            return float('inf')
        distance = (self.known_door_width * self.focal_length) / bbox_width
        return distance

    # ===== DIRECTION GUIDANCE =====
    def get_direction(self, door_center_x):
        frame_center_x = self.frame_width // 2
        threshold = 50

        if door_center_x < frame_center_x - threshold:
            raw_direction = "left"
        elif door_center_x > frame_center_x + threshold:
            raw_direction = "right"
        else:
            raw_direction = "forward"

        self.direction_history.append(raw_direction)
        if len(self.direction_history) > self.direction_history_max:
            self.direction_history.pop(0)

        # Use most common direction for stability
        direction_counts = {}
        for d in self.direction_history:
            direction_counts[d] = direction_counts.get(d, 0) + 1
        
        smoothed_direction = max(direction_counts, key=direction_counts.get)
        self.last_direction = smoothed_direction
        
        return smoothed_direction

    # ===== NAVIGATION VISUALIZATION =====
    def draw_navigation_arrow(self, frame, door_center_x):
        frame_center_x = self.frame_width // 2
        threshold = 50
        arrow_length = 100
        arrow_thickness = 3
        arrow_color = (0, 255, 0)  # Green

        direction = self.get_direction(door_center_x)
        
        if direction == "left":
            cv2.arrowedLine(
                frame,
                (frame_center_x, self.frame_height - 30),
                (frame_center_x - arrow_length, self.frame_height - 30),
                arrow_color,
                arrow_thickness
            )
            cv2.putText(
                frame,
                "LEFT",
                (frame_center_x - arrow_length - 50, self.frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                arrow_color,
                2
            )
        elif direction == "right":
            cv2.arrowedLine(
                frame,
                (frame_center_x, self.frame_height - 30),
                (frame_center_x + arrow_length, self.frame_height - 30),
                arrow_color,
                arrow_thickness
            )
            cv2.putText(
                frame,
                "RIGHT",
                (frame_center_x + arrow_length + 10, self.frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                arrow_color,
                2
            )
        else:
            cv2.arrowedLine(
                frame,
                (frame_center_x, self.frame_height - 30),
                (frame_center_x, self.frame_height - 30 - arrow_length),
                arrow_color,
                arrow_thickness
            )
            cv2.putText(
                frame,
                "FORWARD",
                (frame_center_x + 10, self.frame_height - 30 - arrow_length),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                arrow_color,
                2
            )

        return frame

    # ===== OBSTACLE DETECTION =====
    def check_path_obstacles(self, door_center_x, obstacles):
        if not obstacles:
            return False, None
            
        frame_center_x = self.frame_width // 2
        path_width = 100  # Width of the path in pixels
        
        # Determine path region based on door position
        if door_center_x < frame_center_x:
            # Door is to the left
            path_left = min(door_center_x, frame_center_x) - path_width//2
            path_right = max(door_center_x, frame_center_x) + path_width//2
        else:
            # Door is to the right
            path_left = min(door_center_x, frame_center_x) - path_width//2
            path_right = max(door_center_x, frame_center_x) + path_width//2
            
        # Check if any obstacle is in the path
        for obstacle in obstacles:
            obs_x = obstacle['center_x']
            if path_left <= obs_x <= path_right and obstacle['distance'] < self.obstacle_warning_threshold:
                return True, obstacle
                
        return False, None

    # ===== VOICE GUIDANCE =====
    def provide_guidance(self, door_detected, door_distance, door_center_x, obstacles):
        if not self.navigating:
            return

        current_time = time.time()
        if current_time - self.last_guidance_time < self.guidance_interval:
            return

        self.last_guidance_time = current_time

        # Check for obstacles in the path first
        if door_detected:
            path_blocked, blocking_obstacle = self.check_path_obstacles(door_center_x, obstacles)

            if path_blocked and not self.obstacle_announced:
                obstacle_distance = blocking_obstacle['distance']
                obstacle_class = blocking_obstacle['class']
                self.speak(f"Caution! {obstacle_class} in your path, {obstacle_distance:.1f} meters ahead. Stop.", priority=True)
                self.obstacle_announced = True
                return
            elif not path_blocked:
                self.obstacle_announced = False

        if door_detected:
            if not self.door_detected:
                self.speak("Door detected")

            if door_distance < self.close_door_threshold:
                if not self.door_announced:
                    self.speak("You have reached the door. Stop.", priority=True)
                    self.door_announced = True
                return

            direction = self.get_direction(door_center_x)

            if direction == "left":
                self.speak("Door is to your left, turn left")
            elif direction == "right":
                self.speak("Door is to your right, turn right")
            else:
                self.speak(f"Door is straight ahead, {door_distance:.1f} meters away")
        else:
            if self.door_detected:
                self.speak("Door lost. Searching...")
            self.door_announced = False

    # ===== FRAME PROCESSING =====
    def process_frame(self, frame):
        # Estimate depth map
        self.depth_map = self.estimate_depth_map(frame)

        # Create perspective views
        self.left_view, self.right_view = self.create_perspective_views(frame)

        # Perform detection
        results = self.model(frame, conf=self.confidence)

        # Process results
        door_detected = False
        door_distance = None
        door_bbox = None
        door_center_x = None
        obstacles = []

        # Get detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = [x1, y1, x2, y2]
                
                # Get class and confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = result.names[cls_id]
                
                # Check if it's a door
                if cls_name.lower() in [name.lower() for name in self.door_class_names]:
                    # If multiple doors, keep the closest/largest one
                    door_width = x2 - x1
                    distance = self.estimate_distance(door_width)
                    
                    if not door_detected or distance < door_distance:
                        door_detected = True
                        door_distance = distance
                        door_bbox = bbox
                        door_center_x = (x1 + x2) // 2
                else:
                    # It's an obstacle
                    obstacle_width = x2 - x1
                    obstacle_distance = self.estimate_distance(obstacle_width)

                    obstacle_info = {
                        'bbox': bbox,
                        'class': cls_name,
                        'confidence': conf,
                        'distance': obstacle_distance,
                        'center_x': (x1 + x2) // 2,
                        'center_y': (y1 + y2) // 2
                    }
                    obstacles.append(obstacle_info)

        # Draw detections on frame
        annotated_frame = results[0].plot()

        # Draw navigation arrow if door detected
        if door_detected:
            annotated_frame = self.draw_navigation_arrow(annotated_frame, door_center_x)

        return annotated_frame, door_detected, door_distance, door_bbox, door_center_x, obstacles

    # ===== VOICE INTERFACE =====
    def speak(self, text, priority=False):
        self.speech_queue.put((text, priority))

    def _speech_worker(self):
        while True:
            try:
                text, priority = self.speech_queue.get()
                if priority:
                    # Clear queue for priority messages
                    while not self.speech_queue.empty():
                        try:
                            self.speech_queue.get_nowait()
                        except queue.Empty:
                            break
                
                self.speech_engine.say(text)
                self.speech_engine.runAndWait()
                self.speech_queue.task_done()
            except Exception as e:
                print(f"Speech error: {e}")

    # ===== VOICE RECOGNITION =====
    def start_voice_recognition(self):
        if self.microphone and not self.listening:
            self.voice_thread = threading.Thread(target=self._listen_for_commands)
            self.voice_thread.daemon = True
            self.voice_thread.start()
            self.speak("Voice commands activated")

    def stop_voice_recognition(self):
        if self.listening:
            self.listening = False
            self.speak("Voice commands deactivated")

    # ===== MAIN PROCESSING LOOP =====
    def process_loop(self):
        try:
            self.frame_count = 0
            
            while self.running:
                # Read frame
                if self.using_sample_images:
                    frame = self.get_sample_frame()
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to grab frame")
                        time.sleep(0.1)
                        continue
                
                # Process every nth frame for better performance
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames != 0:
                    continue
                
                # Process frame
                _, door_detected, door_distance, door_bbox, door_center_x, obstacles = self.process_frame(frame)

                # Update state
                self.door_detected = door_detected
                self.door_distance = door_distance
                self.door_bbox = door_bbox
                self.door_center_x = door_center_x
                self.obstacles = obstacles

                # Provide guidance
                if self.navigating:
                    self.provide_guidance(door_detected, door_distance, door_center_x, obstacles)
                
                # Display frame
                cv2.imshow("Vision Guard", frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.start_navigation()
                elif key == ord('s'):
                    self.stop_navigation()
                elif key == ord('v'):
                    if not self.listening:
                        self.start_voice_recognition()
                    else:
                        self.stop_voice_recognition()

        except Exception as e:
            print(f"Error in processing loop: {e}")
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

    # ===== NAVIGATION CONTROL =====
    def start_navigation(self):
        if not self.navigating:
            self.navigating = True
            self.door_announced = False
            self.obstacle_announced = False
            self.speak("Starting navigation to the door. Please move slowly.", priority=True)

    def stop_navigation(self):
        if self.navigating:
            self.navigating = False
            self.speak("Navigation stopped.", priority=True)

    # ===== SYSTEM CONTROL =====
    def start(self):
        if self.running:
            return

        self.running = True

        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()

        print("Vision Guard started")
        print("Press 'q' to quit, 'n' to start navigation, 's' to stop navigation, 'v' to toggle voice commands")

        # Initial announcement
        self.speak("Vision Guard is ready. Press N to start navigation or say take me to the door.")

    def stop(self):
        self.running = False
        self.listening = False
        
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
            
        print("Vision Guard stopped")


# ===== MAIN FUNCTION =====
def main():
    # Create Vision Guard system
    vision_guard = VisionGuard(
        model_path='runs/train/yolov8_door_detection/weights/best.pt',
        camera_id=0,
        confidence=0.4,
        door_class_names=['door', 'Door', 'hinged', 'knob', 'lever'],
        frame_width=800,
        frame_height=600
    )

    try:
        # Start system
        vision_guard.start()

        # Keep main thread alive
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop system
        vision_guard.stop()


if __name__ == "__main__":
    main()
