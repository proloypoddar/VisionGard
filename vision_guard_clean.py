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

class VisionGuard:
    def __init__(self,
                 model_path='runs/train/yolov8_door_detection/weights/best.pt',
                 camera_id=0,
                 confidence=0.4,
                 door_class_names=None,
                 obstacle_class_names=None,
                 frame_width=800,  # Increased resolution for better visibility
                 frame_height=600,
                 focal_length=800,
                 known_door_width=0.9):  # meters
        # Initialize YOLO model
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)

        # Set default door class names if not provided
        if door_class_names is None:
            self.door_class_names = ['door', 'Door', 'hinged', 'knob', 'lever']
        else:
            self.door_class_names = door_class_names

        # Set default obstacle class names if not provided
        if obstacle_class_names is None:
            # Include all class names except door-related ones as potential obstacles
            # This ensures we detect any object that's not a door as an obstacle
            self.obstacle_class_names = []
            if hasattr(self.model, 'names'):
                for _, name in self.model.names.items():  # Use _ for unused index
                    if name.lower() not in [door_name.lower() for door_name in self.door_class_names]:
                        self.obstacle_class_names.append(name)

            # If model doesn't have names or no names were found, use COCO class names
            if not self.obstacle_class_names:
                self.obstacle_class_names = [
                    'person', 'bicycle', 'car', 'motorcycle', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
                    # Add more generic obstacle names
                    'wall', 'furniture', 'object', 'obstacle'
                ]
        else:
            self.obstacle_class_names = obstacle_class_names

        print(f"Door class names: {self.door_class_names}")
        print(f"Obstacle class names: {self.obstacle_class_names}")

        # Initialize camera
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
        self.guidance_interval = 2.0  # seconds
        self.close_door_threshold = 1.0  # meters
        self.obstacle_warning_threshold = 1.5  # meters
        self.door_announced = False
        self.obstacle_announced = False

        # Frame processing rate control
        self.process_every_n_frames = 2  # Process every 2nd frame for better performance
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
        self.safety_margin = 30  # pixels

        # For perspective views
        self.left_view = None
        self.right_view = None

        # For decision making
        self.decision = None
        self.decision_confidence = 0.0

        # Initialize TTS engine
        self.is_windows = platform.system() == 'Windows'
        if self.is_windows:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
        else:
            self.tts_engine = None

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
        except Exception as e:
            print(f"Microphone initialization error: {e}")
            print("Voice commands will be disabled. You can still use keyboard controls.")
            print("To enable voice commands, install PyAudio: pip install pyaudio")
            self.microphone = None

        # Message queue for TTS
        self.message_queue = queue.Queue()
        self.speaking = False

        # Start the TTS thread
        self.tts_thread = threading.Thread(target=self._process_tts_queue, daemon=True)
        self.tts_thread.start()

        # Voice command thread
        self.listening = False
        self.listen_thread = None

        # Display frames
        self.display_frames = {}

    def _process_tts_queue(self):
        """Process messages in the TTS queue."""
        while True:
            try:
                message = self.message_queue.get(timeout=0.1)
                self.speaking = True
                if self.is_windows and self.tts_engine:
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
                else:
                    # Use subprocess for non-Windows platforms
                    subprocess.Popen(['espeak', '-s', '150', '-a', '200', message])
                    time.sleep(len(message) * 0.1)  # Approximate time to speak
                self.speaking = False
                self.message_queue.task_done()
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"TTS error: {e}")
                self.speaking = False

    def speak(self, text, priority=False):
        print(f"Speaking: {text}")

        if priority:
            # Clear the queue
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.task_done()
                except queue.Empty:
                    break

            # Stop current speech
            if self.is_windows and self.tts_engine:
                self.tts_engine.stop()

        self.message_queue.put(text)

    def initialize_camera(self):
        """Initialize the camera or use a sample image if camera is not available."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        # Try to open the camera
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

            # Try to read a frame to confirm camera is working
            ret, _ = self.cap.read()
            if not ret:
                raise RuntimeError("Camera opened but failed to read frame")

        except Exception as e:
            print(f"Camera error: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None

        # Check if camera opened successfully
        if self.cap is None or not self.cap.isOpened():
            print(f"Warning: Could not open camera {self.camera_id}")
            print("Using sample images instead...")

            # Create a list of sample images to simulate a video feed
            self.sample_images = []

            # Try to find sample images in the dataset
            sample_paths = [
                "Door_Detection_Research_Project/images/results",
                "RaspberryPi_Door_Assistant/images/results",
                "train/images",
                "valid/images",
                "test/images"
            ]
            # Look for images in each path
            for path in sample_paths:
                if os.path.exists(path):
                    image_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    if image_files:
                        print(f"Found {len(image_files)} sample images in {path}")
                        self.sample_images.extend(image_files[:20])  # Limit to 20 images per folder
                        if len(self.sample_images) >= 5:  # If we have at least 5 images, that's enough
                            break

            # If no images found, create a blank image
            if not self.sample_images:
                print("No sample images found. Creating a blank image.")
                blank_image = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                cv2.putText(blank_image, "No camera available", (50, self.frame_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                self.sample_images = [blank_image]
                self.current_sample_index = 0
                self.using_sample_images = True
                self.using_sample_image_paths = False
            else:
                self.current_sample_index = 0
                self.using_sample_images = True
                self.using_sample_image_paths = True

            print(f"Using {len(self.sample_images)} sample images for simulation")
        else:
            # Get actual camera properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.using_sample_images = False
            print(f"Camera initialized with resolution: {self.frame_width}x{self.frame_height}")

    def estimate_distance(self, bbox_width):
        if bbox_width == 0:
            return float('inf')

        distance = (self.known_door_width * self.focal_length) / bbox_width
        return distance

    def estimate_depth_map(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Sobel filter to get gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Normalize to 0-255
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Invert (stronger edges are closer)
        depth_map = 255 - magnitude.astype(np.uint8)

        # Apply Gaussian blur to smooth the depth map
        depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)

        return depth_map

    def create_perspective_views(self, frame):
        # Get frame dimensions
        height, width = frame.shape[:2]

        # Split the frame into left and right halves
        mid = width // 2
        left_view = frame[:, :mid].copy()
        right_view = frame[:, mid:].copy()

        # Add a vertical line to show the split
        cv2.line(left_view, (mid-1, 0), (mid-1, height), (0, 255, 255), 2)
        cv2.line(right_view, (0, 0), (0, height), (0, 255, 255), 2)

        # Add labels to the views
        cv2.putText(left_view, "LEFT VIEW", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(right_view, "RIGHT VIEW", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return left_view, right_view

    def create_obstacle_map(self, obstacles):
        obstacle_map = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)

        # Mark obstacles on the map
        for obstacle in obstacles:
            x1, y1, x2, y2 = obstacle['bbox']
            # Add safety margin around obstacles
            x1 = max(0, x1 - self.safety_margin)
            y1 = max(0, y1 - self.safety_margin)
            x2 = min(self.frame_width, x2 + self.safety_margin)
            y2 = min(self.frame_height, y2 + self.safety_margin)

            obstacle_map[y1:y2, x1:x2] = 1

        return obstacle_map

    def find_path(self, start_point, goal_point, obstacle_map):
        from scipy.ndimage import distance_transform_edt

        # Create distance transform from obstacles
        # This gives each pixel its distance to the nearest obstacle
        dist_transform = distance_transform_edt(1 - obstacle_map)

        # Create attractive potential field (goal)
        y, x = np.indices((self.frame_height, self.frame_width))
        goal_x, goal_y = goal_point
        attractive = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)

        # Combine potential fields
        # Higher values of dist_transform mean safer areas
        # Lower values of attractive mean closer to goal
        # We want to maximize safety while minimizing distance to goal
        potential = attractive - 5.0 * dist_transform

        # Find path using gradient descent
        path = []
        current = np.array(start_point)
        path.append(current.copy())

        max_iterations = 100  # Reduced for performance
        step_size = 5
        goal_threshold = 20

        for _ in range(max_iterations):
            # Check if we're close enough to the goal
            if np.linalg.norm(current - np.array(goal_point)) < goal_threshold:
                break

            # Get current position (rounded to integers)
            cx, cy = np.round(current).astype(int)
            cx = np.clip(cx, 0, self.frame_width - 1)
            cy = np.clip(cy, 0, self.frame_height - 1)

            # Sample potential field in neighborhood
            window_size = 5
            x_min = max(0, cx - window_size)
            x_max = min(self.frame_width, cx + window_size + 1)
            y_min = max(0, cy - window_size)
            y_max = min(self.frame_height, cy + window_size + 1)

            window = potential[y_min:y_max, x_min:x_max]
            min_idx = np.unravel_index(np.argmin(window), window.shape)

            # Move towards minimum potential
            next_y, next_x = min_idx
            next_point = np.array([x_min + next_x, y_min + next_y])

            # Ensure we're not stuck at the same point
            if np.array_equal(next_point, np.round(current)):
                # If stuck, take a random step
                angle = np.random.uniform(0, 2 * np.pi)
                next_point = current + step_size * np.array([np.cos(angle), np.sin(angle)])
                next_point = np.clip(next_point, [0, 0], [self.frame_width - 1, self.frame_height - 1])

            # Update current position
            direction = next_point - current
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm

            current = current + step_size * direction
            current = np.clip(current, [0, 0], [self.frame_width - 1, self.frame_height - 1])

            # Add to path
            path.append(current.copy())

            # Check if we're stuck in an obstacle
            cx, cy = np.round(current).astype(int)
            if obstacle_map[cy, cx] == 1:
                # If in obstacle, backtrack and try again
                if len(path) > 1:
                    current = path[-2]
                    path.pop()

        self.path = path
        return path

    def draw_path(self, frame, path=None):
        if path is None:
            path = self.path

        if not path:
            return frame

        # Draw path as a line
        points = np.array([point for point in path], dtype=np.int32)
        cv2.polylines(frame, [points], False, (0, 255, 255), 2)

        # Draw start and end points
        if len(path) > 0:
            cv2.circle(frame, tuple(points[0]), 5, (0, 255, 0), -1)
            cv2.circle(frame, tuple(points[-1]), 5, (0, 0, 255), -1)

        return frame

    def get_direction(self, door_center_x):
        frame_center_x = self.frame_width // 2
        threshold = 50  # Threshold for considering door as centered

        # Determine raw direction
        if door_center_x < frame_center_x - threshold:
            raw_direction = "left"
        elif door_center_x > frame_center_x + threshold:
            raw_direction = "right"
        else:
            raw_direction = "forward"

        # Add to history
        self.direction_history.append(raw_direction)
        if len(self.direction_history) > self.direction_history_max:
            self.direction_history.pop(0)

        # Count occurrences
        left_count = self.direction_history.count("left")
        right_count = self.direction_history.count("right")
        forward_count = self.direction_history.count("forward")

        # Determine smoothed direction
        if left_count > right_count and left_count > forward_count:
            smoothed_direction = "left"
        elif right_count > left_count and right_count > forward_count:
            smoothed_direction = "right"
        else:
            smoothed_direction = "forward"

        # Only update if direction has changed or it's the first time
        if self.last_direction != smoothed_direction or self.last_direction is None:
            self.last_direction = smoothed_direction

        return self.last_direction

    def draw_navigation_arrow(self, frame, door_center_x):
        frame_center_x = self.frame_width // 2
        arrow_length = 50
        arrow_color = (0, 255, 255)
        arrow_thickness = 2

        # Determine direction
        direction = self.get_direction(door_center_x)

        # Draw appropriate arrow
        if direction == "left":
            # Door is to the left
            cv2.arrowedLine(
                frame,
                (frame_center_x, self.frame_height - 30),
                (frame_center_x - arrow_length, self.frame_height - 30),
                arrow_color,
                arrow_thickness
            )
            # Add text
            cv2.putText(
                frame,
                "LEFT",
                (frame_center_x - arrow_length - 40, self.frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                arrow_color,
                2
            )
        elif direction == "right":
            # Door is to the right
            cv2.arrowedLine(
                frame,
                (frame_center_x, self.frame_height - 30),
                (frame_center_x + arrow_length, self.frame_height - 30),
                arrow_color,
                arrow_thickness
            )
            # Add text
            cv2.putText(
                frame,
                "RIGHT",
                (frame_center_x + arrow_length + 10, self.frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                arrow_color,
                2
            )
        else:
            # Door is centered, draw forward arrow
            cv2.arrowedLine(
                frame,
                (frame_center_x, self.frame_height - 30),
                (frame_center_x, self.frame_height - 30 - arrow_length),
                arrow_color,
                arrow_thickness
            )
            # Add text
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

    def check_path_obstacles(self, door_center_x, obstacles):
        if not obstacles:
            return False, None

        # Define path corridor width
        corridor_width = self.frame_width // 4

        # Get direction to door
        direction = self.get_direction(door_center_x)

        # Check each obstacle
        for obstacle in obstacles:
            obstacle_x = obstacle['center_x']
            obstacle_distance = obstacle['distance']

            # Skip obstacles that are too far
            if obstacle_distance > self.obstacle_warning_threshold:
                continue

            # Check if obstacle is in the path
            if direction == "forward":
                # For forward direction, check if obstacle is in the center corridor
                if abs(obstacle_x - self.frame_width // 2) < corridor_width:
                    return True, obstacle
            elif direction == "left":
                # For left direction, check if obstacle is in the left corridor
                if obstacle_x < self.frame_width // 2 and abs(obstacle_x - door_center_x) < corridor_width:
                    return True, obstacle
            elif direction == "right":
                # For right direction, check if obstacle is in the right corridor
                if obstacle_x > self.frame_width // 2 and abs(obstacle_x - door_center_x) < corridor_width:
                    return True, obstacle

        return False, None

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
                # Announce obstacle
                obstacle_distance = blocking_obstacle['distance']
                obstacle_class = blocking_obstacle['class']
                self.speak(f"Caution! {obstacle_class} in your path, {obstacle_distance:.1f} meters ahead. Stop.", priority=True)
                self.obstacle_announced = True
                return
            elif not path_blocked:
                self.obstacle_announced = False

        if door_detected:
            # Door is detected
            if not self.door_detected:
                # First time detecting the door
                self.speak("Door detected")

            # Check distance to door
            if door_distance < self.close_door_threshold:
                if not self.door_announced:
                    self.speak("You have reached the door. Stop.", priority=True)
                    self.door_announced = True
                return

            # Provide directional guidance
            direction = self.get_direction(door_center_x)

            if direction == "left":
                self.speak("Door is to your left, turn left")
            elif direction == "right":
                self.speak("Door is to your right, turn right")
            else:
                self.speak(f"Door is straight ahead, {door_distance:.1f} meters away")

        elif self.door_detected:
            # Door was detected before but not now
            self.speak("Door lost. Please look around.")
            self.door_announced = False
        else:
            # No door detected
            self.speak("No door detecAVted. Please look around.")

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
                # Get class and confidence
                cls = int(box.cls[0])
                cls_name = self.model.names[cls]
                conf = float(box.conf[0])

                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                bbox = (x1, y1, x2, y2)

                # Check if it's a door
                if cls_name.lower() in [name.lower() for name in self.door_class_names]:
                    # If multiple doors, choose the closest/largest one
                    if not door_detected or (x2 - x1) > (door_bbox[2] - door_bbox[0]):
                        door_detected = True
                        door_bbox = bbox
                        door_distance = self.estimate_distance(x2 - x1)
                        door_center_x = (x1 + x2) // 2

                # Check if it's an obstacle - any object that's not a door
                if cls_name.lower() not in [name.lower() for name in self.door_class_names]:
                    # Calculate distance to obstacle
                    obstacle_width = x2 - x1
                    obstacle_distance = self.estimate_distance(obstacle_width)

                    # Store obstacle information
                    obstacle_info = {
                        'bbox': bbox,
                        'class': cls_name,
                        'confidence': conf,
                        'distance': obstacle_distance,
                        'center_x': (x1 + x2) // 2,
                        'center_y': (y1 + y2) // 2
                    }
                    obstacles.append(obstacle_info)
                    # Detected obstacle (no print to keep console clean)

        # Always add a simulated obstacle for testing if no real obstacles detected
        if not obstacles:
            # Create a simulated obstacle for testing
            sim_x1, sim_y1 = self.frame_width // 4, self.frame_height // 3
            sim_x2, sim_y2 = sim_x1 + 100, sim_y1 + 200
            sim_distance = 2.5

            # Add to obstacles list
            obstacles.append({
                'bbox': (sim_x1, sim_y1, sim_x2, sim_y2),
                'class': 'simulated_obstacle',
                'confidence': 0.95,
                'distance': sim_distance,
                'center_x': (sim_x1 + sim_x2) // 2,
                'center_y': (sim_y1 + sim_y2) // 2
            })
            # Added simulated obstacle (no print to keep console clean)

        # Draw detections on frame
        annotated_frame = results[0].plot()

        # Create obstacle map and find path if door detected
        if door_detected and obstacles:
            obstacle_map = self.create_obstacle_map(obstacles)

            # Find path from bottom center to door
            start_point = (self.frame_width // 2, self.frame_height - 10)
            goal_point = (door_center_x, door_bbox[1] + (door_bbox[3] - door_bbox[1]) // 2)

            self.find_path(start_point, goal_point, obstacle_map)

            # Draw path on frame
            annotated_frame = self.draw_path(annotated_frame)

        # Draw depth map (small overlay)
        depth_small = cv2.resize(self.depth_map, (self.frame_width // 4, self.frame_height // 4))
        depth_color = cv2.applyColorMap(depth_small, cv2.COLORMAP_JET)

        # Place depth map in top-right corner
        h, w = depth_color.shape[:2]
        annotated_frame[10:10+h, self.frame_width-10-w:self.frame_width-10] = depth_color

        # Draw distance if door detected
        if door_detected and door_distance is not None:
            x1, y1, x2, y2 = door_bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Draw distance text
            cv2.putText(
                annotated_frame,
                f"{door_distance:.2f}m",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Draw navigation arrow
            annotated_frame = self.draw_navigation_arrow(annotated_frame, door_center_x)

        # Draw obstacles
        for obstacle in obstacles:
            x1, y1, x2, y2 = obstacle['bbox']

            # Draw distance text for close obstacles
            if obstacle['distance'] < self.obstacle_warning_threshold:
                cv2.putText(
                    annotated_frame,
                    f"{obstacle['class']}: {obstacle['distance']:.2f}m",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

                # Highlight dangerous obstacles
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

        # Draw navigation status
        status_text = "Navigating" if self.navigating else "Standby"
        cv2.putText(
            annotated_frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Store frames for display
        self.display_frames['main'] = annotated_frame
        self.display_frames['depth'] = cv2.applyColorMap(self.depth_map, cv2.COLORMAP_JET)
        self.display_frames['left'] = self.left_view
        self.display_frames['right'] = self.right_view

        # Create obstacle-only view - use a copy of the frame for better context
        obstacle_view = frame.copy()

        # Add a semi-transparent overlay to make obstacles stand out
        overlay = np.zeros_like(frame)

        for obstacle in obstacles:
            x1, y1, x2, y2 = obstacle['bbox']

            # Draw filled rectangle on overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

            # Draw rectangle on main view
            cv2.rectangle(obstacle_view, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Draw distance and class
            cv2.putText(
                obstacle_view,
                f"{obstacle['class']}: {obstacle['distance']:.2f}m",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # Draw direction arrow from bottom center to obstacle
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            start_point = (self.frame_width // 2, self.frame_height - 30)
            end_point = (center_x, center_y)

            # Draw arrow
            cv2.arrowedLine(obstacle_view, start_point, end_point, (0, 255, 255), 2)

            # Add distance text along the arrow
            mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
            cv2.putText(
                obstacle_view,
                f"{obstacle['distance']:.1f}m",
                mid_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # Determine direction (left, right, center)
            direction = self.get_direction(center_x)

            # Add direction text
            cv2.putText(
                obstacle_view,
                f"Go {direction.upper()}",
                (self.frame_width // 2 - 80, self.frame_height - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )

        # If still no obstacles, show a message
        if not obstacles:
            cv2.putText(
                obstacle_view,
                "NO OBSTACLES DETECTED",
                (self.frame_width // 2 - 150, self.frame_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
        else:
            # Make the obstacle warning more prominent
            cv2.putText(
                obstacle_view,
                "CAUTION: OBSTACLES DETECTED",
                (self.frame_width // 2 - 200, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        # Blend overlay with main view
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, obstacle_view, 1 - alpha, 0, obstacle_view)

        # Add title
        cv2.putText(
            obstacle_view,
            "OBSTACLE DETECTION",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        self.display_frames['obstacles'] = obstacle_view

        # Create door-only view - use a copy of the frame for better context
        door_view = frame.copy()

        # Add a semi-transparent overlay
        overlay = np.zeros_like(frame)

        if door_detected:
            x1, y1, x2, y2 = door_bbox

            # Draw filled rectangle on overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)

            # Draw rectangle on main view
            cv2.rectangle(door_view, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw distance
            cv2.putText(
                door_view,
                f"Door: {door_distance:.2f}m",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            # Draw direction arrow from bottom center to door
            start_point = (self.frame_width // 2, self.frame_height - 30)
            end_point = (door_center_x, (y1 + y2) // 2)

            # Draw arrow
            cv2.arrowedLine(door_view, start_point, end_point, (0, 255, 255), 3)

            # Add direction text
            direction = self.get_direction(door_center_x)
            cv2.putText(
                door_view,
                f"Go {direction.upper()}",
                (self.frame_width // 2 - 80, self.frame_height - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )
        else:
            # No door detected
            cv2.putText(
                door_view,
                "NO DOOR DETECTED",
                (self.frame_width // 2 - 150, self.frame_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        # Blend overlay with main view
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, door_view, 1 - alpha, 0, door_view)

        # Add title
        cv2.putText(
            door_view,
            "DOOR DETECTION",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        self.display_frames['door'] = door_view

        return annotated_frame, door_detected, door_distance, door_bbox, door_center_x, obstacles

    def _listen_for_commands(self):
        """Listen for voice commands in a loop."""
        if not self.microphone:
            print("Microphone not available. Voice commands disabled.")
            return

        self.listening = True

        while self.listening:
            try:
                with self.microphone as source:
                    print("Listening for commands...")
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

                try:
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"Recognized: {command}")

                    # Process command
                    if "take me to the door" in command or "find door" in command:
                        self.start_navigation()
                    elif "stop" in command:
                        self.stop_navigation()
                    elif "where is the door" in command:
                        if self.door_detected and self.door_distance is not None:
                            direction = self.get_direction(self.door_center_x)
                            self.speak(f"Door is {direction}, {self.door_distance:.1f} meters away")
                        else:
                            self.speak("No door detected")
                    elif "what's in front of me" in command or "what is in front of me" in command:
                        if self.obstacles:
                            close_obstacles = [o for o in self.obstacles if o['distance'] < self.obstacle_warning_threshold]
                            if close_obstacles:
                                obstacle_names = [o['class'] for o in close_obstacles[:3]]  # Limit to 3 obstacles
                                self.speak(f"I see {', '.join(obstacle_names)} in front of you")
                            else:
                                self.speak("Path is clear")
                        else:
                            self.speak("Path is clear")

                except sr.UnknownValueError:
                    # Speech was unintelligible
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")

            except Exception as e:
                print(f"Listening error: {e}")
                time.sleep(0.1)

    def start_voice_recognition(self):
        """Start voice command recognition."""
        if self.listening:
            return

        self.listen_thread = threading.Thread(target=self._listen_for_commands, daemon=True)
        self.listen_thread.start()

        # Announce that the system is ready
        self.speak("Voice commands are now active. Say 'take me to the door' to start navigation.")

    def stop_voice_recognition(self):
        """Stop voice command recognition."""
        self.listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=1.0)
            self.listen_thread = None

    def process_loop(self):
        """Main processing loop."""
        try:
            self.initialize_camera()

            while self.running:
                # Read frame
                if self.using_sample_images:
                    # Use sample images instead of camera
                    if self.using_sample_image_paths:
                        # Load image from file
                        frame = cv2.imread(self.sample_images[self.current_sample_index])
                        if frame is None:
                            print(f"Error: Failed to load sample image {self.sample_images[self.current_sample_index]}")
                            time.sleep(0.1)
                            continue
                    else:
                        # Use pre-loaded image
                        frame = self.sample_images[self.current_sample_index]

                    # Resize to match desired frame size
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))

                    # Move to next sample image every 30 frames
                    if self.frame_count % 30 == 0:
                        self.current_sample_index = (self.current_sample_index + 1) % len(self.sample_images)

                    ret = True
                else:
                    # Use camera
                    ret, frame = self.cap.read()

                    if not ret:
                        print("Error: Failed to capture frame")
                        time.sleep(0.1)
                        continue

                # Process only every n-th frame for better performance
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames != 0:
                    # Skip processing but don't show camera view
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
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

                # Display frames
                # Main view
                cv2.imshow("Vision Guard - Main", self.display_frames['main'])
                display_h, display_w = 300, 400  # Larger size for better visibility

                # Create a combined left-right view (full width)
                stereo_view = np.hstack((self.display_frames['left'], self.display_frames['right']))

                # Resize all views to the same size
                depth_view = cv2.resize(self.display_frames['depth'], (display_w, display_h))
                stereo_view = cv2.resize(stereo_view, (display_w*2, display_h))  # Double width for stereo view
                door_view = cv2.resize(self.display_frames['door'], (display_w, display_h))

                # Create a simplified 2-row grid
                top_row = np.hstack((depth_view, door_view))  # Show depth and door detection

                # Create a 2-row grid: top row and stereo view
                grid = np.vstack((top_row, stereo_view))

                # Add labels with larger font and better visibility
                cv2.putText(grid, "Depth Map", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(grid, "Door Detection", (display_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                cv2.imshow("Vision Guard - Analysis", grid)

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

    def start_navigation(self):
        """Start navigation to the door."""
        if not self.navigating:
            self.navigating = True
            self.door_announced = False
            self.obstacle_announced = False
            self.speak("Starting navigation to the door. Please move slowly.", priority=True)

    def stop_navigation(self):
        """Stop navigation."""
        if self.navigating:
            self.navigating = False
            self.speak("Navigation stopped.", priority=True)

    def start(self):
        """Start the Vision Guard system."""
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
        """Stop the Vision Guard system."""
        self.running = False
        self.navigating = False

        # Stop voice recognition
        self.stop_voice_recognition()

        # Wait for processing thread to finish
        if self.process_thread:
            self.process_thread.join(timeout=1.0)

        print("Vision Guard stopped")


def main():
    """Main function."""
    # Create Vision Guard system
    vision_guard = VisionGuard(
        model_path='runs/train/yolov8_door_detection/weights/best.pt',
        camera_id=0,
        confidence=0.4,
        door_class_names=['door', 'Door', 'hinged', 'knob', 'lever'],
        frame_width=800,  # Increased resolution for better visibility
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
    
