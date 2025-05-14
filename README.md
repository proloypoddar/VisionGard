# Door Navigation Assistant for Visually Impaired

A voice-controlled assistant system that helps visually impaired people navigate to doors safely. The system uses computer vision to detect doors, estimate distances, avoid obstacles, and provide voice guidance.

## Features

- **Door Detection**: Uses YOLOv8 to detect doors and door components (knobs, levers, hinges)
- **Distance Estimation**: Estimates the distance to doors using monocular vision
- **Path Planning**: Creates a virtual path to the door while avoiding obstacles
- **Voice Guidance**: Provides voice commands to guide the user to the door
- **Voice Control**: Responds to voice commands like "take me to the door"
- **Obstacle Avoidance**: Detects and avoids obstacles in the path
- **Proximity Alert**: Announces when the user is close to the door

## System Requirements

### Standard Version
- Python 3.8+
- Webcam or USB camera
- Microphone
- Speakers
- CUDA-capable GPU (recommended for better performance)

### Raspberry Pi Version
- Raspberry Pi 4 (2GB RAM minimum, 4GB recommended)
- Raspberry Pi Camera Module or USB camera
- Microphone
- Speakers or headphones
- Optional: Battery pack for portability

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/door-navigation-assistant.git
   cd door-navigation-assistant
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. For Raspberry Pi, install additional dependencies:
   ```
   sudo apt-get update
   sudo apt-get install espeak
   ```

## Usage

### Standard Version

Run the full-featured door navigation assistant:

```
python door_navigation_assistant.py
```

### Raspberry Pi Version

Run the optimized version for Raspberry Pi:

```
python raspberry_pi_door_assistant.py
```

### Controls

- Press 'q' to quit
- Press 'n' to start navigation
- Press 's' to stop navigation
- Say "take me to the door" to start navigation
- Say "stop" to stop navigation

## Configuration

You can modify the following parameters in the code:

- `model_path`: Path to the trained YOLOv8 model
- `camera_id`: Camera device ID (default: 0)
- `confidence`: Confidence threshold for detections (default: 0.4)
- `door_class_names`: List of class names that represent doors
- `frame_width` and `frame_height`: Camera resolution
- `focal_length`: Focal length of the camera in pixels
- `known_door_width`: Known width of a door in meters (default: 0.9m)

## How It Works

1. **Door Detection**:
   - The system uses a YOLOv8 model trained on door images to detect doors in the camera feed.
   - It identifies door components like knobs, levers, and hinges.

2. **Distance Estimation**:
   - Uses the principle that objects appear smaller as they get farther away.
   - Calculates distance based on the apparent width of the door in the image.

3. **Path Planning**:
   - Creates a virtual path to the door while avoiding obstacles.
   - Uses a simplified potential field approach for path planning.

4. **Voice Guidance**:
   - Provides directional commands (left, right, forward) to guide the user.
   - Announces when the user is close to the door.
   - Responds to voice commands for starting and stopping navigation.

## Optimizations for Raspberry Pi

The Raspberry Pi version includes several optimizations:

1. **Reduced Resolution**: Uses a lower resolution (320x240) to improve performance.
2. **Frame Skipping**: Processes only every 3rd frame to reduce CPU load.
3. **Simplified Path Planning**: Uses a more efficient approach for navigation guidance.
4. **Lightweight TTS**: Uses espeak instead of pyttsx3 for better performance.
5. **Minimal Dependencies**: Reduces the number of required libraries.

## Training Your Own Model

If you want to train your own YOLOv8 model for door detection:

1. Prepare your dataset with labeled door images.
2. Create a data.yaml file with paths to train, validation, and test sets.
3. Run the training script:
   ```
   python train_yolov8.py
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ultralytics for the YOLOv8 implementation
- The open-source computer vision community
- Contributors to the assistive technology field
