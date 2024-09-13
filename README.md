# Real-Time Object Detection with YOLOv8 and Text-to-Speech

This Python program uses the YOLOv8 model for real-time object detection with a webcam feed and announces detected objects using text-to-speech (TTS).

## Features
- Real-time object detection using YOLOv8.
- Displays detected objects with bounding boxes and confidence scores.
- Announces detected objects and their counts every 3 seconds using TTS.
- Supports English voices (Zira or David) if available.

## Requirements
- Python 3.9 (recommended)
- Required libraries listed in `requirements.txt`

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Ahmed5827/Object-Detection.git
    cd real-time-object-detection
    ```

2. **Install the Required Libraries:**

    Install the necessary libraries using the provided `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download YOLOv8 Weights:**

   If the `yolov8n.pt` weights file is not detected, it will be downloaded automatically. However, you can also manually download the YOLOv8 pre-trained weights file `yolov8n.pt` from [Ultralytics](https://github.com/ultralytics/yolov5/releases) and save it in the `yolo-Weights/` directory.

## Usage

1. **Run the Program:**

    Execute the following command to start object detection:

    ```bash
    python app.py
    ```

2. **Controls:**
   - Press `q` to quit the program.

## How It Works

- The program initializes the TTS engine and sets an English voice.
- It then opens the webcam feed and loads the YOLOv8 model.
- The webcam frames are processed to detect objects and draw bounding boxes around them.
- Every 3 seconds, the detected objects and their counts are announced using TTS.
- The feed is displayed in a window that can be closed by pressing the `q` key.

## Troubleshooting

- If the webcam feed does not open, ensure your camera is properly connected and accessible.
- If the TTS engine does not use an English voice, it may not be available on your system; in such cases, the default voice will be used.
- Make sure the YOLOv8 weights file is in the correct directory (`yolo-Weights/`) if automatic download fails.

## Contributing

Feel free to open issues or pull requests if you find bugs or have suggestions for improvements.
