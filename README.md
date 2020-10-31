# Tensorflow Object Detection

Simple detection model using:
* Sliding Window
* Image Pyramids
* Non maximum suppression

## Setup
Generate Virtual environment
```bash
python3 -m venv ./venv
```
Enter environment
```bash
source venv/bin/activate
```
Install required libraries
```bash
pip install -r requirements.txt
```
Run Program
```bash
python3 main.py --image image.jpg --confidence 0.6
```
