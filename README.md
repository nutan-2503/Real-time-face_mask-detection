# Real-time-face_mask-detection
Real-time face mask detection model

Pre-requisites required:
1. dataset taken from https://drive.google.com/drive/u/0/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG
2. Tensorflow version>=2.2
3. Keras version>=2.4
4. OpenCV

CNN model is created and trained on the dataset diving 96.7% accuracy. This model and their weights is then saved in the system and further used in detect.py for real time detection

### Sample cases and the results from the model:<br>
* With mask:<br>
![image](https://user-images.githubusercontent.com/60135434/124780154-0180ba00-df60-11eb-8047-0a89642dd29f.png)<br>
* No mask:<br>
![image](https://user-images.githubusercontent.com/60135434/124780183-06de0480-df60-11eb-9899-0e146d83a079.png)<br>
* Multiple faces:<br>
![image](https://user-images.githubusercontent.com/60135434/124780223-0f363f80-df60-11eb-8cf8-f8c0fa58f7cc.png)<br>
* Mouth covered with hand (no mask):<br>
![image](https://user-images.githubusercontent.com/60135434/124780244-13625d00-df60-11eb-8c7d-7c1fed410e8a.png)
