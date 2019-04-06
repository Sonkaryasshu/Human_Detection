# Human_Detection
Binary classifier to detect presence of human in a photo
I have used YOLO-COCO(v3) model here, which you can simply download from https://pjreddie.com/media/files/yolov3.weights.
After downloading the weight file, just move weight file in your project directory under yolo-coco folder(Or simply change the file location in code).

For Human detection task, I have performed object detection using yolo and then I checked if any human/person is detected or not?
If any person is detected with the confidance higher than our threshold confidence value then we output that human is detected.

For now, you have to manually change the image location in code, as I will be adding argument parser shorlty.
Also you can use image of any size.
