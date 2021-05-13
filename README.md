# sportradar_scoreboard
repository for the technical task interview

1) run the script create_dataset.py to create the dataset in the yolov5 format
2) copy the data.yaml file from the main folder to the dataset in order to run yolo
3) run yolov5/train.py to train the network for the scoreboard detection task
4) run yolov5/test.py to test the network performance on test (or validation) set
5) run yolov5/main.py to process the video frame by frame (not exactly all, just the most relevant to save time) to find the scoreboard and to read the content
