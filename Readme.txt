###############finalVersion#################
detection		--Store the results of picture detection
detecttion_video	--Store the results of video detection
font--
logs--backup_txt_map	--Store the test information
    --video		--Store videos to be detected
    --vol_images_set	--Store images to be detected
    			--Store model files
model_data		--Store training sample categories and prior box size information
utils			--Store json to txt program code
     			--Sample distribution statistics before and after data enhancement
     			--Sample statistics program(statistics_utils)
     			--txt to xml program code
    			--xml to txt program code
     			--Data Enhancement Code(XMLaug_utils)
yolo3			--Stores Yolov3's network structure model program code
     			--Stores some functions used to build the model(utils)
convert			--
kmeans			--Clustering Algorithm
models			--Linker to store mongodb database
train_policy		--Store training strategy program
user_predict		--Store image detection program(where we input the image to be detected)
voc_annotation		--Make training samples based on data in voc format
yolo			--Stores the detection interface (detects the underlying functions of the video and image)
yolo_video		--Store video detection program(where we input the video to be detected)
###############mAP-master#################
input--detection-results--Test results, txt text required
     --ground-truth	--Image tag file, need txt text
     --images-optional	--Detected image (optional)
results			--mAP calculation results
scripts			--File format converter
main			--mAP calculation main program
###############VOCdevkit#################
Annotations		--Image tag file (xml)
backup			--backup file
ImageSets		--Store text messages to be trained
JPEGImages		--Store image dataset
test			--The data set is converted into a training sample, a validation set, and a test set
###############mongodb#################
			--mongodb database environment
###############robo 3t#################
.exe			--MongoDB database visualization software
