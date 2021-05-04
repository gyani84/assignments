import cv2 as cv
import numpy as np
import glob

from numpy.lib.function_base import append




def extract_frames(numbers):
    video = cv.VideoCapture("Assignment_MV_02_video.mp4")
    counter = 0
    img_list = []
    while video.isOpened():
        ret,img= video.read()
        if not ret:
            break
        if len(numbers)>0:
            counter+=1
            if counter in numbers:
                img_list.append(img)
        else:
            img_list.append(img)        
    video.release()
    return img_list

def extract_first_frame():
    return extract_frames([1])

# termination criteria
#criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40,1)
images = glob.glob('*.png')
objp = np.zeros((7*5,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
index = 0
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,5), None)
   
     # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners,(15,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv.drawChessboardCorners(img, (7,5), corners2,ret)
        index = index+1
        cv.imshow('img'+str(index),img)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)        
print(" Camera matrix:")
print(mtx)
#Principle Length mat[0][0]
#Aspect ration matx[1][1]/mtx[0][0]
#Principle point(x0,y0) mtx[0][2],mtx[1][2]

print("Principle Length:",mtx[0][0])
print("Aspect ratio:",mtx[1][1]/mtx[0][0])
print("Principle Point(x0,y0):",(mtx[0][2],mtx[1][2]))

image_from_video_list = extract_frames([])

image_from_video_1st = image_from_video_list[0]
src_gray = cv.cvtColor(image_from_video_1st, cv.COLOR_BGR2GRAY)
features_to_track = cv.goodFeaturesToTrack(src_gray, 200, 0.3, 7)  


# Set the needed parameters to find the refined feature points
winSize = (5, 5)
zeroZone = (-1, -1)
criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
# Calculate the refined feature points

refined_feature_points = cv.cornerSubPix(src_gray, features_to_track, winSize, zeroZone, criteria)

radius = 1
for i in range(corners.shape[0]):
    cv.circle(image_from_video_1st, (int(refined_feature_points[i,0,0]), int(refined_feature_points[i,0,1])), radius, (0,0,255), cv.FILLED)
cv.imshow("Good Feature",image_from_video_1st)

initial_features_point = features_to_track
remaining_points = features_to_track

# initialise tracks
index = np.arange(len(features_to_track))
tracks = {}
for i in range(len(features_to_track)):
    tracks[index[i]] = {0:features_to_track[i]}


current_image = image_from_video_1st
print("Initial features Point:", initial_features_point)

for itr in range(1,len(image_from_video_list)):

    current_image = cv.cvtColor(image_from_video_list[itr-1], cv.COLOR_RGB2GRAY)
    next_image = cv.cvtColor(image_from_video_list[itr], cv.COLOR_RGB2GRAY)
    if len(features_to_track)>0:
        p1, st, err  = cv.calcOpticalFlowPyrLK(current_image, next_image, features_to_track, None)
        
         # visualise points
        for i in range(len(st)):
            if st[i]:
                cv.circle(next_image, (p1[i,0,0],p1[i,0,1]), 2, (0,0,255), 2)
                cv.line(next_image, (features_to_track[i,0,0],features_to_track[i,0,1]), \
                    (int(features_to_track[i,0,0]+(p1[i][0,0]-features_to_track[i,0,0])*5),int(features_to_track[i,0,1]+(p1[i][0,1]-features_to_track[i,0,1])*5)), \
                         (0,0,255), 2) 
        #cv.imshow(str(itr),next_image)
        features_to_track = p1[st==1].reshape(-1,1,2)
        index = index[st.flatten()==1]

         # update tracks
        for i in range(len(features_to_track)):
            if index[i] in tracks:
                tracks[index[i]][itr+1] = features_to_track[i]
            else:
                tracks[index[i]] = {itr+1: features_to_track[i]}

        remaining_points = remaining_points[st==1].reshape(-1,1,2)
        #refined_feature_points = cv.cornerSubPix(next_image, feature_points, winSize, zeroZone, criteria)

print("Remianing features Point:", remaining_points)
print("Final features Point:", features_to_track)
cv.waitKey(0)