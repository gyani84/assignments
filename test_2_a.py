import cv2 
import numpy as np
import glob
import random
import matplotlib.pyplot as plt
from numpy.core.numeric import outer


#Termination criteria for iterative least-squares estimation
TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#Number of inner corners
PATTERN_SIZE = (7,5)

def Task_1_A_extarct_and_display_corner_to_pixel_accuracy():
    #All PNG images staring with Assignment_MV_02*
    caliberation_images = glob.glob('Assignment_MV_02*.png')
    corner_2D_cordinates = []    
    IMAGE_INDEX = 0 #hold image count
    image_list = []
    for item in caliberation_images:
        bgr_image = cv2.imread(item)
        #convert to gray scale
        gray_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2GRAY)
        ret_status, conrner_point_list = cv2.findChessboardCorners(gray_image, PATTERN_SIZE, None)
        if ret_status == True:
            #Calculate subpixel accuracy
            subpixel_accurate_corner_point_list = cv2.cornerSubPix(gray_image,conrner_point_list,(15,11),(-1,-1),TERMINATION_CRITERIA)
            corner_2D_cordinates.append(subpixel_accurate_corner_point_list)
            # Draw and display the corners
            marked_image = cv2.drawChessboardCorners(bgr_image, (7,5), subpixel_accurate_corner_point_list,ret_status)
            IMAGE_INDEX+=1
            image_list.append(gray_image)
            cv2.imshow('IMAGE:'+str(IMAGE_INDEX),marked_image)
    cv2.waitKey(0)
    return corner_2D_cordinates, image_list

def Task_1_B_camera_caliberation_matrix(corner_2D_cordinates,image_list):
    """
    Need 3D point(real world) & 2D point(image) for cv2.calibrateCamera
    corner_2D_cordinates is 2d points extracted from image
    For 3D real world point, lets assume checkerboard starts from origin and as its plane surface, consider z = 0
    so corner co-ordinate would be (0,0),(0,1)..(6,4) ->total 35 points
    """
    real_world_points=[]
    #initialised with zero total 35 3d points
    points = np.zeros((7*5,3), np.float32)
    points[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
    #Multiple image needed for camera caliberation. Here 6 images are provided. so needed 3d points for all images
    for item in corner_2D_cordinates:
        real_world_points.append(points)
    #calibrateCamera
    ret_val, caliberation_matrix, radial_distortion, rotation_vector, camera_position = \
        cv2.calibrateCamera(real_world_points, corner_2D_cordinates, image_list[0].shape[::-1],None,None)

    print("Camera Caliberation Matrix (K):")
    print(caliberation_matrix)
    
    #Principle Length [0][0]
    print("Principle Length:",caliberation_matrix[0][0])
    #Aspect ration [1][1]/[0][0]
    print("Aspect ratio:",caliberation_matrix[1][1]/caliberation_matrix[0][0])
    #Principle point(x0,y0) ([0][2],[1][2])
    print("Principle Point(x0,y0):",(caliberation_matrix[0][2],caliberation_matrix[1][2]))
    return caliberation_matrix

def get_frame_from_video_ex():
    video = cv2.VideoCapture("Assignment_MV_02_video.mp4")
    counter = 0
    frame_list = []

    while video.isOpened():
        ret,farme_image = video.read()
        if not ret:
            break
        frame_list.append(farme_image)       
    
    video.release()
    
    return frame_list


def get_frame_from_video():
    video = cv2.VideoCapture("Assignment_MV_02_video.mp4")
    counter = 0
    frame_list = []

    while video.grab():
        ret,farme_image = video.retrieve()
        if not ret:
            break
        frame_list.append(farme_image)       
    
    video.release()
    cv2.destroyAllWindows()
    return frame_list

def get_gray_frame_list(frame_list):
    #Get gray image frame from frame list
    #frame_list is in RGB
    gray_frame_list = []
    for item in frame_list:
        gray_frame_list.append(cv2.cvtColor(item,cv2.COLOR_BGR2GRAY))
    return gray_frame_list

def Task_1_C_good_feature_track_in_first_frame(gray_frame_image):
    #Extract good feature to track in first frame with subpixel accuracy
    features_to_track = cv2.goodFeaturesToTrack(gray_frame_image, 200, 0.3, 7) 
    refined_feature_points = cv2.cornerSubPix(gray_frame_image, features_to_track, (11,11), (-1,-1), TERMINATION_CRITERIA)
    return refined_feature_points


def get_correspondance_point_list(gray_frame_list,frame_list):

    p0 = cv2.goodFeaturesToTrack(gray_frame_list[0], 200, 0.3, 7)  
    p0 = cv2.cornerSubPix(gray_frame_list[0], p0, (11,11), (-1,-1), TERMINATION_CRITERIA)  

    # initialise tracks
    index = np.arange(len(p0))
    tracks = {}
    for i in range(len(p0)):
        tracks[index[i]] = {0:p0[i]}


    for frame in range(1,len(gray_frame_list)):
        p1, st, err  = cv2.calcOpticalFlowPyrLK(gray_frame_list[frame-1], gray_frame_list[frame], p0, None)   
        p1 = cv2.cornerSubPix(gray_frame_list[frame-1], p1, (11,11), (-1,-1), TERMINATION_CRITERIA)


        p0 = p1[st==1].reshape(-1,1,2)            
        index = index[st.flatten()==1]

        # update tracks
        for i in range(len(p0)):
            if index[i] in tracks:
                tracks[index[i]][frame] = p0[i]
            else:
                tracks[index[i]] = {frame: p0[i]}



   
    # Display the feature track
    # Get first frame of the viedo 
    first_frame = frame_list[0].copy()
    frame = len(frame_list)
    for i in range(len(index)):
            for f in range(0,frame):
                if (f in tracks[index[i]]) and (f+1 in tracks[index[i]]):
                    cv2.line(first_frame,
                             (tracks[index[i]][f][0,0],tracks[index[i]][f][0,1]),
                             (tracks[index[i]][f+1][0,0],tracks[index[i]][f+1][0,1]), 
                             (0,255,0), 1)            
    cv2.imshow("Feature_Tracks", first_frame)   
    cv2.waitKey(0)

    correspondences = []
    corr_p1 = []
    corr_p2 = []
    for track in tracks:
        if(len(tracks[track])==31):            
            x1 = [tracks[track][0][0,0],tracks[track][0][0,1],1]
            x2 = [tracks[track][30][0,0],tracks[track][30][0,1],1]
            correspondences.append((np.array(x1), np.array(x2)))
            corr_p1.append(np.array(x1))
            corr_p2.append(np.array(x2))
    corr_p1 = np.array(corr_p1).flatten().reshape(-1,3)
    corr_p2 = np.array(corr_p2).flatten().reshape(-1,3)
    print("Total Correspoindance Point:",len(correspondences))

    
    return corr_p1,corr_p2, index, tracks

def get_T_val(mean,devi):
    return np.array(
        [
        [1/devi[0],  0 ,         -mean[0]/devi[0]],
        [0,          1/devi[1],  -mean[1]/devi[1]],
        [0 ,         0 ,           1             ]
        ], dtype=np.float)

def get_normalised_feature_coordinates(point_x1,point_x2):
    # x1 points from 1st frame x2 points from last frame    
    x1_mean = np.mean(point_x1[:,0])
    y1_mean = np.mean(point_x1[:,1])
    #Get Std deviation x1
    x1_std = np.std(point_x1[:,0])
    y1_std = np.std(point_x1[:,1])

    #Get mean x1
    x2_mean = np.mean(point_x2[:,0])
    y2_mean = np.mean(point_x2[:,1])
    #Get Std deviation x2
    x2_std = np.std(point_x2[:,0])
    y2_std = np.std(point_x2[:,1])



    T1 = get_T_val([x1_mean,y1_mean],[x1_std,y1_std])
    T2 = get_T_val([x2_mean,y2_mean],[x2_std,y2_std])


    point_x1_1 = np.matmul(T1,point_x1.T).T
    point_x2_1 = np.matmul(T2,point_x2.T).T


    return point_x1_1,point_x2_1,T1,T2

def select_eight_correspondance_random(point_x1,point_x2):
    random_index = np.random.randint(0,len(point_x1),8)
    #Selected points
    selected_point_x1 = point_x1[random_index]
    selected_point_x2 = point_x2[random_index]

    return selected_point_x1,selected_point_x2

def run_eight_point_dlt(point_x1,point_x2,T1,T2):
    A = np.zeros((0,9))
    for x1,x2 in zip(point_x1,point_x2):
        ai = np.kron(x1.T,x2.T)
        A = np.append(A,[ai],axis=0) 

    U,S,V = np.linalg.svd(A)    
    F = V[8,:].reshape(3,3).T
    #Singularity
    U,S,V = np.linalg.svd(F)
    #Fundamental Matrix
    F_Bar = np.matmul(U,np.matmul(np.diag([S[0],S[1],0]),V))
    F = T2.T*F_Bar*T1

    return F

def get_observance_covar_matrix():
    return np.array([[1 , 0 , 0],[0, 1, 0],[0, 0, 0]])

cxx = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

W = np.array([[0, -1, 0], 
              [1, 0,  0], 
              [0, 0,  1]])

Z = np.array([[0, 1, 0], 
              [-1,0, 0], 
              [0, 0, 0]])

def main_alogorithm(norm_point_x1,norm_point_x2,T1,T2,point_x1,point_x2):

    selected_points_index = set(random.sample(range(len(norm_point_x1)),8))
    remaining_points_index = set(range(len(norm_point_x1))).difference(selected_points_index)
    selected_points_index = list(selected_points_index)
    remaining_points_index = list(remaining_points_index)

    ####random_index = np.random.randint(0,len(point_x1),8)
    #Selected points
    ####selected_point_x1 = norm_point_x1[random_index]
    ####selected_point_x2 = norm_point_x2[random_index]

    selected_point_x1 = norm_point_x1[selected_points_index]
    selected_point_x2 = norm_point_x2[selected_points_index]

    A = np.zeros((0,9))
    for x1,x2 in zip(selected_point_x1,selected_point_x2):
        ai = np.kron(x1.T,x2.T)
        A = np.append(A,[ai],axis=0) 

    U,S,V = np.linalg.svd(A)    
    F = V[8,:].reshape(3,3).T
    #Singularity
    U,S,V = np.linalg.svd(F)
    #Fundamental Matrix
    F_Bar = np.matmul(U,np.matmul(np.diag([S[0],S[1],0]),V))
    F = T2.T @ F_Bar @ T1

    cxx = get_observance_covar_matrix()
    gi_list = []
    sigma_sq_list = []

    # Calculating value of model equation for remaining correspondance points
    #
    for index in remaining_points_index:
        x1 = point_x1[index].reshape((3,1))
        x2 = point_x2[index].reshape((3,1))
        gi =  x2.T @ F @ x1
        #sigma_sq = x2.T @ F.T @ cxx @ F @ x2 + x1.T @ F @ cxx @ F.T @ x1  
        sigma_sq = x2.T @ F @ cxx @ F.T @ x2 + x1.T @ F.T @ cxx @ F @ x1  

        gi_list.append(gi**2)
        sigma_sq_list.append(sigma_sq)
    
    
    Ti = np.array(gi_list)/np.array(sigma_sq_list)

    Ti = Ti.flatten()

    inlier_sum = np.sum(Ti[Ti<= 6.635])
    #outliers = (Ti>6.635)
    #outlier_count = np.sum(outliers==True)

    outlier_index = np.array(remaining_points_index,dtype=int)[(Ti>6.635)]
    outlier_number = outlier_index.shape[0]
    return F, inlier_sum, outlier_index,outlier_number


def get_final_matrix(norm_point_x1,norm_point_x2,T1,T2,point_x1,point_x2):

    final_F = None
    final_outlier = None
    final_inlier_sum = 0
    final_outlier_count = point_x2.shape[0]+1

    for _ in range(0,10000):

        F, inlier_sum, outliers,outlier_count = main_alogorithm(norm_point_x1,norm_point_x2,T1,T2,point_x1,point_x2)

        if final_outlier_count> outlier_count or  (final_outlier_count == outlier_count and inlier_sum> final_inlier_sum):
            final_F = F
            final_outlier = outliers
            final_inlier_sum = inlier_sum
            final_outlier_count = outlier_count 

    return final_F, final_inlier_sum,final_outlier,final_outlier_count

def calculate_epipoles(F):
    U,S,V = np.linalg.svd(F)    
    e1 = V[2,:]

    U,S,V = np.linalg.svd(F.T)    
    e2 = V[2,:]

    return e1,e2   

def show_outlier_inlier(corr_index,outlier_index,frame_list,e1,e2,tracks):
    last_frame = frame_list[30].copy()
    
    frame = len(frame_list)
    for i in range(len(corr_index)):
            for f in range(0,frame):
                if (f in tracks[corr_index[i]]) and (f+1 in tracks[corr_index[i]]):

                    if corr_index[i] in outlier_index:
                        cv2.line(last_frame,
                                (tracks[corr_index[i]][f][0,0],tracks[corr_index[i]][f][0,1]),
                                (tracks[corr_index[i]][f+1][0,0],tracks[corr_index[i]][f+1][0,1]), 
                                (0,0,255), 5)  
                                          
                    else:
                        cv2.line(last_frame,
                                (tracks[corr_index[i]][f][0,0],tracks[corr_index[i]][f][0,1]),
                                (tracks[corr_index[i]][f+1][0,0],tracks[corr_index[i]][f+1][0,1]), 
                                (0,255,0), 2)            
    
    e1_int= e1.astype(int)                    
    e2_int= e2.astype(int)                    
    cv2.circle(last_frame, (e1_int[0],e1_int[1]), 15, (255, 255, 0), -1 )
    cv2.circle(last_frame, (e2_int[0],e2_int[1]), 10, (0, 255, 255), -1 )
    
    cv2.imshow("Outlier_inlier_tracks_with_epi_points", last_frame)   

def get_essesential_matrix(K,F):

    E_cap = np.linalg.inv(K)@F@K
    U,S,V = np.linalg.svd(E_cap)

    S[2] = 0
    # lamda = (lamda1+lamda2)/2
    avg_s = (S[0]+S[1])/2
    S[0] = avg_s
    S[1] = avg_s

    if np.linalg.det(U) <0:
        U[:,2] *=-1     

    if np.linalg.det(V) <0:
        V[2,:] *=-1   
    
    final_E = U @ np.diag(S) @ V.T

    return final_E,U,S,V

def get_rotation_matrix(U,W,V):
    return U@W@V.T, U@W.T@V.T

def get_translation_matrix(U,Z,beeta):

    T1 = beeta * (U @ Z @ U.T)
    T2 = -1.0 * T1

    #Last colums is t
    return T1[:,2].reshape(3,1),T2[:,2].reshape(3,1)

def get_beta(total_frame_count):

    time = total_frame_count/30
    distance = time* 50* (1000/(60*60))

    return distance

def get_lamda_mu(m1,m2,r,t):
    
    numerator = np.array([[t.T @ m1],
                          [t.T @ r @ m2]])


    denumerator = np.array([[m1.T@m1, -1* (m1.T @ r @ m2)],
                            [m1.T @ r @ m2, -1 * (m2.T@m2)]])

    solution = np.linalg.inv(denumerator) @ numerator.reshape(2,1)
    solution = solution.flatten()
    return solution[0],solution[1]

def get_3d_point(lamda_mu, R, t, inlier_m1_m2):

    points = []
    for index in range(len(inlier_m1_m2)):
        lamda = lamda_mu[index][0]
        mu = lamda_mu[index][1]
        m1 = inlier_m1_m2[index][0]
        m2 = inlier_m1_m2[index][1]
       
        x_lamda = lamda * m1
        x_mu = t + mu * (R @ m2)

        pt = (x_lamda+x_mu)/2
        points.append(pt)

    return points

def plot_3d_points_camera_centers(C1,C2,points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(C1[0], C1[1], C1[2], marker = "^")
    ax.scatter(C2[0], C2[1], C2[2], marker = "^")

    for x,y,z in points_3d:
        ax.scatter(x, y, z, marker = "+")

    plt.show()

def get_reprojected_points(points_3d,R,t,K):

    reprojected_x = []
    for point in points_3d:
        x1 = K@point
        x2 = K @ R.T @ (point - t)

        x1 = (x1[0]/x1[2],x1[1]/x1[2],1)
        x2 = (x2[0]/x2[2],x2[1]/x2[2],1)
        reprojected_x.append((x1,x2))
    
    return reprojected_x

def main():

    random.seed(10)
    np.random.seed(10)

    corner_2D_cordinates, image_list = Task_1_A_extarct_and_display_corner_to_pixel_accuracy()
    calib_matrix = Task_1_B_camera_caliberation_matrix(corner_2D_cordinates, image_list)
    frame_list = get_frame_from_video()
    gray_frame_list = get_gray_frame_list(frame_list)
    #Task_1_C_good_feature_track_in_first_frame(gray_frame_list[0])
    point_x1,point_x2, corres_index, tracks = get_correspondance_point_list(gray_frame_list,frame_list)
    norm_point_x1,norm_point_x2,T1,T2 = get_normalised_feature_coordinates(point_x1,point_x2)
    
    final_F, final_inlier_sum,final_outlier,final_outlier_count = get_final_matrix(norm_point_x1,norm_point_x2,T1,T2,point_x1,point_x2)
 
    

    e1,e2 = calculate_epipoles(final_F)
    print(e1/e1[2])    
    print(e2/e2[2]) 
    e1 = np.divide(e1, e1[2])
    e2 = np.divide(e2, e2[2])
    
    show_outlier_inlier(corres_index,final_outlier,frame_list,e1,e2,tracks)
    
    cv2.waitKey(0)

    #Remove outlier point from correspondance
    filter = np.full(norm_point_x1.shape[0],True)
    filter[final_outlier] = False

    final_point_x1 = point_x1[filter==True]
    final_point_x2 = point_x2[filter==True]

    direction_m1= []
    direction_m2= []
    for itr in range(final_point_x1.shape[0]):
        m1 = np.linalg.inv(calib_matrix) @ final_point_x1[itr]        
        m2 = np.linalg.inv(calib_matrix) @ final_point_x2[itr]

        direction_m1.append(m1)
        direction_m2.append(m2)

  


    E,U,S,V = get_essesential_matrix(calib_matrix,final_F)
    r1, r2 = get_rotation_matrix(U,W,V)
    beeta = get_beta(len(frame_list))
    t1,t2 = get_translation_matrix(U,Z,beeta)

    print("\nR1=",r1)
    print("\nR2=",r2)
    print("\nt1=",t1)
    print("\nt2=",t2)
    rt_combination = [(r1,t1),(r2,t2),(r1,t2),(r2,t1)]
    positive_count=[0,0,0,0]
    
    
    for index in range(len(rt_combination)):
        r,t = rt_combination[index]

        for m1,m2 in zip(direction_m1,direction_m2):
            lamda,mu = get_lamda_mu(m1,m2,r,t)
            if lamda>0 and mu >0:
                positive_count[index] = positive_count[index] + 1
    
    
    max_positive_count= max(positive_count)
    rt_selected_index = positive_count.index(max_positive_count)
    R,t = rt_combination[rt_selected_index]

    print("\nR=",R)
    print("\nt=",t)
    
    lamda_mu =[]
    inlier_m1_m2 = []        
    corresp_x1_x2 = []

    for index in range(len(direction_m1)):
        lamda,mu = get_lamda_mu(direction_m1[index],direction_m2[index],R,t)        
        if lamda>0 and mu>0:
            inlier_m1_m2.append((direction_m1[index],direction_m2[index]))
            lamda_mu.append((lamda,mu))
            corresp_x1_x2.append((point_x1[index],point_x2[index]))
        
    
    points_3d = get_3d_point(lamda_mu, R, t, inlier_m1_m2)
    camera_c1 = np.array([0,0,0])
    camera_c2 = t
    plot_3d_points_camera_centers(camera_c1,camera_c2,points_3d)

    reprojected_points = get_reprojected_points(points_3d,R,t,calib_matrix)
    
    cv2.destroyWindow("Feature_Tracks")
    cv2.destroyWindow("Outlier_inlier_tracks_with_epi_points")

main()

