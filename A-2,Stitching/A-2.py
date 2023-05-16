import os
import cv2
import numpy as np
from PIL import Image

def execute(img1,img2):

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    # Define Hessian corner detector function
    def hessian_corner_detector(img, threshold=999999, window_size=3):
        dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        dxx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=3)
        dxy = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=3)
        dyy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=3)
        H = np.zeros_like(img, dtype=np.float32)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                H[y,x] = dxx[y,x]*dyy[y,x] - dxy[y,x]*dxy[y,x]
        corners = []
        for y in range(window_size, img.shape[0]-window_size):
            for x in range(window_size, img.shape[1]-window_size):
                if H[y,x] > threshold and np.max(H[y-window_size:y+window_size+1, x-window_size:x+window_size+1]) == H[y,x]:
                    corners.append((x,y))
        nms_corners = []
        for i in range(len(corners)):
            xi, yi = corners[i]
            is_maximum = True
            for j in range(len(corners)):
                if i != j:
                    xj, yj = corners[j]
                    distance = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                    if distance < window_size:
                        if H[yi,xi] < H[yj,xj]:
                            is_maximum = False
                            break
            if is_maximum:
                nms_corners.append((xi,yi))
        return nms_corners

    # Detect corners in both images
    corners1 = hessian_corner_detector(gray1)
    corners1=np.array(corners1)

    for corner in corners1:
        x, y = corner
        cv2.circle(gray1, (x, y), 3, (255, 255, 255), -1)

    # Display image with detected corners
    screen_res = 720, 720
    imgc1 = cv2.resize(gray1, screen_res)
    cv2.imshow('corners1', imgc1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    corners2 = hessian_corner_detector(gray2)
    corners2=np.array(corners2)

    for corner in corners2:
        x, y = corner
        cv2.circle(gray2, (x, y), 3, (255, 255, 255), -1)

    # Display image with detected corners
    screen_res = 720, 720
    imgc2 = cv2.resize(gray2, screen_res)
    cv2.imshow('corners2', imgc2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    corners1=corners1[:200]
    corners2=corners2[:200]

    # Define SSD matching function
    def ssd_score(img1, img2, c1, c2, W):
        matching_score = 0
        for i in range(-W, W+1):
            for j in range(-W, W+1):
                x1, y1 = c1[0]+j, c1[1]+i
                x2, y2 = c2[0]+j, c2[1]+i
                if x1 >= 0 and x1 < img1.shape[1] and y1 >= 0 and y1 < img1.shape[0] and \
                x2 >= 0 and x2 < img2.shape[1] and y2 >= 0 and y2 < img2.shape[0]:
                    matching_score += np.sum((img1[y1][x1] - img2[y2][x2])**2)
        return matching_score


    def implement_matching(img1, img2, corners1, corners2, R, SSDth, ratio=0.9):
        matching = []
        ssd = np.zeros((len(corners1), len(corners2)), dtype=np.float32)

        for i in range(len(corners1)):
            for j in range(len(corners2)):
                ssd[i][j] = ssd_score(img1, img2, corners1[i], corners2[j], R)
            
            # get the indices of the two best matches for corner i
            best_matches = np.argsort(ssd[i])[:2]
            
            # calculate the ratio of the SSD values
            ratio_val = ssd[i][best_matches[0]] / ssd[i][best_matches[1]]
            
            if ssd[i][best_matches[0]] < SSDth and ratio_val < ratio:
                matching.append((corners1[i], corners2[best_matches[0]]))
            
            # set the SSD value of the second best match to a large value to prevent it from being selected again
            ssd[i][best_matches[1]] *= 100
            
        return matching

    R = 3
    SSDth = 20000
    matches = implement_matching(img1, img2, corners1, corners2, R, SSDth)
  

    src_points = np.array([m[0] for m in matches])
    dst_points = np.array([m[1] for m in matches])


    match_img = np.concatenate((img1, img2), axis=1)
    for i in range(len(matches)):
        
        x1, y1 = src_points[i]
        cv2.circle(match_img, (x1,y1), 10,(0,0, 255), -1)
        x2, y2  = dst_points[i]
        x2 += img1.shape[1]
        cv2.circle(match_img, (x2, y2), 10, (0, 255, 255), -1)
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        cv2.line(match_img, (x1, y1), (x2, y2), color, 2)
    cv2.imshow("matches", match_img)
    cv2.waitKey(0)

    return matches    


def Stitching(img2, img1,H):

    nrows, ncols, _ = np.asarray(img1).shape
    Hinv = np.linalg.inv(H)
    Hinvtuple = (Hinv[0, 0], Hinv[0, 1], Hinv[0, 2], Hinv[1, 0], Hinv[1, 1], Hinv[1, 2])
    Pano = np.asarray(img1.transform((ncols * 3, nrows * 3), Image.AFFINE, Hinvtuple))
    Pano = np.asarray(img1.transform((ncols * 3, nrows * 3), Image.AFFINE, Hinvtuple)).copy()

    Pano.setflags(write=1)
    

    Hinv = np.linalg.inv(np.eye(3))
    Hinvtuple = (Hinv[0, 0], Hinv[0, 1], Hinv[0, 2], Hinv[1, 0], Hinv[1, 1], Hinv[1, 2])
    AddOn = np.asarray(img2.transform((ncols * 3, nrows * 3), Image.AFFINE, Hinvtuple))
    AddOn = np.asarray(img2.transform((ncols * 3, nrows * 3), Image.AFFINE, Hinvtuple)).copy()
    AddOn.setflags(write=1)
    

    result_mask = np.sum(Pano, axis=2) != 0
    temp_mask = np.sum(AddOn, axis=2) != 0
    add_mask = temp_mask | ~result_mask
    for c in range(Pano.shape[2]):
        cur_im = Pano[:, :, c]
        temp_im = AddOn[:, :, c]
        cur_im[add_mask] = temp_im[add_mask]
        Pano[:, :, c] = cur_im
    

    # Cropping
    boundMask = np.where(np.sum(Pano, axis=2) != 0)
    Pano = Pano[:np.max(boundMask[0]), :np.max(boundMask[1])]


    return Pano



# Set the directory path containing the images to be stitched
dir_path = 'D:/COMPUTER Vision/a2/data2/3'

# Get the filenames of all the images in the directory
filenames = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jpg')]

# Read the images and store them in a list
images = [cv2.imread(f) for f in filenames]
panorama=images[0]


count=0

for i in range(1,len(images)):
    if i!=1:
        panorama=cv2.imread(dir_path+"pano.jpg")
    if count<2:    
        panorama = cv2.resize(panorama, (640, 480))
        count=count+1
    
    images[i] = cv2.resize(images[i], (640, 480))
    matches=execute(panorama,images[i])

    src_points = np.array([m[0] for m in matches])
    dst_points = np.array([m[1] for m in matches])

    A, inliers = cv2.estimateAffinePartial2D(dst_points,src_points, method=cv2.RANSAC,ransacReprojThreshold=5.0)
    warped = cv2.warpAffine(images[i], A, (images[i].shape[1] + panorama.shape[1], panorama.shape[0]))  
    A = np.vstack([A, [0, 0, 1]])

    panorama = Image.fromarray(panorama)
    images[i] = Image.fromarray(images[i])

    Pano = Stitching(panorama,images[i],A)
    cv2.imshow("current_panorama", Pano)
    cv2.imwrite(dir_path+"pano.jpg",Pano)

screen_res = 640, 480
Pano = cv2.resize(Pano, screen_res)
cv2.imshow("Final_Panorama", Pano)
cv2.imwrite(dir_path+"Final_Panorama.jpg",Pano)
cv2.waitKey(0)

