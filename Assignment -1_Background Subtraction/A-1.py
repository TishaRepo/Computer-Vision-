import numpy as np
import cv2
import os
import cv2
import argparse
import os


path="C:/Users/TISHA MADAME/Desktop/CV -data/CAVIAR1/CAVIAR1/input/"
dir_list = os.listdir(path)
dir_list.sort()
frames = []
n = 0
index = 100

# Load the images 
for i in dir_list:
    
    frames.append(cv2.imread(os.path.join(path, dir_list[n])).transpose())
    n=n+1
frames = np.array(frames)
threshold = 15

rows = frames.shape[2]
cols = frames.shape[3]
res = np.zeros((rows, cols))
PI = 3.14159
frames.shape


# Calculate the KDE for each frame
for k in range(1,len(dir_list)):
    for i in range(0,rows):
        for j in range(0,cols):
            # Get the number of frames used for the current pixel
            N = k
            # Calculate the standard deviation
            sig = (0.8 / N)**(3/7)
            # Calculate the divisors for the KDE calculation
            div1 = 1/(N*((2*np.pi)**1.5)*sig)
            div2 = 2*(sig**2)
            # Get the color values for all previous frames for the current pixel
            r = frames[:N,0,i,j]/100
            g = frames[:N,1,i,j]/100
            b = frames[:N,2,i,j]/100 
            # Calculate the KDE value for the current pixel
            res[i][j] = div1 * np.sum(np.exp(-1*((r-r[N-1])*2 + (b-b[N-1])*2 + (g-g[N-1])*2)/div2)/N) 

             # Threshold the KDE results
            if(res[i][j]>=threshold):
                res[i][j]=1
     
     # Transpose the KDE results and convert to grayscale           
    a = (1-res).transpose()* 255

    #Foreground Aggregation
    
    kernel = np.ones((2,2),np.uint8)
    
    img = cv2.dilate(a,kernel)
    
    img = np.uint8(img)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the bounding rectangle on the original image
        img=cv2.rectangle(img, (x, y), (x + w, y + h), (160, 32, 240), 1)

    
        cv2.imwrite("C:/Users/TISHA MADAME/Desktop/CV -data/output/{}.png".format(index), img)
    
    

    
    index+=1


## Generate Video from the output images 


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = 'C:/Users/TISHA MADAME/Desktop/CV -data/output/'
ext = args['extension']
output = args['output']

images = []
for f in os.listdir(dir_path):
    if f.endswith(ext):
        images.append(f)
images.sort()
# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(50) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))