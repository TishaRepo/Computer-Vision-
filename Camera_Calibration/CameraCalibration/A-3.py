import cv2
import numpy as np
import os
import scipy.optimize
import glob
import matplotlib.pyplot as plt

CB_SIZE = (9,6)
square_size=3
three_d_square_size = max(min(CB_SIZE[0] - 1, square_size), 1)
output="output"

SIZE_SQ = 21.5
def main():

   
    image_path = "Calib_Imgs_Custom"
    corner_points = load_data(image_path)
    print("corners detected")
    
    # Initial estimate
    H_list = get_H_list(corner_points)
    b = estimate_B(corner_points, H_list)
    K = estimate_K(b)
    print("Initial Estimate of K:\n", K)
    
    extrinsics = estimate_RT(K,H_list)
    
    world_points = get_world_points()
    # Optimization Part
    alpha, beta, gamma,u0,v0 = K[0, 0], K[1, 1], K[0, 1] ,K[0, 2], K[1, 2]
    k1,k2=0,0
    initial_params = [alpha, beta, gamma,u0,v0,k1,k2]
    # using the initial estimates of K and dist. coefficients to generate image points from world points.
    projection_error = estimate_reprojection_error(initial_params,world_points,corner_points,extrinsics)
    print("projection error:\n",np.mean(projection_error))

    print("Performing non Linear Optimization")
    K_new, kc = optimize(initial_params,world_points,corner_points,extrinsics)
    
    print("The new intrinsic matrix K is:\n",K_new)
    print("kc is:", kc)

    extrinsics = estimate_RT(K_new,H_list)
    
    
    K = K_new
    alpha, beta, gamma,u0,v0 = K[0, 0], K[1, 1], K[0, 1] ,K[0, 2], K[1, 2]
    k1,k2=kc
    initial_params = [alpha, beta, gamma,u0,v0,k1,k2]

    projection_error = estimate_reprojection_error(initial_params,world_points,corner_points,extrinsics)
    print("projection error:\n",np.mean(projection_error))
    distortion = np.array([kc[0],kc[1],0,0,0],dtype=float)
    print("distortion is :" , distortion)

    x=0

    for image in sorted(glob.glob(f"{image_path}/*.jpg")):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # cv2.imshow("before", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        gray = cv2.undistort(gray,K_new,distortion)
        # cv2.imshow("after", gray2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ret, corners = cv2.findChessboardCorners(gray,(9,6),None)
        out= cv2.drawChessboardCorners(img,(9,6),corners,ret)
        resize=640,480
        out=cv2.resize(out,resize)
        cv2.imwrite("intrinsic"+str(x)+".jpg",out)
        # cv2.imshow("corners", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        img2=augment(gray,corners,K_new,distortion)
        resize=640,480
        img2=cv2.resize(img2,resize)
        cv2.imwrite("ar"+str(x)+".jpg",img2)
        x=x+1
        # cv2.imshow("ar",img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    

    
def augment(gray,corners,mtx,dist):

    objp2 = np.zeros((CB_SIZE[1] * CB_SIZE[0], 3), np.float32)
    objp2[:, :2] = np.mgrid[0:CB_SIZE[0], 0:CB_SIZE[1]].T.reshape(-1, 2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    spaces_cnt = three_d_square_size
    axis3 = np.float32([[0, 0, 0], [0, spaces_cnt, 0], [spaces_cnt, spaces_cnt, 0], [spaces_cnt, 0, 0],
                        [0, 0, -spaces_cnt], [0, spaces_cnt, -spaces_cnt], [spaces_cnt, spaces_cnt, -spaces_cnt],
                        [spaces_cnt, 0, -spaces_cnt]])
    ret2, rvecs2, tvecs2 = cv2.solvePnP(objp2, corners2, mtx, dist)
    imgpts3, jac3 = cv2.projectPoints(axis3, rvecs2, tvecs2, mtx, dist)
    
    img = draw3(gray, corners2, imgpts3)

    return img

def draw3(img, corners, imgpts):
        imgpts = np.int32(imgpts).reshape(-1, 2)
        # draw ground floor in green
        img= cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), thickness=-3)
        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, pt1=tuple(imgpts[i]), pt2=tuple(imgpts[j]), color=(255), thickness=3)
        # draw top layer in red color
        img = cv2.drawContours(img, contours=[imgpts[4:]], contourIdx=-1, color=(0, 0, 255), thickness=3)
        return img    

def loss(initial_params,world_points,img_points_set,RT):
    final_error = []
    error = []
    for i,RT3 in enumerate(RT):
        mi_hat = projection(initial_params,world_points,RT3)
        mi = img_points_set[i].reshape(54,2)

        for m, m_ in  zip(mi, mi_hat.squeeze()):
            e = np.linalg.norm(m - m_, ord=2) # compute L2 norm
            error.append(e)
        err = np.sum(error)
       
        final_error.append(err)
 
    return final_error

def optimize(initial_params,world_points_set,img_points_set,RT):
    opt = scipy.optimize.least_squares(fun = loss, x0 = initial_params, method="lm", args = [world_points_set, img_points_set, RT])
    params = opt.x

    alpha, beta, gamma, u0, v0, k1 ,k2 = params
    K_new= np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,      1]])
    kc = (k1,k2)

    return K_new,kc

def estimate_reprojection_error(initial_params,world_points,img_corners,RT):
    final_error = []
    error = []
    
    for i,RT3 in enumerate(RT):
        
        mi_hat = projection(initial_params,world_points,RT3)
        mi = img_corners[i]
        
        for m, m_ in  zip(mi, mi_hat.squeeze()):
            e = np.linalg.norm(m - m_, ord=2) # compute L2 norm
            error.append(e)

        err = np.mean(error)
       
        final_error.append(err)

    return final_error

def projection(initial_params,world_points,RT):
    alpha, beta, gamma,u0,v0,k1,k2=initial_params

    K = np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,      1]])
    m_i_ = []
    
    for M in world_points:
        M = np.float64(np.hstack((M,0,1)))
        projected_pt = np.dot(RT,M)
        projected_pt = projected_pt/projected_pt[-1]

        #compute radius of distortion
        x = projected_pt[0]
        y = projected_pt[1]
        r = x**2 + y**2
      
        #projected image coordinates
        uv = np.dot(K,projected_pt)
        u = uv[0]/uv[-1]
        v = uv[1]/uv[-1]
      
        #eq 11 and 12 from the report
        u_hat = u+ (u-u0)*(k1*r + k2*(r**2))
        v_hat = v + (v-v0)*(k1*r + k2*(r**2))
       
        m_ = np.hstack((u_hat,v_hat))
        
        m_i_.append(m_)
    return np.array(m_i_)


def estimate_RT(K, H_list):
    extrinsic = []
    for h in H_list:
        h1,h2,h3 = h.T # get the column vectors
        K_inv = np.linalg.inv(K)
        lamda = 1/np.linalg.norm(K_inv.dot(h1),ord =2 )
        r1 = lamda*K_inv.dot(h1)
        r2 = lamda*K_inv.dot(h2)
        r3 = np.cross(r1,r2)
        
        t = lamda*K_inv.dot(h3)
        RT = np.vstack((r1,r2,r3, t)).T
        extrinsic.append(RT)
    return extrinsic

def estimate_K(b):
    b11, b12, b22, b13, b23, b33 = b[0],b[1],b[2],b[3],b[4],b[5]

    
    v0 = (b12*b13 - b11*b23)/(b11*b22 - b12**2)
    lamda = b33 - (b13**2 + v0*(b12*b13 - b11*b23))/b11
    alpha = np.sqrt(lamda/b11)
    beta = np.sqrt(lamda*b11 /(b11*b22 - b12**2))
    gamma = -b12*(alpha**2)*beta/lamda
    u0 = gamma*v0/beta -b13*(alpha**2)/lamda
    K = np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,      1]])

    return K

def get_H_list(corner_points):
    H_list = list()
    M = get_world_points()
    for img in corner_points:
        m = corner_points[img]
        H_list.append(cv2.findHomography(M,m)[0])
    H_list = np.array(H_list)    
    return H_list

def estimate_B(corner_points,H_list):
    
    # From eq 8 of the report, we only need v11, v12 and v22 to estimate V.
    V = list()
    for h in H_list:
        v11 = get_vij(h,1,1)
        v22 = get_vij(h,2,2)
        v12 = get_vij(h,1,2)
        V.append(v12.T)
        V.append((v11-v22).T)
    V = np.array(V)
    u,s,v = np.linalg.svd(V)
    b = v[-1,:]
    return b

def get_vij(H,i,j):
    i,j = i-1,j-1
    v_ij = np.array([H[0, i]*H[0, j],
                    H[0, i]*H[1, j] + H[1, i]*H[0, j],
                    H[1, i]*H[1, j],
                    H[2, i]*H[0, j] + H[0, i]*H[2, j],
                    H[2, i]*H[1, j] + H[1, i]*H[2, j],
                    H[2, i]*H[2, j] 
                    ])
    return v_ij

def get_world_points():
    x_val = np.arange(0,CB_SIZE[0]*SIZE_SQ,SIZE_SQ)
    y_val = np.arange(0,CB_SIZE[1]*SIZE_SQ,SIZE_SQ)
    xx,yy = np.meshgrid(x_val,y_val)
    y = yy.reshape((CB_SIZE[0]*CB_SIZE[1],1))
    
    # FLIP because the checker board corners are returned row wise left to right from the top, not bottom.
    x = np.flip(xx.reshape((CB_SIZE[0]*CB_SIZE[1],1)))
    world_points = np.hstack((y,x))
    
    return world_points

def load_data(image_path):
    img_files = os.listdir(image_path)
    for i in range(len(img_files)):
        img_files[i] = os.path.join(image_path, img_files[i])
    img_checkerBoardCorners = dict()
    for i in range(len(img_files)):
        gray_image = cv2.imread(img_files[i])
        gray_image = cv2.cvtColor(gray_image,cv2.COLOR_BGR2GRAY)

        _,cb_corners = cv2.findChessboardCorners(image=gray_image,patternSize=CB_SIZE)
        cb_corners = cb_corners.reshape(-1,2)
        # print(cb_corners)
        img_checkerBoardCorners[i] = cb_corners
    return img_checkerBoardCorners




if __name__=='__main__':
    main()