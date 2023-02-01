import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

img = cv2.imread('ABC01.jpg', cv2.IMREAD_GRAYSCALE) #讀入灰階字母樣本

def pret(img):#撰寫產生4X4方向直方圖描述子的程式
    #img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA) #縮小至 16×16 像素尺寸
    k=1 #畫圖初始值
    cell_size = 4 #將影像分成 4×4 區域
    all_cell = np.zeros( shape = ( 4, 4, 4) ) #特徵向量儲存
    for i in range(cell_size):
        for j in range(cell_size):
            cell=np.zeros(((cell_size),(cell_size)),dtype=np.uint16)
            cell=img[4*i:4+4*i,4*j:4+4*j]
            grad,phase =sobel(cell) #這 16 個像素的梯度絕對值 M 與方向角
            bin = hog(grad,phase)#2.5 繪製直方圖
            plt.subplot(4,4,k)#2.5 繪製直方圖
            draw(bin)
            k+=1
            all_cell[i][j] = bin
    #plt.show()
    return all_cell

def sobel(img): #Sobel 垂直與水平邊緣檢測的結果(不要做二值化)
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude=abs(gradient_values_x)+abs(gradient_values_y)
    #gradient_magnitude = (gradient_magnitude * 255 / np.max(gradient_magnitude)).astype(np.uint8)

    sobel_phase=np.zeros(((img.shape[0]),(img.shape[1])),dtype=np.uint16)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            sobel_phase[x][y] = (math.atan2(gradient_values_y[x][y],gradient_values_x[x][y])/math.pi*180) % 180 #弧度轉角度

    return gradient_magnitude,sobel_phase

def hog(grad,phase):
    angle_unit = 180 / 4 #以 45 度為間隔
    binn = np.zeros(4)
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            if grad[i,j]>0:
                binn[np.int16( phase[i,j] / angle_unit)] += 1 #int(grad[i,j])
    return binn

def draw(bin):#2.4 繪製直方圖
    x = np.arange(4)
    str1 = ( '1', '2', '3', '4')
    plt.bar( x, height = bin, width = 0.8, label = 'cell histogram', tick_label = str1)
    plt.ylim(0,20)

def input(num): #2.1撰寫主程式 讀取字母編號

    c='ABC'+str(num).rjust(2,'0')+'.jpg' #2.2 生成檔名字串
    img = cv2.imread(c, cv2.IMREAD_GRAYSCALE) #讀取樣本
    img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA) #2.3 讀取灰階字母樣本，縮小至 16×16 像素尺寸
    hog_vector=pret(img) #2.4&2.5 套用前述的HOG函式，獲得字母樣本HOG向量 繪製直方圖
    
    test_img = cv2.imread('t2.jpg', cv2.IMREAD_GRAYSCALE) #2.6 讀取樣本
    test_img = cv2.resize(test_img, (128, 128), interpolation=cv2.INTER_AREA) #2.6 縮小至 128×128 像素尺寸
    L2_result = np.zeros( shape = ( 8, 8) ) #儲存結果


    for i in range(8): #2.7 用迴圈讀取測試影像不同位置的ROI,ROI的大小為16X16像素
        for j in range(8):
            roiImg=test_img[16*i:16+16*i,16*j:16+16*j]
            roiImg_vector=pret(roiImg)# 2.8 套用前述的HOG函式，獲得ROI區域的向量
            L2=np.sqrt(np.sum((hog_vector.flatten()- roiImg_vector.flatten())**2)) #2.9 計算字母樣本與ROI區域的HOG的L2距離
            L2_result[i][j]=L2 #儲存L2距離結果

    # #2.10
    L2_flatten = np.array(L2_result.flatten())
    arr_p = findThreeSmallest(L2_flatten)


    #製作紅色遮罩
    mask_img = np.zeros((16, 16, 3), np.uint8)
    mask_img[0:16, 0:16, :] = [0, 0, 255] 

    test2_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)  
    for i in range(3):

        row = np.int(arr_p[i]/8)
        col = np.int(arr_p[i] % 8)
        print('row',row,'col',col)

        roiImg_change=test_img[16*row:16+16*row,16*col:16+16*col]
        image = cv2.cvtColor(roiImg_change, cv2.COLOR_GRAY2RGB)
        mask_img_result = cv2.add(image, mask_img)
        test2_img[16*row:16+16*row,16*col:16+16*col]=mask_img_result

        cv2.imshow("roi", mask_img_result)
        cv2.waitKey(0)

    cv2.imshow('Image', test2_img)
    cv2.waitKey(0)




def findSmallest(arr):
    smallest = arr[0] #儲存最小的直
    smallest_index = 0 #儲存位置
    for i in range(1,len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index


def findThreeSmallest(arr):
    arr_p = np.zeros(3)
    for i in range(len(arr_p)):
        idx = findSmallest(arr)
        arr_p[i]=idx+i
        arr = np.delete(arr, idx)
    return arr_p



input(15)

#cv2.imshow('Image', test_img)
# cv2.imwrite("Image-test.jpg", img)
#cv2.waitKey(0)

