import cv2
from A1 import *
from skimage import measure,color
import matplotlib.patches as mpatches

#2-1
def Read(num): #讀取字母編號

    all_cell = np.zeros( shape = (64) ) #HOG特徵向量儲存

    for i in range(1,num+1): #2-1 範圍是[1 26]的整數。
        c='ABC'+str(i).rjust(2,'0')+'.jpg' #生成檔名字串
        img = cv2.imread(c, cv2.IMREAD_GRAYSCALE) #讀取樣本
        img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA) #縮小至 16×16 像素尺寸
        hog_vector=pret(img)#分別算出 HOG 特徵向量
        L2_flatten = np.array(hog_vector.flatten())#壓扁
        all_cell=np.vstack([all_cell, L2_flatten])

    return all_cell

All_letter=Read(26) 
All_letter=np.delete(All_letter,0,0)
#print("all",All_letter)

test_img = cv2.imread('p3.jpg', cv2.IMREAD_GRAYSCALE) #2-2 讀取樣本

ret, binary = cv2.threshold(test_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #2-3二值化
#binary = np.where(binary==0,1,0).astype(np.uint8)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4, ltype=None) #連通區域分離
stats = np.delete(stats,0,0) #移除外框
max_x_length = np.max(stats,axis=0)[2] #取x長度最大值
max_y_length = np.max(stats,axis=0)[3] #取y長度最大值

letter_class = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

for istat in stats:

        ##由左至右切多個字母區塊
        x=np.array(istat[0:1]+[-4])[0] #左上x
        x_L = np.array(istat[2:3])[0] #左上x的長
        x_offset = int((max_x_length-x_L)/2) #x位移長度
        x=x-x_offset
        x_length=np.array(x+max_x_length+[4])[0]
        y=np.array(istat[1:2]+[-4])[0]
        y_length=np.array(y+max_y_length+[4])[0]

        str_img=test_img[y:y_length,x:x_length]

        cv2.imshow("test_img", str_img)
        cv2.waitKey(0)

        str_img=cv2.resize(str_img, (16, 16), interpolation=cv2.INTER_AREA)
        hog_vector_str_img=pret(str_img)#算出 HOG 特徵向量
        L2_flatten_str_img = np.array(hog_vector_str_img.flatten())#壓扁

        L2_result = np.zeros(26) #儲存結果
        for j in range(26):
            #print('j=',j,All_letter[j])
            #print(L2_flatten_str_img)
            L2=np.sqrt(np.sum((All_letter[j]- L2_flatten_str_img)**2)) #計算字母樣本與ROI區域的HOG的L2距離
            L2_result[j]=L2 #儲存L2距離結果
        
        list_L2_result = L2_result.tolist()
        #print(list_L2_result)
        list_a_min_list = min(list_L2_result) 
        min_index = list_L2_result.index(min(list_L2_result)) # 返回最小值的索引
    
        print(letter_class[min_index])


# 

#plt.show()
##


