import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___) 
        gaussian_images = []
        
        octave_images_1 = []
        octave_images_1.append(image)
        for level in range(1, self.num_guassian_images_per_octave): #每一組octave包含輸入4個模糊化結果 (**1-4)
            sigma = self.sigma**(level)
            filtered = cv2.GaussianBlur(image, (0, 0), sigma)
            octave_images_1.append(filtered)

        #image_2 = cv2.resize(first_ocatane[-1][-1], (0, 0), fx=0.5, fy=0.5) #將圖變成1/2
        image_2 = cv2.resize(octave_images_1[-1], (0, 0), fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST) #將圖變成1/2  
        
        octave_images_2 = []
        octave_images_2.append(image_2)
        for level in range(1, self.num_guassian_images_per_octave): #每一組octave包含輸入4個模糊化結果 (**1-4)
            sigma = self.sigma**(level)
            filtered = cv2.GaussianBlur(image_2, (0, 0), sigma)
            octave_images_2.append(filtered)

        gaussian_images = [octave_images_1, octave_images_2]     
        #print(gaussian_images)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = [] #2維列表，第一维表示不同的 octave，第二维表示不同的高斯模糊圖相減之結果
        for i in range(self.num_octaves):
            GI = gaussian_images[i]
            octave_dogs = []
          # for octave_images in gaussian_images: #將Step 1圖一個一個丟進迴圈中
            for j in range(self.num_DoG_images_per_octave): #表示每組octave要生成的DoG數量
                # print(octave_images[i][1].shape)
                # print(octave_images[i][0].shape)
                # print(type(octave_images[i][i+1]))
                # dog = cv2.subtract(octave_images[0][i+1], octave_images[0][i]) #將圖相減
                # dog = cv2.subtract(octave_images[j+1], octave_images[j]) #將圖相減
                dog = cv2.subtract(GI[j], GI[j+1])
                octave_dogs.append(dog)
                # Normalize DoG images to [0,225] ?
                M, m = max(dog.flatten()), min(dog.flatten())
                norm = (dog - m) * 255 / (M - m)
                # cv2.imwrite(f'testdata/DoG{i+1}-{j+1}.png', norm)
            dog_images.append(octave_dogs) #將迴圈中的結果存回dog_image中
        
        # print(type(dog_images))
        # print(type(dog_images[-1]))

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        '''
        method_1(not working)
        keypoints = [] #將從Step 2得到之圖算出local maxima後儲存
        for octave_dogs in dog_images:
            for i in range(1, self.num_DoG_images_per_octave-1): #1到3範圍
                prev_dog = octave_dogs[i-1]
                curr_dog = octave_dogs[i]
                next_dog = octave_dogs[i+1]

                thres_curr_dog = curr_dog > self.threshold
                thres_prev_dog = prev_dog > self.threshold
                thres_next_dog = next_dog > self.threshold

                extremum = np.logical_and(np.logical_and(thres_curr_dog, thres_prev_dog), thres_next_dog)
                keypoints_octave = np.transpose(np.nonzero(extremum))

                # convert keypoint index to image coordinates
                octave_scale = 2**(octave_dogs[0].shape[0] - i)
                for kp in keypoints_octave:
                    scale = self.sigma**(i+1)
                    x, y = tuple(scale*octave_scale*kp[::-1])
                    keypoints.append([x, y])
        '''

        # method_2(參考方法)
        keypoints = []
        for octave in range(self.num_octaves): # 跑2次大迴圈
            # 把 octave 轉換成一個 3 維陣列
            dogs = np.array(dog_images[octave])
            height, width = dogs[octave].shape

            # 檢查每個 3x3 的立方體是否為local extremum(大或小)
            # 跑過每個 DoG 圖像
            for dog in range(1, self.num_DoG_images_per_octave-1): # 不要跑0和3
                # 跑果每一個 pixel
                for x in range(1, width-2):
                    for y in range(1, height-2):
                        pixel = dogs[dog,y,x]
                        # 取得 3x3 的立方體
                        cube = dogs[dog-1:dog+2, y-1:y+2, x-1:x+2]

                        # 檢查是否為local extremum(大或小)
                        if (np.absolute(pixel) > self.threshold) and ((pixel >= cube).all() or (pixel <= cube).all()):
                            # 把關鍵點的座標加入列表中
                            # 如果 i 為 0，則乘以 2
                            if octave == 0:
                                keypoints.append([y, x])
                            else:
                                keypoints.append([y*2, x*2])

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        # keypoints = np.array(keypoints) # 將 keypoints 轉換為 numpy 陣列
        keypoints = np.unique(keypoints, axis=0)
        # print(keypoints)
        
        # sort 2d-point by y, then by x
        # if keypoints.ndim == 2:
        #     keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        # return keypoints
        
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints