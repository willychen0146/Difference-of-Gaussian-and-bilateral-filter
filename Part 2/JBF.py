import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        # 以上作padded，使用BORDER_REFLECT做

        ### TODO ###
        # 通過 np.mgrid 函數生成一個二維坐標矩陣，其中矩陣的每一個元素都是對應的 x 和 y 坐標值，網格範圍為 [-pad_w, pad_w+1]
        x, y = np.mgrid[-self.pad_w:self.pad_w + 1, -self.pad_w:self.pad_w + 1] 
        
        # 生成標準差為 sigma_s 的 Gaussian kernel，將結果存儲在變數 s_kernel 中 (空間kernel) - 與位置相關 (愈近影響愈大)
        # x 和 y 的範圍都是 [-self.pad_w, self.pad_w + 1]，這是因為高斯濾波器的作用範圍是以中心為原點的正方形區域
        # 而中心點到四邊的距離是 self.pad_w，因此需要在 x 和 y 軸上各取 self.pad_w 個點，並且再加上中心點，這樣才能確保生成的坐標矩陣覆蓋了整個濾波器的作用範圍
        # 因此，生成的坐標矩陣的大小是 (2 * self.pad_w + 1) x (2 * self.pad_w + 1)
        s_kernel = np.exp((x ** 2 + y ** 2) / (-2 * (self.sigma_s ** 2))) 

        # 這個函數中，輸入的 guidance 圖像是一個灰度圖像，像素值表示每個像素的亮度強度。
        # 在之後的計算中，會將 guidance 圖像與 x，y 坐標矩陣（生成的 s_kernel）一起使用，以便在空間域和灰度域同時執行濾波操作
        # 為了確保 guidance 和 s_kernel 具有相同的數值範圍，需要將 guidance 數組的值除以 255，將其歸一化到 [0,1] 範圍內
        # 當在濾波器中使用這兩個數組時，它們會有相同的權重，從而避免了因權重不均勻而導致的不良結果
        padded_guidance = padded_guidance / 255 # 將 guidance圖 的值從 [0, 255] 範圍norm到 [0, 1] 範圍，方便後面計算權重

        
        # 在雙邊濾波器中，對於輸入圖像的每一個像素，都需要對其進行一系列計算並得到一個新的像素值，因此需要一個與輸入圖像相同大小的數組來存儲這些計算結果。    
        # output 數組被初始化為與輸入圖像 img 相同大小的零矩陣
        # 隨著算法的執行，output 數組中的每個像素將被逐漸計算和更新，最終形成輸出圖像
        output = np.zeros(img.shape) # 創建一個與輸入圖像 img 大小相同的零矩陣，用於存儲濾波後的結果

        # 對每個通道進行運算
        for c in range(3): # 迴圈每個chanel（RGB 3個），將chanel固定為RGB某一個channel，將它變為一個二微陣列進行運算
            padded_img_c = padded_img[..., c] # 取出當前通道的值，用 padded_img[..., c] 取出
            output_c = np.zeros(img.shape[:2]) # 創建一個與輸入圖像 img 大小相同的零矩陣 output_c (假設img的形狀是(512, 512, 3)，返回x及y的方向上的大小)，用於存儲當前通道 c 的濾波結果
            # 循環處理輸入圖像，對每個像素進行以下運算
            for i in range(self.pad_w, padded_guidance.shape[0] - self.pad_w): # 迴圈 padded_guidance圖的每個像素位置 (i,j)，取出以 (i,j) 為中心的 wndw_size x wndw_size 的矩形區域 w，用來計算與當前像素的相似性權重
                for j in range(self.pad_w, padded_guidance.shape[1] - self.pad_w):
                    # 取出當前像素周圍的引導圖像區域 'w'
                    w = padded_guidance[i - self.pad_w:i + self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1]
                    # 計算卷積核 r_kernel，其計算方式是基於引導圖像的相似度，根據高斯函數計算出每個像素和當前像素的距離，越相似的像素權重越高
                    r_kernel = ((w - padded_guidance[i, j]) ** 2) / (-2 * (self.sigma_r ** 2))
                    if len(padded_guidance.shape) == 3:
                        r_kernel = np.sum(r_kernel, axis=2)
                    r_kernel = np.exp(r_kernel)

                    # 計算 g 卷積核，其計算方式是基於空間相似性和引導圖像相似性，用一個高斯函數將二者合併起來
                    g = s_kernel * r_kernel
                    # 將 g 卷積核除以總和，將其normalize
                    g /= np.sum(g)
                    # 計算當前像素的輸出，用 np.sum(g * padded_img_c[i - self.pad_w:i + self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1]) 計算
                    output_c[i - self.pad_w, j - self.pad_w] = np.sum(g * padded_img_c[i - self.pad_w:i + self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1])
                    
            # 將每個通道的 output_c 設定給 output 的相應通道，以得到最終的輸出結果
            output[..., c] = output_c

            # 計算與當前像素在引導圖上距離不超過 pad_w 的像素點在高斯核下的權重，這裡通過歐幾里得距離計算 w 與當前像素的距離，然後通過標準差為 sigma_r 的高斯核計算權重 r_kernel。
            # 如果引導圖是三通道的圖像，則需要對 r_kernel 在第三維度上求和，將其轉為二維矩陣。然後將 s_kernel 與 r_kernel 相乘，得到最終的權重 g，除以總和使其歸一化
        
        return np.clip(output, 0, 255).astype(np.uint8)