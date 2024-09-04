import os
import cv2
import numpy as np
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, img_path, save_base_path):
        self.img_path = img_path
        self.save_base_path = save_base_path
        self.img = cv2.imread(img_path)

    def save_image(self, folder, suffix, img):
        save_path = os.path.join(self.save_base_path, folder)
        os.makedirs(save_path, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(self.img_path))[0] + suffix + ".jpg"
        cv2.imwrite(os.path.join(save_path, file_name), img)

class MorphologicalOperations(ImageProcessor):
    def __init__(self, img_path, save_base_path):
        super().__init__(img_path, save_base_path)

    def morph_erode(self):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        erosion = cv2.erode(self.img, k)
        self.save_image("morph_erode", "_erode", erosion)

    def morph_dilate(self):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilation = cv2.dilate(self.img, k)
        self.save_image("morph_dilate", "_dilate", dilation)

    def morph_open_close(self):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, k)
        closing = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, k)
        self.save_image("morph_open_close", "_open", opening)
        self.save_image("morph_open_close", "_close", closing)

    def morph_gradient(self):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(self.img, cv2.MORPH_GRADIENT, k)
        self.save_image("morph_gradient", "_gradient", gradient)

    def morph_hat(self):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tophat = cv2.morphologyEx(self.img, cv2.MORPH_TOPHAT, k)
        blackhat = cv2.morphologyEx(self.img, cv2.MORPH_BLACKHAT, k)
        self.save_image("morph_hat", "_tophat", tophat)
        self.save_image("morph_hat", "_blackhat", blackhat)

class EdgeDetection(ImageProcessor):
    def __init__(self, img_path, save_base_path):
        super().__init__(img_path, save_base_path)

    def canny_edge(self):
        canny = cv2.Canny(self.img, 100, 200)
        self.save_image("edge_canny", "_canny", canny)

    def edge_laplacian(self):
        laplacian_1 = cv2.Laplacian(self.img, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian_1)
        self.save_image("edge_laplacian", "_laplacian_1", laplacian_1)
        self.save_image("edge_laplacian", "_laplacian", laplacian)

    def edge_sobel(self):
        sobelx = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        sobel = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)
        self.save_image("edge_sobel", "_sobelx", cv2.convertScaleAbs(sobelx))
        self.save_image("edge_sobel", "_sobely", cv2.convertScaleAbs(sobely))
        self.save_image("edge_sobel", "_sobel", sobel)

    def edge_scharr(self):
        scharrx = cv2.Scharr(self.img, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(self.img, cv2.CV_64F, 0, 1)
        scharr = cv2.addWeighted(cv2.convertScaleAbs(scharrx), 0.5, cv2.convertScaleAbs(scharry), 0.5, 0)
        self.save_image("edge_scharr", "_scharrx", cv2.convertScaleAbs(scharrx))
        self.save_image("edge_scharr", "_scharry", cv2.convertScaleAbs(scharry))
        self.save_image("edge_scharr", "_scharr", scharr)

    def edge_prewitt(self):
        kernelx = cv2.getDerivKernels(1, 0, 3, normalize=True)
        kernely = cv2.getDerivKernels(0, 1, 3, normalize=True)
        prewittx = cv2.filter2D(self.img, cv2.CV_64F, kernelx[0] * kernelx[1].T)
        prewitty = cv2.filter2D(self.img, cv2.CV_64F, kernely[0] * kernely[1].T)
        prewitt = cv2.addWeighted(cv2.convertScaleAbs(prewittx), 0.5, cv2.convertScaleAbs(prewitty), 0.5, 0)
        self.save_image("edge_prewitt", "_prewittx", cv2.convertScaleAbs(prewittx))
        self.save_image("edge_prewitt", "_prewitty", cv2.convertScaleAbs(prewitty))
        self.save_image("edge_prewitt", "_prewitt", prewitt)

    def edge_roberts(self):
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        robertsx = cv2.filter2D(self.img, cv2.CV_64F, kernelx)
        robertsy = cv2.filter2D(self.img, cv2.CV_64F, kernely)
        roberts = cv2.addWeighted(cv2.convertScaleAbs(robertsx), 0.5, cv2.convertScaleAbs(robertsy), 0.5, 0)
        self.save_image("edge_roberts", "_robertsx", cv2.convertScaleAbs(robertsx))
        self.save_image("edge_roberts", "_robertsy", cv2.convertScaleAbs(robertsy))
        self.save_image("edge_roberts", "_roberts", roberts)

    def edge_kirsch(self):
        kernelx = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
        kernely = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
        kirschk = cv2.filter2D(self.img, cv2.CV_64F, kernelx)
        kirschy = cv2.filter2D(self.img, cv2.CV_64F, kernely)
        kirsch = cv2.addWeighted(cv2.convertScaleAbs(kirschk), 0.5, cv2.convertScaleAbs(kirschy), 0.5, 0)
        self.save_image("edge_kirsch", "_kirschk", cv2.convertScaleAbs(kirschk))
        self.save_image("edge_kirsch", "_kirschy", cv2.convertScaleAbs(kirschy))
        self.save_image("edge_kirsch", "_kirsch", kirsch)

    def edge_freichen(self):
        kernelx = np.array([[1, np.sqrt(2), 1], [0, 0, 0], [-1, -np.sqrt(2), -1]])
        kernely = np.array([[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]])
        freichenx = cv2.filter2D(self.img, cv2.CV_64F, kernelx)
        freicheny = cv2.filter2D(self.img, cv2.CV_64F, kernely)
        freichen = cv2.addWeighted(cv2.convertScaleAbs(freichenx), 0.5, cv2.convertScaleAbs(freicheny), 0.5, 0)
        self.save_image("edge_freichen", "_freichenx", cv2.convertScaleAbs(freichenx))
        self.save_image("edge_freichen", "_freicheny", cv2.convertScaleAbs(freicheny))
        self.save_image("edge_freichen", "_freichen", freichen)

    def edge_diff(self):
        kernelx = np.array([[1, 1, 1], [-1, -1, -1], [0, 0, 0]])
        kernely = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        diffx = cv2.filter2D(self.img, cv2.CV_64F, kernelx)
        diffy = cv2.filter2D(self.img, cv2.CV_64F, kernely)
        diff = cv2.addWeighted(cv2.convertScaleAbs(diffx), 0.5, cv2.convertScaleAbs(diffy), 0.5, 0)
        self.save_image("edge_diff", "_diffx", cv2.convertScaleAbs(diffx))
        self.save_image("edge_diff", "_diffy", cv2.convertScaleAbs(diffy))
        self.save_image("edge_diff", "_diff", diff)

    def edge_nevatia_babu(self):
        kernels = [np.array([[100, 100, 100], [0, 0, 0], [-100, -100, -100]]),
                   np.array([[100, 100, 0], [100, 0, -100], [0, -100, -100]]),
                   np.array([[0, 100, 100], [-100, 0, 100], [-100, -100, 0]])]
        edges = [cv2.filter2D(self.img, cv2.CV_64F, kernel) for kernel in kernels]
        nevatia_babu = cv2.addWeighted(cv2.convertScaleAbs(edges[0]), 0.33,
                                       cv2.addWeighted(cv2.convertScaleAbs(edges[1]), 0.33,
                                                       cv2.convertScaleAbs(edges[2]), 0.34, 0), 1, 0)
        self.save_image("edge_nevatia_babu", "_nevatia_babu", nevatia_babu)

    def edge_marr_hildreth(self):
        blurred_img = cv2.GaussianBlur(self.img, (5, 5), 0)
        laplacian_2 = cv2.Laplacian(blurred_img, cv2.CV_64F)
        marr_hildreth = cv2.convertScaleAbs(laplacian_2)
        self.save_image("edge_marr_hildreth", "_marr_hildreth", marr_hildreth)

class ToBinaryImage(ImageProcessor):
    def __init__(self, img_path, save_base_path):
        super().__init__(img_path, save_base_path)
    
    def threshold_otsu(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, t_130 = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)        
        t, t_otsu = cv2.threshold(gray, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        self.save_image("threshold_otsu", "_otsu_binary", binary)
        self.save_image("threshold_otsu", "_otsu_130", t_130)
        self.save_image("threshold_otsu", "_otsu", t_otsu)
        
    
    def adaptive_threshold(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        blk_size = 9 
        C = 5
        
        ret, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                            cv2.THRESH_BINARY, blk_size, C)
        th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                            cv2.THRESH_BINARY, blk_size, C)
        
        self.save_image("adaptive_threshold", "_global_otsu", th1)
        self.save_image("adaptive_threshold", "_adaptive_mean", th2)
        self.save_image("adaptive_threshold", "_adaptive_gaussian", th3)

class HistoProcessing(ImageProcessor):
    def __init__(self, img_path, save_base_path):
        super().__init__(img_path, save_base_path)
    
    def histo_normalize(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        img_f = gray.astype(np.float32)
        img_norm = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min()))
        img_norm = img_norm.astype(np.uint8)

        img_norm2 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        self.save_image("histogram_normalize", "_normalize", img_norm)
        self.save_image("histogram_normalize", "_normalize2", img_norm2)
        
    def histo_equalization(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) / (gray.shape[0] * gray.shape[1]) * 255
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        gray2 = cdf[gray]

        equalized = cv2.equalizeHist(gray)
        
        self.save_image("histogram_equalization", "_equalized", equalized)
        self.save_image("histogram_equalization", "_equalized2", gray2)
    
    def histo_equalize_color(self):
        img = cv2.imread(self.img_path)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        img2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        self.save_image("histogram_equalize_color", "_equalized", img2)

    def histo_clahe(self):
        ing_yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        
        img_eq = ing_yuv.copy()
        img_eq[:,:,0] = cv2.equalizeHist(img_eq[:,:,0])
        img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)
        
        img_clahe = ing_yuv.copy()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])
        img_clahe = cv2.cvtColor(ing_yuv, cv2.COLOR_YUV2BGR)
        
        self.save_image("clahe", "_clahe", img_clahe)
        self.save_image("clahe", "_equalized", img_eq)

class BrightnessContrast(ImageProcessor):
    def __init__(self, img_path, save_base_path):
        super().__init__(img_path, save_base_path)

    def adjust_brightness(self, brightness=0):
        img_bright = cv2.convertScaleAbs(self.img, beta=brightness)
        self.save_image("brightness", f"_brightness_{brightness}", img_bright)

    def adjust_contrast(self, contrast=1.0):
        img_contrast = cv2.convertScaleAbs(self.img, alpha=contrast)
        self.save_image("contrast", f"_contrast_{contrast}", img_contrast)

    def adjust_brightness_contrast(self, brightness=0, contrast=1.0):
        img_bright_contrast = cv2.convertScaleAbs(self.img, alpha=contrast, beta=brightness)
        self.save_image("brightness_contrast", f"_brightness_{brightness}_contrast_{contrast}", img_bright_contrast)

        
if __name__ == "__main__":
    base_path_list = os.listdir("dataset/original")
    path_list = [f"dataset/original/{file}" for file in base_path_list]
    
    # alpha=contrast, beta=brightness
    contrast=6.5 # alpha
    brightness = 70 # beta

    for path in tqdm(path_list):
        img_list = os.listdir(path)
        # print(img_list)
        # Full Path List (Path + img_list)
        Full_path_list = [f"{path}/{img}" for img in img_list]
        
        save_path = os.path.join("data", os.path.basename(path))
        
        for img_path in tqdm(Full_path_list):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            # print(f"Processing {img_name}")

            morph = MorphologicalOperations(img_path, save_path)
            morph.morph_erode()
            morph.morph_dilate()
            morph.morph_open_close()
            morph.morph_gradient()
            morph.morph_hat()

            edge = EdgeDetection(img_path, save_path)
            edge.canny_edge()
            edge.edge_laplacian()
            edge.edge_sobel()
            edge.edge_scharr()
            edge.edge_prewitt()
            edge.edge_roberts()
            edge.edge_kirsch()
            edge.edge_freichen()
            edge.edge_diff()
            edge.edge_nevatia_babu()
            edge.edge_marr_hildreth()
            
            binary = ToBinaryImage(img_path, save_path)
            binary.threshold_otsu()
            binary.adaptive_threshold()
            
            histo = HistoProcessing(img_path, save_path)
            histo.histo_normalize()
            histo.histo_equalization()
            histo.histo_equalize_color()
            histo.histo_clahe()
            
            BrightnessandContrast = BrightnessContrast(img_path, save_path)
            BrightnessandContrast.adjust_brightness(brightness=brightness)
            BrightnessandContrast.adjust_contrast(contrast=contrast)
            BrightnessandContrast.adjust_brightness_contrast(brightness=brightness, contrast=contrast)