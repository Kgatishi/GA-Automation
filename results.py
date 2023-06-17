import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, io
from PIL import Image
import matrics

# Convert an image based on the threshold values provided
def threshold_converter(im, th = [100]):
    #print("--------")
    #print(th)
    th = [ int(v) for v in th]
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im <= th[0]] = np.mean(im[ im <= th[0]] )

    for i in range(len(th)-1):
        thresholded_im[(im > th[i]) & (im <= th[i+1]) ] = np.mean(im[ (im > th[i]) & (im <= th[i+1])] )

    thresholded_im[im > th[-1]] = np.mean(im[ im > th[-1]] )

    return thresholded_im

# Plot images
# Test and compare threshold image by visualizing the images
def image_plot(img,img_updated):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax = axes.ravel()

    #print("---------------------------------")
    ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax[0].set_title('Gray Image')
    #print("-----------")
    ax[1].imshow(img_updated, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax[1].set_title('Segmented')
    
    #print("-----------")
    plt.tight_layout()
    plt.show()

# Read results from external file to convert threshold and test matrices
def measure_accuracy(df,img_name,img_hist):

    # create matrics columns
    def psnr(arr):
        img_updated = threshold_converter(img_hist.copy(),arr)
        results = matrics.psnr(img_hist,img_updated)
        return results

    
    def rmse(arr):
        img_updated = threshold_converter(img_hist.copy(),arr)
        results = matrics.rmse(img_hist,img_updated)
        return results

    def ssim(arr):
        img_updated = threshold_converter(img_hist.copy(),arr)
        results = matrics.ssim(img_hist,img_updated)
        return results

    def fsim(arr):
        img_updated = threshold_converter(img_hist.copy(),arr)
        results = matrics.fsim(img_hist,img_updated)
        return results
    
    df['psnr'] = df['thresholds'].apply(psnr)
    df['rmse'] = df['thresholds'].apply(rmse)
    df['ssim'] = df['thresholds'].apply(ssim)
    df['fsim'] = df['thresholds'].apply(fsim)

    #df = df.drop(['num_threshold', 'config'], axis=1)
    #df = df[["fitness","thresholds","psnr","rmse","ssim","fsim","time"]]
    df = df[["fitness","thresholds","psnr","rmse","ssim","fsim"]]
    return df



if __name__ == "__main__":
    
    test_images = [
        './images/lenna.png',
        './images/pepper.tiff',
        './images/house.tiff',
        './images/boats.bmp',
        './images/lake.bmp',
        './images/airplane.png'
    ]
    
    list_gray_img = []
    for im in test_images:
        print("------------------------")
        img = io.imread(im)
        img2 = np.array(Image.fromarray(img).convert('L'))

        list_gray_img.append(img2)
        list_gray_img.append(img2)
        list_gray_img.append(img2)
        list_gray_img.append(img2)
        ''''
        img_updated = threshold_converter(img2.copy(),[128])
        image_plot(img2,img_updated)
        img_updated = threshold_converter(img2.copy(),[85, 124, 160])
        image_plot(img2,img_updated)
        img_updated = threshold_converter(img2.copy(),[67, 98, 123, 147, 172])
        image_plot(img2,img_updated)
        break
        '''
    #df = measure_accuracy("./base_Results.csv")
    df = measure_accuracy("./GA_results.csv")
    print(df)
    output = df.values.tolist()
    count = 1
    for row in output:
       #fitness , thresholds , psnr , rmse , ssim , fsim , time = row
       fitness , thresholds , psnr , rmse , ssim , fsim  = row
       print("%.6e & %s & %.6f & %.6f & %.6f & %.6f " % (fitness , thresholds , psnr , rmse , ssim , fsim  ))
       if count%4 == 0 : print("-----------------------------------------------------------------------")
       count += 1
