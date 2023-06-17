from glob import glob
import pygad
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import rand
from skimage import data, io
from PIL import Image


class Segmentation:
    def __init__(self, _image, _otsu= False, _kapur=False):
        self.image_histograms  = _image
        self.otsu = _otsu
        self.kapur = _kapur

    def evaluation_function(self, solution, solution_idx):
        if(len(solution)>len(np.unique(solution))):
            return -1

        hist = self.image_histograms 
        thresholds = np.sort(solution) 

        # Cumulative histogram
        c_hist = np.cumsum(hist)
        cdf = np.cumsum(np.arange(len(hist)) * hist)

        # Extending histograms for convenience
        c_hist = np.append(c_hist, [0])
        cdf = np.append(cdf, [0])

        # Extending thresholds for convenience
        e_thresholds = [-1]
        e_thresholds.extend(thresholds)
        e_thresholds.extend([len(hist) - 1])

        return hist, c_hist, cdf, e_thresholds

    
    def otsu_eval(self, ga_instanse, solution, solution_idx):
        hist, c_hist, cdf, thresholds = self.evaluation_function(solution, solution_idx)
        variance = 0
        for i in range(len(thresholds) - 1):
            # Thresholds
            t1 = thresholds[i] + 1
            t2 = thresholds[i + 1]

            weight = c_hist[t2] - c_hist[t1 - 1]                # Cumulative histogram
            r_cdf = cdf[t2] - cdf[t1 - 1]                       # Region CDF
            r_mean = r_cdf / weight if weight != 0 else 0       # Region mean

            variance += weight * r_mean ** 2

        return variance

    def kapur_eval(self, ga_instanse, solution, solution_idx):
        hist, c_hist, cdf, thresholds = self.evaluation_function(solution, solution_idx)
        total_entropy = 0
        for i in range(len(thresholds) - 1):
            # Thresholds
            t1 = thresholds[i] + 1
            t2 = thresholds[i + 1]

            hc_val = c_hist[t2] - c_hist[t1 - 1]                        # Cumulative histogram
            h_val = hist[t1:t2 + 1] / hc_val if hc_val > 0 else 1       # Normalized histogram
            entropy = -(h_val * np.log(h_val + (h_val <= 0))).sum()     # entropy

            total_entropy += entropy
        return total_entropy

    def threshold_GA(self, config, num_thresholds ):
        
        # ----------------------------------------------------
        #gene_space = [ [ v for v in range(255)] for i in range(num_genes)]
        # ----------------------------------------------------
        fitness_func = 0
        if self.otsu: fitness_func = self.otsu_eval 
        else: fitness_func = self.kapur_eval

        ga_instance = pygad.GA(num_generations = config["num_generations"],
                        #save_solutions=True,
                        num_parents_mating=2,
                        fitness_func = fitness_func ,
                        sol_per_pop = config["sol_per_pop"],
                        num_genes=num_thresholds,
                        gene_space=np.arange(255),
                        gene_type=int,
                        allow_duplicate_genes = False,
                        parent_selection_type = config["parent_selection_type"],
                        K_tournament = config["K_tournament"],
                        crossover_type = config["crossover_type"],
                        crossover_probability = config["crossover_probability"],
                        mutation_type = config["mutation_type"],
                        mutation_probability = config["mutation_probability"],
                        random_seed=2
                    )
        
        ga_instance.run()
        '''
        p1 = ga_instance.plot_fitness()
        p2 = ga_instance.plot_new_solution_rate()
        ga_instance.plot_genes()
        '''
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        return np.sort(solution) , solution_fitness


def main(im):
    img = io.imread(im)
    img2 = np.array(Image.fromarray(img).convert('L'))
    hist, bins = np.histogram(img2, bins=range(256), density=False)

    #print("Hists", hist.shape, hist.max(), hist.min())
    #print("Bins", bins.shape, bins.max(), bins.min())
    
    gene_space = {
                    "fitness_function": 0.9,        # random.randint(10, 90+1) ,
                    "num_generations": 50 ,         # random.randint(50, 300+1) ,
                    "sol_per_pop": 50,              # random.randint(50, 300+1) ,
                    "parent_selection_type": "sss", # ["sss","rws","sus","rank","random" ,"tournament"]
                    "crossover_type": "single_point", # ["single_point","two_points","uniform","scattered"] 
                    "crossover_probability": 0.6 ,  # [0-1]
                    "mutation_type": "random",          # ["random","inversion","scramble","swap"]
                    "mutation_probability": 0.3 ,   # [0-1]
                    "K_tournament": 3               # [3-10]
    }
    
    #print(hist)
    #print(gene_space)
    data = []
    list_thresholds = [1,3,5,7]
    contrib = [0.89,0.9]
    for thres in list_thresholds:
        gene_space["fitness_function"] = contrib[random.randint(0,1)]
        th,f = Segmentation.threshold_GA(config=gene_space, img_hist=hist, num_thresholds=thres)
        print(thres,f)
        data.append([f,th])
    return data
    '''
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 4))
    ax = axes.ravel()

    print("---------------------------------")
    ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[0].set_title('Original Image')
    print("-----------")
    ax[1].bar(bins[:-1],height=hist)
    ax[1].set_title('Histogram')
    print("-----------")
    ax[2].imshow(img2, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax[2].set_title('Gray Image')
    print("-----------")
    ax[3].bar(bins[:-1],height=hist)
    ax[3].set_title('Histogram')
    print("-----------")
    plt.tight_layout()
    plt.show()
    '''
if __name__ == "__main__":
    
    test_images = [
        './images/baboon.png',
        './images/Lenna.png',
        './images/pepper.tiff',
        './images/plane.png',
        './images/house.tiff',
        './images/pubbles.tiff',
    ]
    results = []
    for im in test_images:
        print("------------------------")
        im_results = main(im)
        for r in im_results:
            results.append(r)
        
    df = pd.DataFrame(results, columns=['fitness', 'thresholds'])
    #df.to_csv("Base_results.csv")
    df.to_csv("TA_results.csv")
    #df.to_csv("TA_results.csv")