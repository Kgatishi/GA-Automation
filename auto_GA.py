#from glob import glob
import pygad
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from skimage import data, io
from PIL import Image
import time

from segmentation_GA import Segmentation

# fitness function ...............
class automation:

    def __init__(self, im, im_name):

        self.image_name = im_name
        img = io.imread(im)
        img2 = np.array(Image.fromarray(img).convert('L'))
        hist, bins = np.histogram(img2, bins=range(256), density=False)

        self.histogram = hist
        self.list_thresholds = [2,3,4,5]
        self.n_thresh = 0

        self.best_solution_thresholds = []
        self.best_solution_fitness = -100
        self.best_configuration = {}
    
    def fitness_function(self, ga_instanse, individual, individual_idx):
        bitstring = [ int(i) for i in individual]
        config = {}
        # ----------------------------------------------------
        config["num_generations"] = bitstring[0]
        config["sol_per_pop"] = bitstring[1]              # population size max 100

        ps = ["sss"  , "rws" , "sus"  , "rank"  , "random"   , "tournament"  ]
        config["parent_selection_type"] = ps[ bitstring[2] ]
        config["K_tournament"] = bitstring[-1]

        ct = ["single_point", "two_points" , "uniform",  "scattered"] 
        config["crossover_type"] = ct[ bitstring[3] ]
        config["crossover_probability"] = bitstring[4] * 0.01

        mt = [ "random" , "inversion" , "scramble" , "swap" ]
        config["mutation_type"] = mt[ bitstring[5] ]
        config["mutation_probability"] =  bitstring[6] * 0.01

        # ----------------------------------------------------
        solution, solution_fitness = Segmentation( self.histogram.copy(), _otsu=True, _kapur=False).threshold_GA(config, self.n_thresh)
        
        if solution_fitness > self.best_solution_fitness:
            self.best_solution_fitness = solution_fitness
            self.best_solution_thresholds = solution
            self.best_configuration = config
        
        #print(solution, solution_fitness)
        return solution_fitness

    #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    def automating_GA(self, n_thresh):
        
        #mutation_t = 3
        #if automation.num_thresholds <= 1: mutation_t = 2
        self.n_thresh = n_thresh
        gene_space =   [
                        {'low': 50, 'high': 300, 'step': 1},    # Number of generation 
                        {'low': 50, 'high': 300, 'step': 1},    # Population size
                        {'low': 0, 'high': 5, 'step': 1},       # Parent_selection_type
                        {'low': 0, 'high': 3, 'step': 1},       # crossover_type
                        {'low': 10, 'high': 90, 'step': 1},     # crossover_probability
                        {'low': 0, 'high': 3, 'step': 1},       # mutation_type
                        {'low': 10, 'high': 90, 'step': 1},     # mutation_probability
                        {'low': 3, 'high': 10, 'step': 1},      # K_tournament
                    ]

        ga_instance = pygad.GA(num_generations=30,
                            num_parents_mating=2,
                            fitness_func=self.fitness_function,
                            sol_per_pop=30,
                            num_genes=8,
                            gene_space=gene_space,
                            gene_type=int,
                            parent_selection_type="tournament",
                            crossover_type="single_point",
                            mutation_type="random",
                            mutation_probability=0.2,
                            K_tournament=6,
                            random_seed=2
                        )
        #print( ga_instance.initial_population)
        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        return solution, solution_fitness

    def iterate(self):

        data = []
        for thres in self.list_thresholds:

            t1 = time.time()
            solution, fitness = self.automating_GA(thres)
            t2 = time.time()
            
            print("-----------------------------------------------------------------------")
            print("Time is", t2-t1)
            print( self.image_name , thres, self.best_configuration, fitness, self.best_solution_thresholds, t2-t1  )
            print("-----------------------------------------------------------------------")
            data.append( [self.image_name, thres, self.best_configuration, fitness, self.best_solution_thresholds, t2-t1 ] )
            #break
        return data

if __name__ == "__main__":
    test_images = [
        './images/lenna.png',
        './images/pepper.tiff',
        './images/house.tiff',
        './images/boats.bmp',
        './images/lake.bmp',
        './images/airplane.bmp'
        
    ]
    image_name = ['lenna','peppers','house','boats','lake','airplane']

    results = []
    for im,im_name in zip(test_images, image_name):
        im_results = automation(im, im_name).iterate()

        for r in im_results:
            results.append(r)
        #break
    df = pd.DataFrame( results, columns=['image','num_threshold','config','fitness Otsu','thresholds','time'])
    df.to_csv("GA_results.csv")
        