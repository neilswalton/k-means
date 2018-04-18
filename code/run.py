#########################################
#                                       #
# Main program that runs k-means on     #
# the specified dataset                 #
#                                       #
# Author: Neil Walton                   #
# Date: April 2018                      #
#                                       #
#########################################

import utils
import kmeans
import time


def main():
    """
    Main call to start the program
    """

    #Dataset
    file_name = 'frogs.csv'
    file_path = '../data/'
    full_path = file_path+file_name
    data = utils.csv_to_arrays(full_path)

    #Parameters
    k = 3
    mode = 'hammerly'
    iterations = 1000
    tests = 10
    model = kmeans.Kmeans(data, k, mode, iterations)

    #Book keeping variables
    total_iterations = 0
    total_dist = 0
    total_time = 0
    
    print("-----" + mode + "-----")
    
    for i in range(tests):

        start = time.time()
        iters, clusters = model.cluster()
        stop = time.time()
        error = model.average_error(clusters)

        run_time = stop-start
        total_time += run_time

        #print ("Stopped after " + str(iters) + 
        #    " iterations, with average distance of: " + str(error))
        model.reset()

        total_dist += error
        total_iterations += iters

    print ("average iterations: " + str(total_iterations/tests))
    print ("average distance: " + str(total_dist/tests))
    print ("average time: " + str(total_time/tests))

if __name__ == '__main__':
    main()