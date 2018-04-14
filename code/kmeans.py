#########################################
#                                       #
# K-means class with three different    #
# implementations: standard, k-means++, #
# and Hammerly's accelerated k-means    #
#                                       #
# Author: Neil Walton                   #
# Date: April 2018                      #
#                                       #
#########################################

import numpy as np
from random import shuffle

class Kmeans:

    def __init__(self, data, k, mode='standard', max_iter=1000):
        self.data = data
        self.k = k
        self.mode = mode
        self.centroids = self._initialize_centroids()
        self.iterations = max_iter

    def _initialize_centroids(self):
        """
        Based on the mode, choose how centroids
        are initialized
        """

        if self.mode == 'standard':
            return self._random_centroids()

        elif self.mode == 'plus':
            return self._plus_plus_centroids()

        elif self.mode == 'hammerly':
            return self._hammerly_centroids()

    def _random_centroids(self):
        """
        Randomly select k values from the data
        as the initial k clusters
        """

        shuffle(self.data)
        return np.array(self.data[:self.k])

    def _plus_plus_centroids(self):
        """
        Initialize centroids using k-means++
        """

        shuffle(self.data)
        centroids = self.data[:1]
        
        for _ in range(self.k - 1):

            distances = self._closest_distances(centroids)
            norm_dists = distances/np.amax(distances)

            rand_num = np.random.random()
            index = np.argmax(norm_dists>rand_num)

            centroids.append(self.data[index])

        return np.array(centroids)


    def _hammerly_centroids(self):
        """
        Initialize centroids using the method
        described by Hammerly
        """

        return None

    def _euclidean_distance(self, point1, point2):
        """
        Compute the Euclidean distance between
        two points
        """

        return np.sqrt(sum((point2 - point1) ** 2))

    def _assign_data(self):
        """
        Assign the data points to their nearest
        centroids, return the resulting clusters
        """

        clusters = [[] for i in range(self.k)]

        for data_point in self.data:

            min_dist = float('inf')
            best_centroid = None

            for j, centroid in enumerate(self.centroids):

                curr_dist = self._euclidean_distance(data_point, centroid)
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    best_centroid = j

            clusters[best_centroid].append(data_point)

        clusters = np.array([np.array(c) for c in clusters])
        return clusters

    def _calculate_centroids(self, clusters):
        """
        Given a cluster assignment, calculate
        the centroid of each cluster
        """

        centroids = np.array([np.mean(c,axis=0) for c in clusters])

        return centroids

    def _closest_distances(self, centroids):
        """
        Return an array of distance of each
        data point to the closest already
        selected centroid for k-means++
        """

        distances = []

        for i, data_point in enumerate(self.data):

            min_dist = float('inf')

            for c in centroids:

                curr_dist = self._euclidean_distance(data_point, c)
                if curr_dist < min_dist:
                    min_dist = curr_dist

            if i==0:
                distances.append(min_dist)
            else:
                distances.append(min_dist+distances[i-1])

        return np.array(distances)       

    def cluster(self):
        """
        Main call to run k-means algorithm
        and return resulting clusters
        """

        for i in range(self.iterations):
            clusters = self._assign_data()
            new_centroids = self._calculate_centroids(clusters)
    
            #When the centroids stop changing, beark early
            if np.array_equal(new_centroids, self.centroids):
                return i+1, clusters

            self.centroids = new_centroids

        return self.iterations, clusters

    def reset(self):
        """
        Reinitialize the centroids so the clustering
        process can be started again
        """

        self.centroids = self._initialize_centroids()

    def average_error(self, clusters):
        """
        Given the current clusters and centroids,
        sum the total distance of the data points
        to their respective centroids
        """

        total_error = 0

        for i, cluster in enumerate(clusters):
            
            for j in cluster:
                total_error += self._euclidean_distance(j, self.centroids[i])

        return total_error/len(self.data)