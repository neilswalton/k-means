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
        self.data_dimensions = len(self.data[0])

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
            return self._plus_plus_centroids()

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

        if self.mode == 'hammerly':
            iters, clusters = self._run_hammerly()
            return iters, clusters
        else:
            iters, clusters = self._run_kmeans()
            return iters, clusters

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

    def _run_kmeans(self):
        """
        Run non-hammerly k-means
        """

        self._initialization()

        for iteration in range(self.iterations):

            old_centroids = np.copy(self.centroids)

            for i in range(len(self.data)):
                
                a_prime = self.cluster_indexes[i]
                self._point_all_centers_2(i)

                #data point i has changed clusters
                if a_prime != self.cluster_indexes[i]:
                    self.points_per_cluster[a_prime] -= 1
                    self.points_per_cluster[self.cluster_indexes[i]] += 1
                    self.cluster_sums[a_prime] = \
                        self.cluster_sums[a_prime] - self.data[i]
                    self.cluster_sums[self.cluster_indexes[i]] = \
                        self.cluster_sums[self.cluster_indexes[i]] + self.data[i]

            self._move_centers_2()           

            if np.array_equal(self.centroids, old_centroids):
                clusters = self._assign_data()
                return iteration+1, clusters

        clusters = self._assign_data()
        return self.iterations, clusters

    def _run_hammerly(self):
        """
        Main Hammerly algorithm
        """

        self._hammerly_initialization()

        for iteration in range(self.iterations):

            old_centroids = np.copy(self.centroids)

            #find minimum intracluster distances
            for j in range(self.k):
                self.cluster_to_cluster_dist[j] = self._cluster_distances(j)

            for i in range(len(self.data)):
                m = min(self.cluster_to_cluster_dist[self.cluster_indexes[i]]/2,
                    self.lower_bounds[i])

                if self.upper_bounds[i] > m:
                    #Tighten the upper bound
                    self.upper_bounds[i] = self._euclidean_distance(self.data[i],
                        self.centroids[self.cluster_indexes[i]])

                    if self.upper_bounds[i] > m:
                        a_prime = self.cluster_indexes[i]
                        self._point_all_centers(i)

                        #data point i has changed clusters
                        if a_prime != self.cluster_indexes[i]:
                            self.points_per_cluster[a_prime] -= 1
                            self.points_per_cluster[self.cluster_indexes[i]] += 1
                            self.cluster_sums[a_prime] = \
                                self.cluster_sums[a_prime] - self.data[i]
                            self.cluster_sums[self.cluster_indexes[i]] = \
                                self.cluster_sums[self.cluster_indexes[i]] + self.data[i]

            self._move_centers()
            self._update_bounds()            

            if np.array_equal(self.centroids, old_centroids):
                clusters = self._assign_data()
                return iteration+1, clusters

        clusters = self._assign_data()
        return self.iterations, clusters

    def _hammerly_initialization(self):
        """
        Initialize variables using the method
        described by Hammerly
        """

        self.points_per_cluster = np.zeros(self.k).astype('int')
        self.cluster_sums = np.zeros((self.k, self.data_dimensions))
        self.upper_bounds = np.zeros(len(self.data))
        self.lower_bounds = np.zeros(len(self.data))
        self.cluster_indexes = np.zeros(len(self.data)).astype('int')
        self.cluster_to_cluster_dist = np.zeros(self.k)
        self.cluster_movement = np.zeros(self.k)

        for i in range(len(self.data)):

            self._point_all_centers(i)
            self.points_per_cluster[self.cluster_indexes[i]] += 1
            self.cluster_sums[self.cluster_indexes[i]] = np.add(
                self.cluster_sums[self.cluster_indexes[i]], self.data[i])

    def _initialization(self):
        """
        Initialize variables for Lloyd's algorithm
        """

        self.points_per_cluster = np.zeros(self.k).astype('int')
        self.cluster_sums = np.zeros((self.k, self.data_dimensions))
        self.cluster_indexes = np.zeros(len(self.data)).astype('int')

        for i in range(len(self.data)):

            self._point_all_centers_2(i)
            self.points_per_cluster[self.cluster_indexes[i]] += 1
            self.cluster_sums[self.cluster_indexes[i]] = np.add(
                self.cluster_sums[self.cluster_indexes[i]], self.data[i])

    def _point_all_centers(self, data_index):
        """
        Assign data_point to its cluster, update
        its upper and lower bounds
        """

        min_1 = float('inf')
        min_2 = float('inf')
        clust_ind = -1

        for i, c in enumerate(self.centroids):
            curr_dist = self._euclidean_distance(c, self.data[data_index])
            if curr_dist < min_1:
                min_1 = curr_dist
                clust_ind = i
            elif curr_dist < min_2:
                min_2 = curr_dist

        self.cluster_indexes[data_index] = clust_ind
        self.upper_bounds[data_index] = min_1
        self.lower_bounds[data_index] = min_2

    def _point_all_centers_2(self, data_index):
        """
        Assign data_point to its cluster, update
        its upper and lower bounds, ignore any
        Hammerly-specific variables
        """

        min_1 = float('inf')
        clust_ind = -1

        for i, c in enumerate(self.centroids):
            curr_dist = self._euclidean_distance(c, self.data[data_index])
            if curr_dist < min_1:
                min_1 = curr_dist
                clust_ind = i

        self.cluster_indexes[data_index] = clust_ind

    def _move_centers(self):
        """
        Update the centroids and the
        centroid movement
        """

        for i in range(self.k):
            c_star = np.copy(self.centroids[i])
            self.centroids[i] = self.cluster_sums[i]/self.points_per_cluster[i]
            self.cluster_movement[i] = self._euclidean_distance(c_star, self.centroids[i])

    def _move_centers_2(self):
        """
        Update the centroids and the
        centroid movement, ignoring
        any Hammerly-specific variables
        """

        for i in range(self.k):
            self.centroids[i] = self.cluster_sums[i]/self.points_per_cluster[i]

    def _update_bounds(self):
        """
        Based on the cluster movement,
        update the bounds
        """

        #Find the two clusters that moved the most
        args = np.argsort(self.cluster_movement)
        r = args[0]
        r_prime = args[1]
        
        for i in range(len(self.upper_bounds)):
            
            #update upper bound
            self.upper_bounds[i] = self.upper_bounds[i] \
                + self.cluster_movement[self.cluster_indexes[i]]

            #data point i's cluster moved the most
            if r == self.cluster_indexes[i]:
                self.lower_bounds[i] = self.lower_bounds[i] - self.cluster_movement[r_prime]

            #data point i's cluster did not move the most
            else:
                self.lower_bounds[i] = self.lower_bounds[i] - self.cluster_movement[r]

    def _cluster_distances(self, cluster):
        """
        Find the minimum distance to
        the nearest cluster, given 
        a cluster
        """

        min_dist = float('inf')

        for i, c in enumerate(self.centroids):
            if i!=cluster:
                curr_dist = self._euclidean_distance(c, self.centroids[cluster])
                if curr_dist < min_dist:
                    min_dist = curr_dist

        return min_dist