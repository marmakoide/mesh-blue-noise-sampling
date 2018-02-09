import sys
import numpy
import argparse

from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree

import stlparser

import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D



def norm2(X):
	return numpy.sqrt(numpy.sum(X ** 2, axis = 0))



def triangle_surface_area(triangle):
	A, B, C = triangle
	return .5 * norm2(numpy.cross(B - A, C - A))



reflection = numpy.array([[0., -1.], [-1., 0.]])
def triangle_picking(triangle):
	X = numpy.random.random(2)
	if numpy.sum(X) > 1:
		X = numpy.dot(X, reflection) + 1.

	A, B, C = triangle
	U, V = B - A, C - A
	return numpy.dot(numpy.array([U, V]).T, X) + A



def uniform_sample_mesh(triangle_list, sample_count):
	# Compute area of each triangle, normalize the sum to 1
	triangle_area = numpy.array([triangle_surface_area(tri) for tri in triangle_list])
	triangle_area /= numpy.sum(triangle_area)

	# For each sample
	ret = numpy.zeros((sample_count, 3))
	for i in range(sample_count):
		# Pick a triangle, with probability proportional to its surface area
		j = numpy.random.choice(triangle_list.shape[0], p = triangle_area)

		# Pick a point with uniform probability on the triangle
		ret[i] = triangle_picking(triangle_list[j])

	# Job done
	return ret



def blue_noise_sample_elimination(point_list, mesh_surface_area, sample_count):
	# Parameters
	alpha = 8
	rmax = numpy.sqrt(mesh_surface_area / ((2 * sample_count) * numpy.sqrt(3.))) 

	# Compute a KD-tree of the input point list
	kdtree = KDTree(point_list)

	# Compute the weight for each sample
	D = numpy.minimum(squareform(pdist(point_list)), 2 * rmax)
	D = (1. - (D / (2 * rmax))) ** alpha

	W = numpy.zeros(point_list.shape[0])
	for i in range(point_list.shape[0]):
		W[i] = sum(D[i, j] for j in kdtree.query_ball_point(point_list[i], 2 * rmax) if i != j)

	# Pick the samples we need
	heap = []
	for i, w in enumerate(W):
		heap.append((w, i))
	heap.sort()

	id_set = set(range(point_list.shape[0]))
	while len(id_set) > sample_count:
		# Pick the sample with the highest weight
		w, i = heap.pop()
		id_set.remove(i)

		neighbor_set = set(kdtree.query_ball_point(point_list[i], 2 * rmax))
		neighbor_set.remove(i)
		heap = [(w - D[i, j], j) if j in neighbor_set else (w, j) for w, j in heap]				
		heap.sort()

	# Job done
	return point_list[sorted(id_set)]



def main():
	# Command line parsing
	parser = argparse.ArgumentParser(description = 'Compute and show a blue noise sampling of a triangul mesh')
	parser.add_argument('-n', '--sample-count', type = int, default = 2048, help = 'number of sample to compute')
	args = parser.parse_args()

	# Load the input mesh as a list of triplets (ie. triangles) of 3d vertices
	try:
		triangle_list = numpy.array([X for X, N in stlparser.load(sys.stdin)])
	except stlparser.ParseError as e:
		sys.stderr.write('%s\n' % e)
		sys.exit(0)

	# Compute an uniform sampling of the input mesh
	point_list = uniform_sample_mesh(triangle_list, 4 * args.sample_count)

	# Compute a blue noise sampling of the input mesh, seeded by the previous sampling
	mesh_surface_area = sum(triangle_surface_area(tri) for tri in triangle_list)
	point_list = blue_noise_sample_elimination(point_list, mesh_surface_area, args.sample_count)

	# Display
	fig = plot.figure()
	ax = fig.gca(projection = '3d')
	ax._axis3don = False
	ax.set_aspect('equal')
	ax.scatter(point_list[:,0], point_list[:,1], point_list[:,2], lw = 0., c = 'k')
	plot.show()
	


if __name__ == '__main__':
	main()
