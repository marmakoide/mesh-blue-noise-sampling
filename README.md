# mesh-blue-noise-sampling

This project demonstrates a simple yet efficient algorithm to compute a blue noise
sampling of a triangle mesh. The number of sample can be set to any value.
The algorithm is described in the paper 
[Sample Elimination for Generating Poisson Disk Sample Sets](http://www.cemyuksel.com/research/sampleelimination/) 
by Cem Yuksel.

![splash screen](https://raw.githubusercontent.com/marmakoide/mesh-blue-noise-sampling/master/splash.png)

## Getting Started

### Prerequisites

You will need

* A Unix-ish environment
* Python 2.7 or above
* [Numpy](http://www.numpy.org)
* [Scipy](http://www.scipy.org)
* [Matplotlib](https://matplotlib.org)
* The [xz](https://en.wikipedia.org/wiki/Xz) compression suite


### Running the demo

The demo 

1. loads an ASCII STL file (a file format for 3d triangle mesh) from the
standard input
2. generates a blue noise sampling of the mesh's surface
3. displays the samples

Using one the sample STL files provided with the naive implementation demo

```
xzcat meshes/fox.stl.xz | python mesh-sampling.py
```

By default, 2048 samples are computed. You can select a different number of
samples, say, 256, using a command-line switch

```
xzcat meshes/fox.stl.xz | python mesh-sampling.py -s256
```


## Implementation notes

This implementation takes a shortcut, by considering Euclidean distances
between samples in the 3d space, not geodesic path lengths on the mesh. In 
practice, for a dense enough sampling, this should have a neglible impact on
the result. However, I plan to add an option to use proper geodesic path lengths.

The algorithm calls for a priority queue, while in this implementation, a plain
sorted list is used. 

## Authors

* **Alexandre Devert** - *Initial work* - [marmakoide](https://github.com/marmakoide)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

