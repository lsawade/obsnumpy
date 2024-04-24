# Combining obspy and numpy to accelerate processing.

This is an intermediate package to streamline the processing from
obspy's incredible slow loops to actions in from of numpy operations.
The main goal however is to immitate.

The objective is to translate Stream.
- rotate
- interpolate
- bandpass filter

The package will assume that the user did the bare minimum of processing the data into NEZ traces and removed the instrument response.

There will be a function that takes in a stream as an argument that will interpolate each trace separately to make it a single array, but that's really not the purpose of this package, but rather a convenience function.


For the source inversion one could think of the following:

After the initial windowing, we can select the traces that have windows and compute corresponding ArrayStreams from the windows

- a taper array
- a weight array
- a normalizing array
    * We want this to be an array because we cannot divide the array after
      summation. We only have to compute this once. since we are normalizing by the data (See Sawade et al. 2022)

Then, the array formulation for the cost function simplifies to

C(m) = 0.5 * sum_k { sum(W * [Dk-Sk(m)]**2 / Nk), }

where W is the combined window and weight array, D is the Data array, S is the simulated data array, and N is the Data normalization array. k denotes the category.

