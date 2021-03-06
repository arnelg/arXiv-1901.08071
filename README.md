# arXiv-1901.08071

example.ipynb is a Jupyter notebook with a minimal example to produce the type of numerical results shown in the preprint arXiv:1901.08071.

The code requires qutip (https://github.com/qutip) to run. Moreover, it requires a recent enhancement of qutip, which is currently a pull request: https://github.com/qutip/qutip/pull/1098. This enhacement is currently available from https://github.com/arnelg/qutip/tree/enhancement-nonsquare-superopreps, and will hopefully be merged into the master branch of qutip soon.

To be able to find optimal recoveries using SDP, the code moreover requires MATLAB to be installed, the python `matlab` engine (https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html), and CVX (http://cvxr.com/cvx/).

This code is made public to improve reproducibility of numerical results in arXiv:1901.08071. The code is not maintained and no effort will be made to make it compatible with software versions other than those listed below.

The code has been tested with the following software:

| Software   | Version         | Version        |
|------------|-----------------|----------------|
| QuTiP      | 4.3.0           | 4.4.0          |
| Numpy      | 1.14.3          | 1.16.3         |
| SciPy      | 1.1.0           | 1.3.0          |
| matplotlib | 2.2.2           | 3.1.0          |
| Python     | 3.6.5           | 3.7.3          |
| MATLAB     | R2017b          |                |
| CVX        | 2.1, Build 1123 |                |
| OS         | posix [darwin]  | posix [darwin] |

Note: Matlab 2017, 2018, and 2019a do not support Python versions >3.6 so the SDP solver wont work for the second configuration.
