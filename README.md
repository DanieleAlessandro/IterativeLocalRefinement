# ILR - Iterative Local Refinement
Implementation in Pytorch of the Iterative Local Refinement (ILR) algorithm.
The repository contains experiments on the uf20-91 dataset of [SATLIB benchmark](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html), where the ILR algorithm is compared with ADAM for SAT solving.

# Instructions
1. Download the [uf20-91 dataset](https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf20-91.tar.gz). 
2. Uncompress the tar.gz and copy the uf20-91 folder inside the project folder
3. Ensure you have PyTorch, MatplotLib, Numpy, etc installed.
4. Run the code of the experiments:
```
python SAT.py
```
5. Run the code to generate the plots:
```
python draw_plots.py
```
6. To visualize the results, check the plots_final folder.

# Change the settings
To modify the settings of the experiments, change the file settings.py.

## License

Copyright (c) 2022, Daniele Alessandro, Emile van Krieken
All rights reserved.

Licensed under the MIT License.
