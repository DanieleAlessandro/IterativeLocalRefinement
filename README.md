# ILR - Iterative Local Refinement
Implementation in Pytorch of the Iterative Local Refinement (ILR) algorithm.
The repository contains experiments on the uf20-91 dataset of [SATLIB benchmark](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html), where the ILR algorithm is compared with adam for SAT solving.

# Instructions
1. Download the [uf20-91 dataset](https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf20-91.tar.gz). 
2. Uncompress the tar.gz and copy the uf20-91 folder inside the project folder
3. Run the code of the experiments:
```
python SAT.py
```
4. Run the code forplots generation:
```
python draw_plots.py
```
5. To visualize the results, check the plots_final folder.

# Change the settings
To modify the settings of the experiments, change the file settings.py.
