# ILR - Iterative Local Refinement
Implementation in Pytorch of the Iterative Local Refinement (ILR) algorithm.
The repository contains experiments on the uf20-91 dataset of [SATLIB benchmark](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html), where the ILR algorithm is compared with ADAM for SAT solving. Additional experiments have been made on the famous [MNIST Addition task](https://github.com/ghosthamlet/deepproblog/tree/master/examples/NIPS/MNIST) proposed by [Manhaeve et al.](https://proceedings.neurips.cc/paper/2018/hash/dc5d637ed5e62c36ecb73b654b05ba2a-Abstract.html).

# Instructions - SATLIB benchmark
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

## Change the settings
To modify the settings of the experiments, change the file settings.py.

# Instructions - MNIST addition
0. Move in the correct folder:
```
cd MNIST_Sum
```
1. Run the code for generation of the data:
```
python generate_data.py
```
2. Run the training:
```
python run.py
```

## Change the settings
To change the amount of training samples, substitute the 3000 with the required amount of samples in the MNIST_Sum/generate_data.py file:
```
gather_examples(mnist_train_data, 'train_data', 3000)
```
To change the batch size, learning rate, and number of epochs, change the MNIST_Sum/run.py.

## License

Copyright (c) 2022, Daniele Alessandro, Emile van Krieken
All rights reserved.

Licensed under the MIT License.
