# A study on the paper ‘Dilated Recurrent Neural Networks’
@author: Lingsong Zhang (Lance Zhang)
## Phase 1
Analyze and summarize the advantages, disadvantages, applications, and contributions of the paper.<br>
## Phase 2
Pytorch implementation of [Dilated Recurrent Neural Networks].<br>
The test environment: Clemson Palmetto, Python 3.6.0, Jupyter notebook 5.4.0, Pytorch 3.5 and NumPy 1.14. 
```python
cell_type = "LSTM" 
hidden_structs = [20] * 5 
dilations = [1, 2, 4, 8, 16] # 
```
## Phase 3
Pytorch implementation of [Dilated Recurrent Neural Networks].<br>
The test environment: Clemson Palmetto, Python 3.6.0, Jupyter notebook 5.4.0, TensorFlow 1.8 and NumPy 1.14. 
Realize and improve the operation of original LSTM recurrent neural networks, the Dilated Recurrent Neural Networks with the cells of RNN, LSTM and GRU.<br>
Test the performance of these recurrent neural networks on the MNIST digits under two settings, the unpermuted setting and the permuted setting. <br>




## Cite
```
@article{chang2017dilated,
  title={Dilated Recurrent Neural Networks},
  author={Chang, Shiyu and Zhang, Yang and Han, Wei and Yu, Mo and Guo, Xiaoxiao and Tan, Wei and Cui, Xiaodong and Witbrock, Michael and Hasegawa-Johnson, Mark and Huang, Thomas},
  journal={arXiv preprint arXiv:1710.02224},
  year={2017}
}
```
