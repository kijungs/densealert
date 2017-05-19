DenseAlert: Incremental Dense-SubTensor Detection in Tensor Streams
========================

**DenseStream** is an incremental algorithm for detecting dense subtensors in tensor streams, and
**DenseAlert** is an incremental algorithm for spotting suddenly emerging dense subtensors.
They have the following properties:
 * *Fast and 'Any Time'*: By maintaining and updating a dense subtensor, our methods detect a dense subtensor in a tensor stream significantly faster than batch algorithms.
 * *Provably Accurate*: Our methods provide theoretical guarantees on their accuracy, and show high accuracy in practice.
 * *Effective*: Our methods successfully identifies anomalies, such as bot activities, rating manipulations, and network intrusions, in real-world tensors.

Datasets
========================
The download links for the datasets used in the paper are [here](http://www.cs.cmu.edu/~kijungs/codes/alert/)

Building and Running DenseAlert
========================
Please see [User Guide](user_guide.pdf)

Running Demo
========================
For demo, please type 'make'

Reference
========================
If you use this code as part of any published research, please acknowledge the following paper.
```
@inproceedings{shin2017densealert,
  title={DenseAlert: Incremental Dense-SubTensor Detection in Tensor Streams},
  author={Shin, Kijung and Hooi, Bryan and Kim, Jisu and Faloutsos, Christos},
  booktitle={Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2017},
  organization={ACM}
}
```