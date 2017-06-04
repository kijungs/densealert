=================================================================================

 DenseAlert: Incremental Dense-SubTensor Detection in Tensor Streams
 Authors: Kijung Shin, Bryan Hooi, Jisu Kim, and Christos Faloutsos

 Version: 1.0
 Date: Feb 3, 2017
 Main Contact: Kijung Shin (kijungs@cs.cmu.edu)

 This software is free of charge under research purposes.
 For commercial purposes, please contact the author.

=================================================================================

DenseStream is an incremental algorithm for detecting dense subtensors in tensor streams, and
DenseAlert is an incremental algorithm for spotting suddenly emerging dense subtensors.
They have the following properties:
- Fast and 'Any Time': By maintaining and updating a dense subtensor, our algorithms detect a dense subtensor in a tensor stream significantly faster than batch algorithms.
- Provably Accurate: Our algorithms provide theoretical guarantees on their accuracy, and show high accuracy in practice.
- Effective: Our algorithms successfully identifies anomalies, such as bot activities, rating manipulations, and network intrusions, in real-world tensors.

For detailed information, see 'user_guide.pdf'

For demo, type 'make'