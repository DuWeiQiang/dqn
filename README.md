Recurrent DQN-in-the-Caffe
================

This code implements a recurrent LSTM version of DQN.

To run this code, please first download and compile the associated Caffe at:

https://github.com/mhauskn/recurrent-caffe/tree/recurrent

NOTE: Be sure to checkout the recurrent branch of the above repo (not the master).

Sample training command:

./dqn -save test_run -rom pong.bin -unroll=1 -alsologtostderr
