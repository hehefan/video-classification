# video-classification

## Dynamic RNN (LSTM, GRU)

Using all frames that a video exactly has without padding.

## Static RNN (LSTM, GRU)
Using a fixed number of frames. 

If video length is greater than the fixed number, uniformly or randomly sampling is used.

If video length is less than the fixed number, all frames are used and the video is padded by 'zero feature's.

## Pooling

### Average Pooling

1. Using all frames.

2. Uniformly sampling a fixed number of frames. If video length is less than the fixed number, all frames are used.

3. Randomly sampling a fixed number of frames. If video length is less than the fixed number, all frames are used.

### Max Pooling

1. Using all frames.

2. Uniformly sampling a fixed number of frames. If video length is less than the fixed number, all frames are used.

3. Randomly sampling a fixed number of frames. If video length is less than the fixed number, all frames are used.

### [ConvLSTM](https://arxiv.org/pdf/1506.04214v1.pdf)

For ConvLSTM, inputs (frame/image features) and states are 3D tensors (Side, Side, Channel). 

Instead of full connections, operations in ConvLSTM are convlutions.

Continuing...

If you find this code useful, please consider citing my video-related works:
```
@inproceedings{fan17iccv,
  author    = {Hehe Fan and Xiaojun Chang and De Cheng and Yi Yang and Dong Xu and Alexander G. Hauptmann},
  title     = {Complex Event Detection by Identifying Reliable Shots from Untrimmed Videos},
  booktitle = {{IEEE} International Conference on Computer Vision, {ICCV} 2017, Venice, Italy, October 22-29, 2017},
  pages     = {736--744},
  year      = {2017},
  doi       = {10.1109/ICCV.2017.86}
}
@inproceedings{fan18ijcai,
  author    = {Hehe Fan and Zhongwen Xu and Linchao Zhu and Chenggang Yan and Jianjun Ge and Yi Yang},
  title     = {Watching a Small Portion could be as Good as Watching All: Towards Efficient Video Classification},
  booktitle = {International Joint Conference on Artificial Intelligence, {IJCAI} 2018, Stockholm, Sweden, July 13-19, 2018},
  pages     = {705--711},
  year      = {2018},
  doi       = {10.24963/ijcai.2018/98}
}
```
