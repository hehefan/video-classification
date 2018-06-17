# video-classification

## Dynamic RNN

Using all frames that a video exactly has without padding.

## Static RNN
Using a fixed number of frames. 

If video length is greater than the fixed number, uniformly or randomly sampling is used.

If video length is less than the fixed number, all frames are used and the video is padded by 'zero feature's.

## Pooling

### Average Pooling

1. Using all frames

2. Uniformly sampling frames

3. Randomly sampling frames

### Max Pooling

1. Using all frames

2. Uniformly sampling frames

3. Randomly sampling frames

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
```
