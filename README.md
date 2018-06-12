# video-classification

Two baseline models for video classification: RNN(frame level) and Average Pooling(video level).

The dataset should be a Python list. Each element in the list is a triplet (video_id, video_frames, video_label).

video_frames is a numpy array with shape (video_length, feature_size).

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
