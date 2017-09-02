# Video-Classificationi

Two baseline models for video classification: RNN(frame level) and Average Pooling(video level).

The dataset should be a Python list. Each element in the list is a triplet (video_id, video_frames, video_label).

video_frames is a numpy array with shape (video_length, feature_size).
