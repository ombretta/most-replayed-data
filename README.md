# Can we predict the Most Replayed data of video streaming platforms?
Official repository of the paper "Can we predict the Most Replayed data of video streaming platforms?" [[ArXiv](https://arxiv.org/pdf/2309.06102)] [[DOI](https://doi.org/10.48550/ARXIV.2309.06102)]

### Dataset
#### Download links
Google Drive: https://drive.google.com/file/d/1R8A7OtA9goaHskOYCxyxBIcZoLoKJU4s/view?usp=sharing

#### Dataset structure
Each key in the H5 file is the id of a single video.
A key corresponds to a Group that contains 2 H5 Datasets, "features" and "heat-markers".

For instance:
```text
/-14Dre9CVjk (Group with VIDEO_ID as the key)
    /features (Dataset with shape (548, 1024), type "<f8")
    /heat-markers  (Dataset with shape (100,), type "<f8")
/-Gm_IKNRqgQ
    ...
```

"features" contains the extracted I3D features of the video

"heat-markers" contains the Most Replayed data from YouTube

To watch the videos you can browse to `youtube.com/watch?v=VIDEO_ID`


### Code
#### Code structure
Entry point: `model/main.py`

User study in `evaluation/user_study/`

Dataset creation scripts in `utils/`
