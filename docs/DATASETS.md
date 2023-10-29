## Download

### Pre-Training 

- CC12M images, https://github.com/google-research-datasets/conceptual-12m
- CC3M images, https://github.com/google-research-datasets/conceptual-captions
- SBU images, https://www.cs.rice.edu/~vo9/sbucaptions/
- COCO images, https://cocodataset.org/#download
- VG images, https://visualgenome.org/api/v0/api_home.html
- WebVid videos, https://github.com/m-bain/webvid

For datasets that only provide urls, you may use [img2dataset](https://github.com/rom1504/img2dataset) to speed up downloading.

### Downstream

- QuerYD videos, https://github.com/oncescuandreea/QuerYD_downloader
- ActivityNet videos, http://activity-net.org/download.html
- DiDeMo videos, https://github.com/LisaAnne/LocalizingMoments
- Condensed Movies videos, https://github.com/m-bain/CondensedMovies

We use the annotations provide by [VINDLU](https://github.com/klauscc/VindLU#data), the files can be found in [Google Drive](https://drive.google.com/drive/folders/12cr94wT8j7pR09AR2nmQg6o26Y1arI50).

Put your data following the following structure:
```bash
didemo
    |-- didemo_30fps_224_trimed30     
        |-- 44369196@N06_4552877020_7d6db1f971.mp4
        |-- ...
    |-- train.jsonl
    |-- val.jsonl
    |-- test.jsonl
    |-- video_names.txt

Activitynet_Captions
    |-- anet_6fps_224     
        |-- v_wlYxVUJSJVI.mp4
        |-- ...
    |-- train.json
    |-- val.json
    |-- video_names.txt

QuerYD
    |-- QuerYD_downloader
        |-- videos
            |-- video-wkP0rYhSrLQ
            |-- ...
    |-- QuerYD-experts
        |-- data
            |-- QuerYD
                |-- structured-symlinks
                    |-- exists_train_list.txt
                    |-- exists_val_list.txt
                    |-- exists_test_list.txt
                    |-- raw_captions_combined_filtered.pkl
CondensedMovies
    |-- metadata
        |-- train_val_challf0.csv
        |-- test_challf0.csv
    |-- videos     
        |-- v_wlYxVUJSJVI.mkv
        |-- ...

Activitynet-QA
    |-- anet_6fps_224     
        |-- v_wlYxVUJSJVI.mp4
        |-- ...
    |-- annos
        |-- train_q.json
        |-- train_a.json
        |-- val_q.json
        |-- val_a.json
        |-- test_q.json
        |-- test_a.json
    |-- video_names.txt
```

### Compressing Videos
We preprocess videos to lower FPS and dimension to reduce storage and to improve data loading. For videos, you may use
```bash
ls -U /path/to/video >> /path/to/video_names.txt

# for DiDeMo
python data/compress.py \
--input_root=/home/renshuhuai/data/didemo/videos \
--output_root=/home/renshuhuai/data/didemo/didemo_30fps_224_trimed30 \
--input_file_list_path=/home/renshuhuai/data/didemo/video_names.txt \
--duration 30 --fps=30 --size=224 --file_type=video --num_workers 24

# for ActivityNet
python data/compress.py \
--input_root=/home/renshuhuai/data/Activitynet-QA/videos \
--output_root=/home/renshuhuai/data/Activitynet-QA/anet_6fps_224 \
--input_file_list_path=/home/renshuhuai/data/Activitynet-QA/video_names.txt \
--fps=6 --size=224 --file_type=video --num_workers 24
```

Note that the audio is also removed from the video files.