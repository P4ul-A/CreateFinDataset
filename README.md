# Finwave Pipeline

A tool to engage with the various modules of the finwave pipeline for extracting
and verifying fins.

# Local Setup
1. Download the source code
2. In the terminal: 
```bash
    cd src
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
    python3 finwave_pipeline.py
```

# Running the Binaries:
1. Visit [the releases page](https://github.com/alexanderbarnhill/FinwavePipeline/releases`) 
and fetch the relevant binaries for your system
2. Download and extract the zip file
3. Run the `finwave_pipeline` file (or `finwave_pipeline.exe` on windows)

# Settings
The most important settings are 
- `input_directory` : the directory from which the tool will look for images
- `output_directory` : Where the files will be written extraction / verification

***make sure to hit 'save' before running the pipeline***

# Orca ID Clustering
Cluster all original JPEG images captured within a time window around each
manually identified ID image:

```bash
python3 src/cluster_orca_id_images.py \
  --id-dir jpeg_output/Orca_ID/2022-09-07_Andøya_RichardKaroliussen_IDs \
  --original-dir jpeg_output/Orca_ID/2022-09-07_Andøya_RichardKaroliussen_All_pictures \
  --output-dir output/orca_id_clusters \
  --timespan 2
```

The script writes one folder per ID name and copies the ID image plus matching
original images into it. It uses JPEG EXIF timestamps when available and falls
back to file modification time unless `--no-mtime-fallback` is set.

To create one visual verification summary slide per ID, with boxes drawn on the
manual ID image(s) and each additional clustered image, add `--draw-boxes`:

```bash
python3 src/cluster_orca_id_images.py \
  --id-dir jpeg_output/Orca_ID/2022-09-07_Andøya_RichardKaroliussen_IDs \
  --original-dir jpeg_output/Orca_ID/2022-09-07_Andøya_RichardKaroliussen_All_pictures \
  --output-dir output/orca_id_clusters \
  --timespan 2 \
  --draw-boxes \
  --max-box-movement-per-step 300 \
  --max-box-size-change-ratio 2.0
```

This calls the `/fin-detect` API and writes `summary__<ID>.jpg` in each ID
output folder. No separate `boxed__...jpg` images are written.

The largest detected fin box in each manual ID image is used as the reference.
Additional images are kept when their closest fin box center moves no more than
`--max-box-movement-per-step` pixels per image step away from the manual image.
They must also pass `--max-box-size-change-ratio`; `2.0` means the candidate
box area can be between half and double the manual reference box area. Kept
boxes are green; discarded boxes are grey.

Approved green boxes are also cropped into a `cropped/` directory inside each
ID output folder.
