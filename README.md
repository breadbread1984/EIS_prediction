# Introduction

this algoritm is to predict EIS from pulse given to battery with encoder-decoder transformer

# Usage

## download dataset

download dataset from [the link](https://dataset-bohr-storage.dp.tech/lbg%2Fdataset%2Fzip%2Fdataset_tiefblue_bohr_16137_ai4spulseeis-lr97_v100933.zip?Expires=1706012010&OSSAccessKeyId=LTAI5tGCcUT7wz9m1fq8cuLa&Signature=nkwjT0b1%2B2GRyO%2F2SP22v1Aesmo%3D)

## generate datasets

### generate pulse dataset

```shell
python3 create_dataset.py --input_dir <path/to/uncompressed/dataset> --output_dir pulse_dataset --type pulse
```

### generate EIS dataset

```shell
python3 create_dataset.py --input_dir <path/to/uncompressed/dataset> --output_dir eis_dataset --type eis
```

### generate dataset for transformer

```shell
python3 create_dataset.py --input_dir <path/to/uncompressed/dataset> --output_dir transformer_dataset --type transformer
```

## training models

### training auto encoder for pulse data

```shell
python3 train_autoencoder.py --dataset pulse_dataset --type pulse
```

### training auto encoder for EIS data

```shell
python3 train_autoencoder.py --dataset eis_dataset --type eis
```

### training transformer

```shell
python3 train_transformer.py --dataset transformer_dataset
```

## test model

```shell
python3 test.py --dataset <path/to/dataset>
```
