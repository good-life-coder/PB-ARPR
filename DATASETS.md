# Dataset acquisition and legal boundaries

Official pages checked on 2026-10-21. This archive contains no dataset copies, images, masks, RLE labels or per-image private predictions. Reviewers obtain data independently and comply with upstream terms.

## MVTec AD 2 / sheet_metal
[Official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2). CC BY-NC-SA 4.0; cite the dataset and comply with attribution/noncommercial/share-alike requirements. Obtain the official sheet_metal release and preserve train/good, validation/good and test_public structure under datasets/MVTec_AD_2/sheet_metal. Frozen counts: 137 normal training,19 normal validation,114 public evaluation (24 normal,90 anomaly). Native 1056x4224. No test_private/test_private_mixed contents are supplied or required for the public results. Private leaderboard submissions are excluded.

## KSDD2
[Official ViCoS page and download link](https://www.vicos.si/resources/kolektorsdd2/). CC BY-NC-SA 4.0; commercial use requires contacting the provider. Official train 2085 normal+246 anomalous, test894 normal+110 anomalous. Preserve filename-to-_GT.png pairing. The historical second-paper protocol uses1876 normal training,209 normal calibration,455 DEV and1004 official test. This is NOT the later third-paper split. Frozen split hashes/protocol and deterministic source are included, but label-bearing split manifests are not redistributed. Do not casually regenerate a different split and call it the historical run; compare authorized originals to frozen identity hashes. Historical test dimension use is disclosed.

## Severstal
[Official acquisition](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data) and [competition rules](https://www.kaggle.com/competitions/severstal-steel-defect-detection/rules). Kaggle login and acceptance of the competition terms are required. The currently accessible rules restrict data use and redistribution; this package does not assert a general research redistribution license. Obtain clarification/permission where your proposed use requires it. Do not transmit competition images or labels with this archive.
The paper used an internal split of official annotated training images, NOT hidden Kaggle competition test labels:4131 normal training,590 validation,1923 DEV,6514 FINAL_BLIND. Native256x1600; union-of-class RLE masks. Only aggregate FINAL_BLIND evidence and protocol are supplied. Exact private identities, RLE strings, per-image metrics and caches are withheld. Full blind reproduction requires appropriately authorized access and is UNVERIFIED here.

## ImageNette auxiliary input
[Maintainer repository/downloads](https://github.com/fastai/imagenette). Obtain the historically required ImageNette variant through upstream links and respect underlying ImageNet/image rights; a repository software license does not license all images. Preserve the sample collection and frozen transform order. No images are bundled.
