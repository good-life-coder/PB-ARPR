"""Lossless derived four-tile dataset. Never writes into the source dataset."""
import csv
import hashlib
import json
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"datasets/MVTec_AD_2/sheet_metal"
DERIVED=ROOT/"datasets/derived/sheet_metal_tile4"
OUT=ROOT/"experiments/phase09b_p0_tile_smoke"
FIELDS=["split","source_image_path","source_image_id","source_label","tile_id","x0","x1","y0","y1",
        "tile_image_path","source_mask_path","tile_mask_path","tile_positive_pixels","tile_is_anomalous"]
COORDS=[(i*1056,(i+1)*1056,0,1056) for i in range(4)]


def manifest_hash():
    return hashlib.sha256((DERIVED/"manifest.csv").read_bytes()).hexdigest()


def save_report(r):
    DERIVED.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    text=json.dumps(r,indent=2,ensure_ascii=False)
    (DERIVED/"dataset_report.json").write_text(text+"\n",encoding="utf-8")
    (OUT/"tile_dataset_report.json").write_text(text+"\n",encoding="utf-8")
    (OUT/"tile_dataset_report.txt").write_text(text+"\n",encoding="utf-8")


def build():
    # Refusing a second build must not invalidate the existing PASS report.
    if (DERIVED/"manifest.csv").exists():
        message="Derived manifest already exists; refusing to overwrite the dataset or its reports"
        print(message,flush=True)
        return dict(status="DATASET_BUILD_FAIL",error_message=message)
    import numpy as np
    from PIL import Image
    import inspect
    from anomalib.data import Folder
    start=time.perf_counter()
    r=dict(status="DATASET_BUILD_FAIL",source_dataset_root=str(SOURCE),derived_root=str(DERIVED),
           source_counts={},tile_counts={},source_anomaly_images=90,image_reconstruction_pass_count=0,
           mask_reconstruction_pass_count=0,file_corruption_count=0,tile_size_distribution={},
           tile_coordinates=COORDS,tile_native_size=[1056,1056],output_format="PNG lossless, no resize",
           folder_api=str(inspect.signature(Folder)),
           datamodule="Project Tile4DataModule(Folder), official FolderDataset; fixed independent validation",
           sampling_relation="B0: 137 images; P0: 548 tiles. Uniform tile sampling gives source probability 4/548=1/137. This is a sampling relation, not a performance-fairness claim.",
           error_message=None)
    try:
        geometry=json.loads((ROOT/"experiments/phase09a_tile_geometry/tile_geometry_report.json").read_text(encoding="utf-8"))
        if geometry["status"]!="PASS" or not geometry["P0_geometry_feasible"]:
            raise ValueError("Phase09A geometry audit must be PASS and geometrically feasible")
        r["source_audit_result"]="Phase09A PASS; source counts/native arrays independently checked below"
        coverage=np.zeros(4224,dtype=np.uint8)
        for x0,x1,y0,y1 in COORDS:
            assert x1-x0==y1-y0==1056
            coverage[x0:x1]+=1
        assert np.array_equal(coverage,np.ones(4224,dtype=np.uint8))
        r["x_coverage_exactly_once"]=True
        sources=[]
        for split,expected in (("train",137),("validation",19),("test_public",114)):
            found=[]
            for label,folder in ((0,"good"),(1,"bad")):
                directory=SOURCE/split/folder
                if directory.exists():
                    found.extend((split,label,p) for p in sorted(directory.iterdir()) if p.suffix.lower() in (".png",".jpg",".jpeg",".tif",".tiff",".bmp"))
            if len(found)!=expected or (split!="test_public" and any(label for _,label,_ in found)):
                raise ValueError(f"Unexpected source split count/labels: {split}")
            r["source_counts"][split]=len(found);sources.extend(found)
        if sum(label for split,label,_ in sources if split=="test_public")!=90:
            raise ValueError("Public source labels must be 90 abnormal /24 normal")
        for split in r["source_counts"]:
            (DERIVED/split/"images").mkdir(parents=True,exist_ok=True)
            (DERIVED/split/"masks").mkdir(parents=True,exist_ok=True)
        rows=[];test_abnormal=0
        for index,(split,label,path) in enumerate(sources):
            source_mask=SOURCE/"test_public/ground_truth/bad"/(path.stem+"_mask"+path.suffix) if label else None
            with Image.open(path) as image:
                original=np.array(image)
                if image.size!=(4224,1056):raise ValueError(f"Unexpected source dimensions: {path}")
                original_mask=None
                if label:
                    with Image.open(source_mask) as mask:
                        if mask.size!=image.size:raise ValueError("Source mask dimensions differ")
                        original_mask=np.array(mask)
                        if original_mask.ndim!=2:raise ValueError("Expected a single-channel original mask")
                tiles=[];masks=[]
                for tile_id,(x0,x1,y0,y1) in enumerate(COORDS):
                    piece_mask=original_mask[:,x0:x1].copy() if label else np.zeros((1056,1056),dtype=np.uint8)
                    positive=int(np.count_nonzero(piece_mask));abnormal=int(positive>0)
                    name=("bad" if label else "good")+"__"+path.stem+f"__tile{tile_id}.png"
                    classdir=("anomaly" if abnormal else "normal") if split=="test_public" else ""
                    image_path=DERIVED/split/"images"/classdir/name
                    image_path.parent.mkdir(parents=True,exist_ok=True)
                    image.crop((x0,y0,x1,y1)).save(image_path,format="PNG",compress_level=1)
                    mask_path=None
                    if split=="test_public":
                        mask_path=DERIVED/split/"masks"/classdir/name
                        mask_path.parent.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(piece_mask).save(mask_path,format="PNG",compress_level=1)
                    try:
                        with Image.open(image_path) as tile:
                            tile.load()
                            if tile.size!=(1056,1056):raise ValueError("Incorrect tile dimensions")
                            tiles.append(np.array(tile))
                        if mask_path:
                            with Image.open(mask_path) as mask:
                                mask.load();masks.append(np.array(mask))
                    except Exception:
                        r["file_corruption_count"]+=1;raise
                    rows.append(dict(split=split,source_image_path=str(path),source_image_id=f"{split}/{'bad' if label else 'good'}/{path.stem}",
                                     source_label=label,tile_id=tile_id,x0=x0,x1=x1,y0=y0,y1=y1,tile_image_path=str(image_path),
                                     source_mask_path=str(source_mask) if source_mask else "",tile_mask_path=str(mask_path) if mask_path else "",
                                     tile_positive_pixels=positive,tile_is_anomalous=abnormal))
                    if split=="test_public":test_abnormal+=abnormal
                if not np.array_equal(np.concatenate(tiles,axis=1),original):raise ValueError(f"Image reconstruction failed: {path}")
                r["image_reconstruction_pass_count"]+=1
                if label:
                    if not np.array_equal(np.concatenate(masks,axis=1),original_mask):raise ValueError(f"Mask reconstruction failed: {source_mask}")
                    r["mask_reconstruction_pass_count"]+=1
            if (index+1)%10==0:print(f"Lossless reconstruction verified: {index+1}/270 sources",flush=True)
        r["tile_counts"]={split:sum(row["split"]==split for row in rows) for split in r["source_counts"]}
        assert r["tile_counts"]==dict(train=548,validation=76,test_public=456)
        assert r["image_reconstruction_pass_count"]==270 and r["mask_reconstruction_pass_count"]==90
        r.update(test_abnormal_tiles=test_abnormal,test_normal_tiles=456-test_abnormal,tile_size_distribution={"1056x1056":1080})
        with (DERIVED/"manifest.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
        r.update(manifest_sha256=manifest_hash(),status="PASS")
    except (Exception,KeyboardInterrupt) as exc:
        r.update(error_message=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        r["elapsed_seconds"]=time.perf_counter()-start;save_report(r)
        print(json.dumps(r,indent=2),flush=True)
    return r
