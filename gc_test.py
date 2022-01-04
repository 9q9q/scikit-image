"""Simple quick test for local SLIC specifying centroids.

aconda skimage-dev (local skimage dev environment)
python gc_test.py

python gc_test.py --dev
"""

# TODO SLIC is merging segments (result is fewer than number of centroids) together.
# try not enforcing connectivity? understand slic better.
# maybe try creating landmarks for background instead of using ellipse? or try gaussian method?

# TODO add flags to make it easier to compare normal slic with modified slic

import argparse
import ast
import cv2
import math
import numpy as np
import os
import pandas as pd
from PIL import Image
from skimage.segmentation import mark_boundaries, slic  # automatically import dev
from tqdm import tqdm

DF = "/home/galen/projects/aflw_vis/df_xs.csv"
LM_OVERLAY = "/home/galen/projects/aflw_vis/lm_xs"
CROPPED = "/home/galen/projects/aflw_vis/crop_xs"


def _np_to_pil(img):
    """Convert NumPy array to PIL Image."""
    # TODO: https://github.com/9q9q/aflw_vis/issues/7
    return Image.fromarray(np.uint8(img))


def _blend(bg, fg, alpha=0.6):
    """Expects images in PIL Image format.
    If alpha > 0.5, foreground will be more prominent.
    """
    blended = Image.blend(bg.convert(
        "RGBA"), fg.convert("RGBA"), alpha)
    return blended.convert("RGB")


def _get_face_mask(img_shape, x, y, ra, rb, theta):
    """Return np.array in the shape of img_shape with with 1s for the face ellipse and 0s otherwise."""
    base = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    mask = cv2.ellipse(base, (x, y), (ra, rb), theta, 0., 360., 1, -1)
    return mask


def _get_slic(img, x, y, ra, rb, theta, centroids=None, num_seg=21, sigma=10):
    """Returns SLIC as np.array."""
    shape = (img.size[1], img.size[0])
    mask = _get_face_mask(shape, x, y, ra, rb, theta)
    seg = slic(img, n_segments=num_seg, sigma=sigma,
               mask=mask, centroids=centroids)
    return seg


def _save_slics(df, imgs_dir, lm_overlay_dir, slic_dir, overlay, dev):
    """Saves SLICs, possibly overlaid over image."""
    os.makedirs(slic_dir, exist_ok=True)
    ellipse_xs = list(map(int, df["ellipse_x"].tolist()))
    ellipse_ys = list(map(int, df["ellipse_y"].tolist()))
    ras = list(map(int, df["ra"].tolist()))
    rbs = list(map(int, df["rb"].tolist()))
    thetas = list(map(float, df["theta"].tolist()))
    feats = df["feats"]
    ids = df.index

    for i, id in tqdm(enumerate(ids), total=len(ids)):
        img_full_path = os.path.join(
            imgs_dir, "image"+str(id)+".jpg")
        if (os.path.isfile(img_full_path)):
            x = ellipse_xs[i]
            y = ellipse_ys[i]
            ra = ras[i]
            rb = rbs[i]
            theta = thetas[i]
            img = Image.open(img_full_path)

            # get centroids for SLIC
            centroids = []
            for _, feat in ast.literal_eval(feats[i]).items():
                centroids.append([0., feat[0], feat[1]])
            centroids = np.array(centroids)

            if dev:
                seg = _get_slic(img, x, y, ra, rb, np.degrees(
                    theta), centroids=centroids, num_seg=len(centroids))
            else:
                seg = _get_slic(img, x, y, ra, rb, np.degrees(
                    theta), num_seg=len(centroids))

            if overlay:
                lm_overlay = Image.open(os.path.join(
                    lm_overlay_dir, str(id)+".png"))
                seg = mark_boundaries(np.full([
                    lm_overlay.height, lm_overlay.width], 255), seg, color=[
                    255, 255, 255], mode="inner")
                seg = _np_to_pil(seg)
                # seg = _np_to_pil(seg*255)
                seg = _blend(lm_overlay, seg)
            else:
                seg = _np_to_pil(seg*255)

            seg.save(os.path.join(slic_dir, str(id)+".png"))

            # make circles red so easier to see
            # img = Image.open(os.path.join(slic_dir, str(id)+".png"))
            # circle_size = math.ceil(img.width * .01)
            # img = np.array(img)
            # for _, feat in ast.literal_eval(feats[i]).items():
            #     cv2.circle(img, (feat[0], feat[1]), circle_size,
            #                (255, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)
            # img = _np_to_pil(img)
            # img.save(os.path.join(slic_dir, str(id)+".png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=bool, default=False,
                    help="Whether to run dev SLIC version.")
    args = vars(ap.parse_args())
    dev = args["dev"]

    print("reading from csv {}".format(DF))
    df = pd.read_csv(DF, dtype=str)
    df = df.set_index("id")

    print("dev: {}".format(dev))
    if dev:
        _save_slics(df, CROPPED, LM_OVERLAY,
                    "gc_test_out", overlay=True, dev=dev)
    else:
        _save_slics(df, CROPPED, LM_OVERLAY,
                    "normal_out", overlay=True, dev=dev)
