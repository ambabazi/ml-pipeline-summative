"""Turn a raw drawing scan into the numbers (features) the model uses."""
import cv2
import numpy as np

IMG_SIZE = 256  # every scan is resized to 256x256 so features are comparable

# The 9 engineered features, in the exact order the model was trained on
FEATURES = ["ink_ratio", "deviation", "fragments", "deviation_p90", "deviation_std",
            "thickness_mean", "thickness_std", "largest_piece", "template_coverage"]


def load_image(source):
    """Read an image from a file path OR from raw bytes (what the API receives)."""
    if isinstance(source, (bytes, bytearray)):                       # uploaded file -> bytes
        arr = np.frombuffer(source, dtype=np.uint8)                   # bytes -> 1-D array of numbers
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)                     # decode JPEG/PNG -> BGR image
    else:
        img = cv2.imread(str(source))                                 # path on disk -> BGR image
    if img is None:                                                   # OpenCV returns None for broken/non-image files
        raise ValueError("Could not read image - is it a valid JPG/PNG?")
    return img


def separate_layers(img):
    """Split a scan into the patient's blue pen stroke and the black printed template."""
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))                       # standard size
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                        # colour space where "blue" / "dark" are easy to describe
    pen = cv2.inRange(hsv, (100, 80, 50), (140, 255, 255))            # mask: blue, saturated pixels = patient's ink
    template = cv2.inRange(hsv, (0, 0, 0), (180, 255, 90))            # mask: dark pixels = printed template
    return img, pen, template


def extract_features(source):
    """Image (path or bytes) -> dict of the 9 engineered features."""
    _, pen, tpl = separate_layers(load_image(source))                 # load, then split into the two masks
    pen_px, tpl_px = int((pen > 0).sum()), int((tpl > 0).sum())       # ink pixels / template pixels

    dist_to_template = cv2.distanceTransform(255 - tpl, cv2.DIST_L2, 3)   # distance of every pixel to the template line
    dev = dist_to_template[pen > 0] if pen_px else np.zeros(1)            # ...measured only where the patient drew

    _, _, stats, _ = cv2.connectedComponentsWithStats(pen)            # separate pieces of ink
    areas = stats[1:, cv2.CC_STAT_AREA]                               # piece sizes (index 0 = background)

    thickness_map = cv2.distanceTransform(pen, cv2.DIST_L2, 3)        # inside the ink: distance to its edge ~ half line width
    thick = thickness_map[pen > 0] if pen_px else np.zeros(1)         # thickness values along the stroke

    near_pen = cv2.dilate(pen, np.ones((5, 5), np.uint8)) > 0         # ink widened by 2 px in every direction

    return {
        "ink_ratio": pen_px / max(tpl_px, 1),                         # line weight / pressure
        "deviation": float(dev.mean()),                               # average distance off the line
        "fragments": int((areas >= 5).sum()),                         # number of broken pieces
        "deviation_p90": float(np.percentile(dev, 90)),               # worst 10% of drift (big tremor swings)
        "deviation_std": float(dev.std()),                            # how erratic the drift is
        "thickness_mean": float(thick.mean()),                        # average line thickness
        "thickness_std": float(thick.std()),                          # pressure variation
        "largest_piece": float(areas.max() / max(pen_px, 1)) if len(areas) else 0.0,  # continuity of the stroke
        "template_coverage": float((near_pen & (tpl > 0)).sum() / max(tpl_px, 1)),    # share of the template traced
    }