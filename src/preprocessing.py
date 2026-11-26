"""Turn a raw drawing scan into the numbers (features) the model uses."""
import cv2
import numpy as np

IMG_SIZE = 256

# Keep the feature order expected by the saved model.
FEATURES = ["ink_ratio", "deviation", "fragments", "deviation_p90", "deviation_std",
            "thickness_mean", "thickness_std", "largest_piece", "template_coverage"]


def load_image(source):
    """Read an image from a file path OR from raw bytes (what the API receives)."""
    if isinstance(source, (bytes, bytearray)):
        if not source:
            raise ValueError("The uploaded image is empty.")
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(source))
    if img is None:
        raise ValueError("Could not read image - is it a valid JPG/PNG?")
    return img


def separate_layers(img):
    """Split a scan into the patient's blue pen stroke and the black printed template."""
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # HSV separates saturated blue ink from the dark guide.
    pen = cv2.inRange(hsv, (100, 80, 50), (140, 255, 255))
    template = cv2.inRange(hsv, (0, 0, 0), (180, 255, 90))
    return img, pen, template


def extract_features(source, *, check_input=False):
    """Extract drawing measurements, with an optional check for uploaded images."""
    _, pen, tpl = separate_layers(load_image(source))
    pen_px = int(np.count_nonzero(pen))
    tpl_px = int(np.count_nonzero(tpl))

    near_pen = cv2.dilate(pen, np.ones((5, 5), np.uint8)) > 0
    ink_ratio = pen_px / max(tpl_px, 1)
    template_coverage = np.count_nonzero(near_pen & (tpl > 0)) / max(tpl_px, 1)

    if check_input and (
        not pen_px
        or not tpl_px
        or not 0.5 <= ink_ratio <= 5
        or template_coverage < 0.08
    ):
        raise ValueError(
            "Could not assess this image. Upload a clear spiral or meander "
            "traced in blue ink over a dark printed guide."
        )

    dist_to_template = cv2.distanceTransform(255 - tpl, cv2.DIST_L2, 3)
    dev = dist_to_template[pen > 0] if pen_px else np.zeros(1)

    _, _, stats, _ = cv2.connectedComponentsWithStats(pen)
    # Connected-components row zero describes the background.
    areas = stats[1:, cv2.CC_STAT_AREA]

    thickness_map = cv2.distanceTransform(pen, cv2.DIST_L2, 3)
    thick = thickness_map[pen > 0] if pen_px else np.zeros(1)

    near_pen = cv2.dilate(pen, np.ones((5, 5), np.uint8)) > 0

    return {
        "ink_ratio": pen_px / max(tpl_px, 1),
        "deviation": float(dev.mean()),
        "fragments": int((areas >= 5).sum()),
        "deviation_p90": float(np.percentile(dev, 90)),
        "deviation_std": float(dev.std()),
        "thickness_mean": float(thick.mean()),
        "thickness_std": float(thick.std()),
        "largest_piece": float(areas.max() / max(pen_px, 1)) if len(areas) else 0.0,
        "template_coverage": float((near_pen & (tpl > 0)).sum() / max(tpl_px, 1)),
    }
