"""
Model-Assisted OBB Annotator
=============================
Loads each image, runs the pre-trained OBB model to propose bounding
boxes, and lets you accept / reject each one individually.  You can
also draw manual 4-corner polygons for cards the model missed.

Usage:
    python obb_annotator.py --images snapshots --labels labels
    python obb_annotator.py --images snapshots --labels labels --conf 0.3

Controls (shown in the side panel):
    LEFT-CLICK   — on a prediction → toggle accept / reject
                   on empty area   → place a manual corner point
    RIGHT-CLICK  — remove the last placed manual point (undo)

    y            — accept ALL predictions
    u            — reject ALL predictions
    n            — save current manual polygon (4 pts) & stay on image
    s            — save everything & go to next image
    d / →        — next image (without saving manual WIP)
    a / ←        — previous image
    c            — clear manual points in progress
    r            — reset image: delete label file & clear everything
    x            — delete image file + label, move to next
    q            — quit (moves labeled images to annotated_images/)
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Colours (BGR) ────────────────────────────────────────────────────
COL_PENDING  = (0, 200, 255)   # orange — model prediction, not yet decided
COL_ACCEPTED = (0, 255, 0)     # green  — accepted prediction
COL_REJECTED = (0, 0, 180)     # dark red — rejected (drawn faintly)
COL_MANUAL   = (255, 0, 0)     # blue   — manually drawn & saved polygons
COL_DRAWING  = (0, 255, 255)   # yellow — corners being placed right now
COL_POINT    = (0, 0, 255)     # red    — individual corner dot
COL_TEXT     = (255, 255, 255)  # white
COL_TEXT_DIM = (160, 160, 160) # grey
COL_BG       = (40, 40, 40)    # panel background

PANEL_W = 420

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "..", "weights", "detector.pt")
DEFAULT_RAW_DIR = os.path.join(SCRIPT_DIR, "..", "raw")
DEFAULT_ANNOTATED_DIR = os.path.join(SCRIPT_DIR, "..", "annotated")
DEFAULT_LABEL_DIR = os.path.join(SCRIPT_DIR, "..", "annotated", "labels")


# ── Geometry helpers ─────────────────────────────────────────────────
def point_in_polygon(px: float, py: float, poly: list[tuple[int, int]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def polygon_center(poly: list[tuple[int, int]]) -> tuple[int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


# ── Prediction wrapper ──────────────────────────────────────────────
class Prediction:
    """One model-proposed OBB with accept/reject state."""

    __slots__ = ("poly", "conf", "accepted")

    def __init__(self, poly: list[tuple[int, int]], conf: float):
        self.poly = poly
        self.conf = conf
        self.accepted: bool | None = None  # None = pending


# ── Annotator ────────────────────────────────────────────────────────
class OBBAnnotator:
    def __init__(self, img_dir: str, lbl_dir: str, annotated_dir: str,
                 model_path: str, conf: float):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir = Path(annotated_dir)
        (self.annotated_dir / "images").mkdir(parents=True, exist_ok=True)
        self.conf = conf

        self.image_paths = sorted(
            list(self.img_dir.rglob("*.jpg")) + list(self.img_dir.rglob("*.png"))
        )
        random.shuffle(self.image_paths)
        if not self.image_paths:
            print(f"[ERROR] No images found in {self.img_dir}")
            exit(1)

        print(f"[INFO] {len(self.image_paths)} images found in {self.img_dir}")
        print(f"[INFO] Loading OBB model: {model_path}")
        self.model = YOLO(model_path)

        self.current_idx = 0
        self.predictions: list[Prediction] = []
        self.manual_polygons: list[list[tuple[int, int]]] = []
        self.current_points: list[tuple[int, int]] = []
        self.img: np.ndarray | None = None
        self.display_img: np.ndarray | None = None
        self.window_name = "OBB Annotator (model-assisted)"

        self._panel_pred_hitboxes: list[tuple[int, int, int]] = []

    # ── Model inference ──────────────────────────────────────────────
    def _run_model(self) -> list[Prediction]:
        results = self.model.predict(
            self.img, imgsz=640, conf=self.conf, verbose=False
        )
        preds: list[Prediction] = []
        if not results or results[0].obb is None:
            return preds

        obb = results[0].obb
        corners_all = obb.xyxyxyxy.cpu().numpy()
        confs = obb.conf.cpu().numpy() if obb.conf is not None else [0.0] * len(corners_all)

        for corners, c in zip(corners_all, confs):
            poly = [(int(x), int(y)) for x, y in corners.reshape(4, 2)]
            preds.append(Prediction(poly, float(c)))
        return preds

    # ── Label I/O ────────────────────────────────────────────────────
    def _load_existing_labels(self):
        """Load previously saved YOLO-OBB labels as manual polygons."""
        img_path = self.image_paths[self.current_idx]
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
        self.manual_polygons = []
        if not lbl_path.exists():
            return
        h, w = self.img.shape[:2]
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 9:
                    coords = [float(p) for p in parts[1:]]
                    poly = []
                    for i in range(0, 8, 2):
                        poly.append((int(coords[i] * w), int(coords[i + 1] * h)))
                    self.manual_polygons.append(poly)

    def _save_labels(self):
        """Write accepted predictions + manual polygons to label file."""
        img_path = self.image_paths[self.current_idx]
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
        h, w = self.img.shape[:2]

        all_polys = [p.poly for p in self.predictions if p.accepted is True]
        all_polys += self.manual_polygons

        if not all_polys:
            if lbl_path.exists():
                lbl_path.unlink()
            return

        lines = []
        for poly in all_polys:
            coords = []
            for px, py in poly:
                coords.extend([f"{px / w:.6f}", f"{py / h:.6f}"])
            lines.append(f"0 {' '.join(coords)}\n")

        with open(lbl_path, "w") as f:
            f.writelines(lines)

    # ── Drawing ──────────────────────────────────────────────────────
    def _estimate_panel_height(self) -> int:
        """Calculate the minimum height the side panel needs."""
        n_pred_lines = len(self.predictions)
        # header(2) + gap + "Predictions" + summary + pred_lines +
        # gap + "Manual" + summary + gap + "Controls" + 8 controls
        n_lines = 2 + 1 + 2 + n_pred_lines + 1 + 2 + 1 + 1 + 8
        return n_lines * 20 + 40  # 20px per line + top/bottom padding

    def _draw(self):
        h, w = self.img.shape[:2]
        min_h = max(h, self._estimate_panel_height())

        # If the image is shorter than the panel needs, pad the image area.
        if min_h > h:
            pad = np.full((min_h - h, w, 3), 0, dtype=np.uint8)
            img_canvas = np.vstack((self.img.copy(), pad))
        else:
            img_canvas = self.img.copy()

        panel = np.full((min_h, PANEL_W, 3), COL_BG, dtype=np.uint8)
        canvas = np.hstack((img_canvas, panel))

        # --- Draw model predictions ---
        for i, pred in enumerate(self.predictions):
            if pred.accepted is True:
                color, thickness = COL_ACCEPTED, 2
            elif pred.accepted is False:
                color, thickness = COL_REJECTED, 1
            else:
                color, thickness = COL_PENDING, 2

            pts = np.array(pred.poly, np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], True, color, thickness)

            cx, cy = polygon_center(pred.poly)
            tag = f"#{i + 1} {pred.conf:.0%}"
            if pred.accepted is True:
                tag += " OK"
            elif pred.accepted is False:
                tag += " X"

            cv2.putText(canvas, tag, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # --- Draw saved manual polygons ---
        for poly in self.manual_polygons:
            pts = np.array(poly, np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], True, COL_MANUAL, 2)
            for pt in poly:
                cv2.circle(canvas, pt, 4, COL_MANUAL, -1)

        # --- Draw current manual points being placed ---
        for i, pt in enumerate(self.current_points):
            cv2.circle(canvas, pt, 6, COL_POINT, -1)
            if i > 0:
                cv2.line(canvas, self.current_points[i - 1], pt, COL_DRAWING, 2)
            if len(self.current_points) == 4 and i == 3:
                cv2.line(canvas, self.current_points[3], self.current_points[0], COL_DRAWING, 2)

        # --- Side panel text ---
        img_path = self.image_paths[self.current_idx]
        x0 = w + 12
        font = cv2.FONT_HERSHEY_SIMPLEX
        ln = 18   # line height in pixels
        sp = 8    # gap height for empty separator lines
        fs = 0.43 # font scale

        y = 22

        def put(text, col, bold=False):
            nonlocal y
            t = 2 if bold else 1
            cv2.putText(canvas, text, (x0, y), font, fs, col, t, cv2.LINE_AA)
            y += ln

        def gap():
            nonlocal y
            y += sp

        put(f"Image {self.current_idx + 1} / {len(self.image_paths)}", COL_TEXT, bold=True)
        put(img_path.name, COL_TEXT_DIM)
        gap()

        put("-- Predictions --", COL_TEXT, bold=True)
        n_accepted = sum(1 for p in self.predictions if p.accepted is True)
        n_rejected = sum(1 for p in self.predictions if p.accepted is False)
        n_pending  = sum(1 for p in self.predictions if p.accepted is None)
        put(f" {len(self.predictions)} total  "
            f"{n_accepted} ok  {n_rejected} rej  {n_pending} pending", COL_TEXT_DIM)

        self._panel_pred_hitboxes = []
        for i, pred in enumerate(self.predictions):
            status = "PENDING" if pred.accepted is None else ("OK" if pred.accepted else "REJECTED")
            col = COL_PENDING if pred.accepted is None else (COL_ACCEPTED if pred.accepted else COL_REJECTED)
            y_top = y - ln + 4
            put(f" #{i + 1}  {pred.conf:.0%}  {status}", col)
            self._panel_pred_hitboxes.append((y_top, y - 4, i))

        gap()
        put("-- Manual boxes --", COL_TEXT, bold=True)
        put(f" {len(self.manual_polygons)} saved, "
            f"{len(self.current_points)}/4 placing", COL_TEXT_DIM)
        gap()

        put("-- Controls --", COL_TEXT, bold=True)
        controls = [
            "click pred -> toggle ok/rej",
            "click empty -> place corner",
            "right-click -> undo last pt",
            "[y] accept all  [u] reject all",
            "[n] save manual box & stay",
            "[s] save all & next image",
            "[d] next  [a] prev  (no save)",
            "[c] clear pts  [r] reset img",
            "[x] delete image  [q] quit",
        ]
        for line in controls:
            put(line, COL_TEXT_DIM)

        self.display_img = canvas
        cv2.imshow(self.window_name, canvas)

    # ── Mouse handling ───────────────────────────────────────────────
    def _mouse_cb(self, event, x, y, flags, param):
        if self.img is None:
            return
        img_w = self.img.shape[1]

        if event == cv2.EVENT_LBUTTONDOWN:
            # --- Click on the side panel: check prediction label hitboxes ---
            if x >= img_w:
                for y_top, y_bot, idx in self._panel_pred_hitboxes:
                    if y_top <= y <= y_bot:
                        pred = self.predictions[idx]
                        if pred.accepted is None or pred.accepted is False:
                            pred.accepted = True
                        else:
                            pred.accepted = False
                        self._draw()
                        return
                return

            # --- Click on image: always place a manual corner ---
            if len(self.current_points) < 4:
                self.current_points.append((x, y))
                self._draw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_points:
                self.current_points.pop()
                self._draw()

    # ── Image loading ────────────────────────────────────────────────
    def _load_image(self):
        img_path = self.image_paths[self.current_idx]
        self.img = cv2.imread(str(img_path))
        self.current_points = []
        self._load_existing_labels()
        self.predictions = self._run_model()

        # Auto-accept predictions that closely match existing labels
        # (so re-opening an already-labeled image keeps them green).
        if self.manual_polygons and self.predictions:
            self._match_existing_labels()

    def _match_existing_labels(self):
        """If a prediction overlaps an existing label, auto-accept it and
        remove the redundant manual polygon so we don't double-save."""
        remaining_manual = []
        for mpoly in self.manual_polygons:
            matched = False
            mc = polygon_center(mpoly)
            for pred in self.predictions:
                if pred.accepted is not None:
                    continue
                if point_in_polygon(mc[0], mc[1], pred.poly):
                    pred.accepted = True
                    matched = True
                    break
            if not matched:
                remaining_manual.append(mpoly)
        self.manual_polygons = remaining_manual

    # ── Save manual polygon helper ───────────────────────────────────
    def _commit_manual_polygon(self) -> bool:
        if len(self.current_points) != 4:
            print("  Need exactly 4 points to save a manual box.")
            return False
        self.manual_polygons.append(list(self.current_points))
        self.current_points = []
        return True

    # ── Main loop ────────────────────────────────────────────────────
    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

        while 0 <= self.current_idx < len(self.image_paths):
            self._load_image()
            self._draw()

            while True:
                key = cv2.waitKey(20) & 0xFF

                if key == ord("s"):
                    if len(self.current_points) == 4:
                        self._commit_manual_polygon()
                    self._save_labels()
                    n = sum(1 for p in self.predictions if p.accepted)
                    img_path = self.image_paths[self.current_idx]
                    print(f"  Saved {img_path.name}: "
                          f"{n} accepted preds + {len(self.manual_polygons)} manual")
                    self._move_to_annotated(self.current_idx)
                    break

                elif key == ord("n"):
                    if self._commit_manual_polygon():
                        print("  Manual box added. Ready for another.")
                        self._draw()

                elif key == ord("y"):
                    for p in self.predictions:
                        p.accepted = True
                    self._draw()

                elif key == ord("u"):
                    for p in self.predictions:
                        p.accepted = False
                    self._draw()

                elif key == ord("d") or key == 83:  # 83 = right arrow on some systems
                    self.current_idx += 1
                    break

                elif key == ord("a") or key == 81:  # 81 = left arrow
                    self.current_idx = max(0, self.current_idx - 1)
                    break

                elif key == ord("c"):
                    self.current_points = []
                    self._draw()

                elif key == ord("r"):
                    img_path = self.image_paths[self.current_idx]
                    lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
                    if lbl_path.exists():
                        lbl_path.unlink()
                    self.manual_polygons = []
                    self.current_points = []
                    for p in self.predictions:
                        p.accepted = None
                    self._draw()
                    print(f"  Reset {img_path.name}")

                elif key == ord("x"):
                    img_path = self.image_paths[self.current_idx]
                    print(f"  Deleting {img_path.name}")
                    if img_path.exists():
                        img_path.unlink()
                    lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
                    if lbl_path.exists():
                        lbl_path.unlink()
                    self.image_paths.pop(self.current_idx)
                    break

                elif key == ord("q"):
                    cv2.destroyAllWindows()
                    self._cleanup()
                    return

        cv2.destroyAllWindows()
        self._cleanup()

    # ── Move annotated image out of raw ─────────────────────────────
    def _move_to_annotated(self, idx: int):
        """Move image + label to annotated/ and remove from raw/."""
        img_path = self.image_paths[idx]
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"

        dest_img = self.annotated_dir / "images" / img_path.name
        dest_lbl = self.annotated_dir / "labels" / f"{img_path.stem}.txt"

        if img_path.exists():
            shutil.move(str(img_path), str(dest_img))
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(dest_lbl))

        self.image_paths.pop(idx)
        print(f"  Moved to {self.annotated_dir.name}/  "
              f"({len(self.image_paths)} images remaining in raw)")

    # ── Cleanup on exit ──────────────────────────────────────────────
    def _cleanup(self):
        remaining = len([p for p in self.image_paths if p.exists()])
        print(f"\n  {remaining} unannotated image(s) remain in {self.img_dir}/")
        print("  Done.")


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model-assisted OBB annotator for playing cards."
    )
    parser.add_argument(
        "--images", default=DEFAULT_RAW_DIR,
        help="Directory with raw images to annotate (searched recursively).",
    )
    parser.add_argument(
        "--labels", default=DEFAULT_LABEL_DIR,
        help="Directory to write YOLO-OBB label files.",
    )
    parser.add_argument(
        "--annotated", default=DEFAULT_ANNOTATED_DIR,
        help="Directory to move annotated images + labels into.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Path to YOLO-OBB weights (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="Model confidence threshold for proposals (default: 0.3).",
    )
    args = parser.parse_args()

    annotator = OBBAnnotator(
        args.images, args.labels, args.annotated, args.model, args.conf
    )
    annotator.run()
