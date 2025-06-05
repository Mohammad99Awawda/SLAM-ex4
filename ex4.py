import os
import cv2
import pickle
import random
import time

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from pathlib import Path

 

ROOT       = Path(r"C:\Users\muham\PycharmProjects\SLAM ex4")
SEQ_PATH   = ROOT / "dataset" / "sequences" / "05"
POSES_FILE = ROOT / "dataset" / "poses" / "05.txt"
CALIB_FILE = r"C:\Users\muham\PycharmProjects\SLAM ex4\dataset\sequences\05\calib.txt"      # one level above sequences/05
OUT_DIR    = ROOT / "out_ex4"
OUT_DIR.mkdir(exist_ok=True)

# Ensure output directory exists
OUT_DIR.mkdir(exist_ok=True, parents=True)

 
#  4.1  TRACK DATABASE  
 
class TrackDB:
    """
    track_id → list of (frame, kpL_index, kpR_index)
    frame    → list of track_ids appearing in that frame
    """
    def __init__(self):
        self.tracks = defaultdict(list)
        self.frames = defaultdict(list)
        self.next_id = 0

    def new_track(self, frame, kpL_idx, kpR_idx):
        """
        Create a brand‐new track with a single observation in 'frame'.
        Returns the new track_id.
        """
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid].append((frame, kpL_idx, kpR_idx))
        self.frames[frame].append(tid)
        return tid

    def extend_track(self, tid, frame, kpL_idx, kpR_idx):
        """
        Add a new observation (frame, kpL_idx, kpR_idx) to existing track 'tid'.
        """
        self.tracks[tid].append((frame, kpL_idx, kpR_idx))
        self.frames[frame].append(tid)

    def track_ids_on_frame(self, frame):
        """Return a list of all track_ids observed in this frame."""
        return self.frames.get(frame, [])

    def frames_of_track(self, tid):
        """Return a sorted list of frame‐IDs in which this track appears."""
        return [f for (f,_,_) in self.tracks[tid]]

    def pixel_triplet(self, frame, tid, kplistL, kplistR):
        """
        Given a track tid and a frame number, return (u_L, u_R, v)
        for that (frame, kpL_idx, kpR_idx) tuple.
        """
        for (f, kpL, kpR) in self.tracks[tid]:
            if f == frame:
                uL, vL = kplistL[kpL].pt
                uR, _  = kplistR[kpR].pt
                return (uL, uR, vL)
        raise RuntimeError(f"Track {tid} not found in frame {frame}")

    def save(self, path):
        """Serialize the database into a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'tracks': dict(self.tracks),
                'frames': dict(self.frames),
                'next_id': self.next_id
            }, f)

    @classmethod
    def load(cls, path):
        """Load a TrackDB from a pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        db = cls()
        db.tracks  = defaultdict(list, data['tracks'])
        db.frames  = defaultdict(list, data['frames'])
        db.next_id = data['next_id']
        return db

 
#   Read KITTI calibration 
 
def read_kitti_calib(p: Path):
    """
    Reads calib.txt in KITTI format:
      line 0: "P0: fx 0 cx  ...  0 0 1 0"
      line 1: "P1: fx 0 cx-B  ...  0 0 1 0"
    Returns (K, P0, P1) as NumPy arrays:
      - K: 3×3 intrinsics
      - P0, P1: 3×4 projection matrices for left, right
    """
    with open(p, 'r') as f:
        lines = f.readlines()
    P0 = np.fromstring(lines[0].split(maxsplit=1)[1], sep=' ').reshape(3, 4)
    P1 = np.fromstring(lines[1].split(maxsplit=1)[1], sep=' ').reshape(3, 4)
    K  = P0[:, :3]   # Both cameras share the same K in KITTI stereo
    return K, P0, P1

 
#  Helper: Load a rectified stereo image 
 
def load_gray(frame: int, cam: int):
    """
    Load a rectified stereo image
    cam = 0 → left image_0,  cam = 1 → right image_1.
    """
    path = SEQ_PATH / f"image_{cam}" / f"{frame:06d}.png"
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

 
#  Helper: Linear least-squares triangulation of one correspondence
 
def triangulate_llsq(Pa, Pb, pt_a, pt_b):
    """
    Linear least-squares triangulation of a single correspondence:
      Pa, Pb: 3×4 projection matrices (for left, right).
      pt_a = (u_L, v_L),  pt_b = (u_R, v_R).
    Returns a 3‐vector [X, Y, Z] in the left-camera frame.
    """
    ua, va = pt_a
    ub, vb = pt_b
    A = np.vstack([
        ua*Pa[2] - Pa[0],
        va*Pa[2] - Pa[1],
        ub*Pb[2] - Pb[0],
        vb*Pb[2] - Pb[1]
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3])

 
#  Helper: Compute stereo inliers and return 3D cloud for one frame
 
akaze = cv2.AKAZE_create()
bf    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def stereo_inliers(frame):
    """
    For a given frame index:
      • load grayscale left & right images
      • detect AKAZE keypoints + descriptors
      • match with KNN + Lowe ratio (.75)
      • filter inliers by Δv ≤ 2 px
      • triangulate each inlier → (X,Y,Z)
    Returns:
      kpL, kpR, inlier_matches, cloud3D
    """
    L = load_gray(frame, 0)
    R = load_gray(frame, 1)
    kpL, desL = akaze.detectAndCompute(L, None)
    kpR, desR = akaze.detectAndCompute(R, None)

    knn  = bf.knnMatch(desL, desR, k=2)
    good = [m for (m,n) in knn if m.distance < 0.75 * n.distance]
    inl  = [m for m in good if abs(kpL[m.queryIdx].pt[1] - kpR[m.trainIdx].pt[1]) <= 2]

    cloud = np.array([
        triangulate_llsq(P0, P1,
                         kpL[m.queryIdx].pt,
                         kpR[m.trainIdx].pt)
        for m in inl
    ])

    return kpL, kpR, inl, cloud

 
#  Helper: Match AKAZE descriptors between two left‐images 
 
def left_to_left_matches(kpLa, desLa, kpLb, desLb):
    """
    Match descriptors from left_frame_A to left_frame_B using BF‐KNN + Lowe ratio.
    Returns a list of good matches (no geometric filtering here).
    """
    knn = bf.knnMatch(desLa, desLb, k=2)
    return [m for (m,n) in knn if m.distance < 0.75 * n.distance]

 
#  Read intrinsics + stereo extrinsics once (for both 4.1 and 4.7)
 
K, P0, P1 = read_kitti_calib(CALIB_FILE)

 
#  4.1  Build the TrackDB along the whole sequence
 
def q4_1():
    """
    4.1 – build a TrackDB over all frames:
      • Initialize first‐frame tracks from stereo inliers.
      • For each subsequent frame: match left_{f−1} to left_f,
        extend existing tracks when possible, otherwise create new tracks.
      • Serialize to OUT_DIR/tracks.pkl.
    """
    global db, kpL0, kpR0, inl0, cloud0, prev_kpL, prev_desL, frames

    frames = sorted(int(f.stem) for f in (SEQ_PATH/"image_0").glob("*.png"))
    db     = TrackDB()
    start  = time.time()

    kpL0, kpR0, inl0, cloud0 = stereo_inliers(frames[0])
    for m in inl0:
        db.new_track(frames[0], m.queryIdx, m.trainIdx)

    prev_kpL, prev_desL = kpL0, akaze.compute(load_gray(frames[0],0), kpL0)[1]

    for f in frames[1:]:
        kpL, kpR, inl, cloud = stereo_inliers(f)
        desL                  = akaze.compute(load_gray(f,0), kpL)[1]

        LL = left_to_left_matches(prev_kpL, prev_desL, kpL, desL)

        prev_map = {}
        for tid in db.track_ids_on_frame(f-1):
            prev_kpL_idx = db.tracks[tid][-1][1]  # last (frame, kpL, kpR)—take kpL
            prev_map[prev_kpL_idx] = tid

        matched_to_prev = set()
        for m in LL:
            if m.queryIdx in prev_map:
                tid = prev_map[m.queryIdx]
                db.extend_track(tid, f, m.trainIdx, None)
                matched_to_prev.add(m.trainIdx)

        for m in inl:
            if m.queryIdx not in matched_to_prev:
                db.new_track(f, m.queryIdx, m.trainIdx)

        prev_kpL, prev_desL = kpL, desL

    elapsed = time.time() - start
    print(f"4.1 Tracking time: {elapsed:.1f} s")

    # Save to pickle
    db.save(OUT_DIR / "tracks.pkl")
    print(f"4.1 TrackDB saved to: {OUT_DIR / 'tracks.pkl'}")

 
#  4.2  
 
def q4_2():
    """
    4.2 – compute and print:
      • Total number of tracks (length  >= 2)
      • Total frames
      • Mean / Max / Min track length
      • Mean connections per frame
    """
    lens  = [len(trk) for trk in db.tracks.values() if len(trk) > 1]
    total_tracks = len(lens)
    total_frames = len(frames)
    mean_len     = np.mean(lens) if lens else 0
    max_len      = max(lens) if lens else 0
    min_len      = min(lens) if lens else 0

    # connectivity: how many tracks in frame f continue to frame f+1
    links = []
    for f in frames[:-1]:
        s1 = set(db.track_ids_on_frame(f))
        s2 = set(db.track_ids_on_frame(f+1))
        links.append(len(s1 & s2))
    mean_conn = np.mean(links) if links else 0

    print("\n4.2 === Tracking statistics (length  >= 2) ===")
    print(f"Total tracks   : {total_tracks}")
    print(f"Total frames   : {total_frames}")
    print(f"Mean length    : {mean_len:.2f}")
    print(f"Max length     : {max_len}")
    print(f"Min length     : {min_len}")
    print(f"Mean connections/frame: {mean_conn:.2f}")

 
#  4.3  Show a track of length  >= 6 
 
def q4_3():
    """
    4.3 – pick a random track of length  >= 6, create 20×20 patches around each keypoint,
         draw a small circle at the center, save to OUT_DIR/track_<tid>.
    """
    lens  = [len(trk) for trk in db.tracks.values()]
    long_tracks = [tid for (tid,trk) in db.tracks.items() if len(trk) >= 6]

    if not long_tracks:
        print("4.3 No track of length  >= 6 found for patch viewer.")
        return

    tid       = random.choice(long_tracks)
    patch_dir = OUT_DIR / f"track_{tid:05d}"
    patch_dir.mkdir(exist_ok=True, parents=True)

    for (frame, kpL_idx, _) in db.tracks[tid]:
        img = load_gray(frame, 0)
        # Determine correct keypoint (u,v)
        if frame == frames[0]:
            u, v = map(int, kpL0[kpL_idx].pt)
        else:
            kps_frame, _ = akaze.detectAndCompute(img, None)
            u, v         = map(int, kps_frame[kpL_idx].pt)
        x0, y0 = max(0, u - 10), max(0, v - 10)
        patch = img[y0:y0+20, x0:x0+20].copy()
        # Draw a small circle at center
        cv2.circle(patch, (min(10, u-x0), min(10, v-y0)), 2, 255, -1)
        cv2.imwrite(str(patch_dir / f"{frame:06d}.png"), patch)

    print(f"4.3 Saved 20×20 patches for track {tid} → {patch_dir}")

 
#  4.4  Connectivity graph
 
def q4_4():
    """
    4.4 – plot “Connectivity: # tracks that continue to next frame”
         and save as connectivity.png.
    """
    conn = []
    for f in frames[:-1]:
        s1 = set(db.track_ids_on_frame(f))
        s2 = set(db.track_ids_on_frame(f+1))
        conn.append(len(s1 & s2))

    plt.figure(figsize=(6,4))
    plt.plot(frames[:-1], conn, linestyle='-')
    plt.title("4.4 Connectivity: # tracks that continue to next frame")
    plt.xlabel("Frame index")
    plt.ylabel("# continuing tracks")
    plt.grid(True)
    plt.tight_layout()
    out_path = OUT_DIR / "connectivity.png"
    plt.savefig(out_path)
    plt.close()
    print(f"4.4 Connectivity plot saved to: {out_path}")

 
#  4.5   Inliers percentage per frame
 
def q4_5():
    """
    4.5 – plot “% Inliers per Frame (approx = continuing / total tracks)”
         and save as pct_inliers.png.
    """
    pct = []
    for i, f in enumerate(frames[:-1]):
        total_f   = len(db.track_ids_on_frame(f))
        cont_f    = len(set(db.track_ids_on_frame(f)) & set(db.track_ids_on_frame(f+1)))
        pct.append((cont_f / max(1, total_f)) * 100)

    plt.figure(figsize=(6,4))
    plt.plot(frames[:-1], pct, linestyle='-')
    plt.title("4.5 % Inliers per Frame (approx)")
    plt.xlabel("Frame index")
    plt.ylabel("% continuing tracks")
    plt.grid(True)
    plt.tight_layout()
    out_path = OUT_DIR / "pct_inliers.png"
    plt.savefig(out_path)
    plt.close()
    print(f"4.5 % Inliers plot saved to: {out_path}")

 
#  4.6  Track-length histogram
 
def q4_6():
    """
    4.6 – plot “Histogram of Track Lengths (length  >= 2)”
         and save as hist_lengths.png.
    """
    lens = [len(trk) for trk in db.tracks.values() if len(trk) > 1]
    plt.figure(figsize=(5,4))
    plt.hist(lens, bins=30, edgecolor='black')
    plt.title("4.6 Histogram of Track Lengths (length  >= 2)")
    plt.xlabel("Track length in frames")
    plt.ylabel("# of tracks")
    plt.tight_layout()
    out_path = OUT_DIR / "hist_lengths.png"
    plt.savefig(out_path)
    plt.close()
    print(f"4.6 Track-length histogram saved to: {out_path}")

 
#  4.7  Reprojection-error for one random track 
 
def q4_7():
    """
    4.7 – pick a random track of length >= 10, triangulate its last‐frame stereo
          point using GT left‐camera, then compute reprojection error on each frame,
          plot vs. frame index, save as reproj_track_<tid>.png.
    """

    gt_all = np.loadtxt(POSES_FILE).reshape(-1, 3, 4)

    long10 = [tid for (tid, trk) in db.tracks.items() if len(trk) >= 10]
    if not long10:
        print("4.7 No track of length ≥ 10 found for reprojection‐error test.")
        return

    tid = random.choice(long10)
    track_frames = [f for (f, _, _) in db.tracks[tid]]
    lastF = track_frames[-1]

    kpL_last, kpR_last, inl_last, _ = stereo_inliers(lastF)

    Xw = None
    for (f, kpL_idx, kpR_idx) in db.tracks[tid]:
        if f == lastF:
            uL, vL = kpL_last[kpL_idx].pt

            correct_kpR_idx = None
            for m in inl_last:
                if m.queryIdx == kpL_idx:
                    correct_kpR_idx = m.trainIdx
                    break

            if correct_kpR_idx is None:
                print(f"Warning: no right‐keypoint for track {tid} at frame {lastF}")
                Xw = None
                break

            uR, vR = kpR_last[correct_kpR_idx].pt

            P_left = gt_all[lastF]
            Xw = triangulate_llsq(
                np.linalg.inv(K) @ P_left,
                np.linalg.inv(K) @ P_left,
                (uL, vL), (uR, vR)
            )
            break

    if Xw is None:
        return

    reproj_errs = []
    for (f, kpL_idx, kpR_idx) in db.tracks[tid]:
        kpL_f, kpR_f, _, _ = stereo_inliers(f)[:4]
        P = gt_all[f]
        xy_proj = P @ np.append(Xw, 1.0)
        x_pixel = xy_proj[:2] / xy_proj[2]
        uL_true, vL_true = kpL_f[kpL_idx].pt
        err = np.linalg.norm(x_pixel - np.array([uL_true, vL_true]))
        reproj_errs.append(err)

    # Plot error vs frame index
    plt.figure(figsize=(5, 4))
    plt.plot(track_frames, reproj_errs)
    plt.title(f"4.7 Reproj Error for Track {tid} (length={len(track_frames)})")
    plt.xlabel("Frame index")
    plt.ylabel("Pixel reprojection error")
    plt.grid(True)
    plt.tight_layout()
    out_path = OUT_DIR / f"reproj_track_{tid}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"4.7 Reprojection‐error plot saved to: {out_path}")

 
#  MAIN
 
def main():
    print("Running Exercise 4 tasks...\n")
    q4_1()
    q4_2()
    q4_3()
    q4_4()
    q4_5()
    q4_6()
    q4_7()
    print("\nAll Exercise 4 tasks complete.")

if __name__ == "__main__":
    main()
