import os, random, pathlib
import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import tqdm

def str_to_grid(s: str):
    """81-char '012345678' string -> 9x9 list of ints."""
    s = s.strip()
    assert len(s) == 81, f"bad length: {len(s)}"
    return [[int(s[9*r+c]) for c in range(9)] for r in range(9)]

def load_puzzles(parquet_path, n_train, n_val):
    """Load puzzles/solutions; return two lists of 9x9 int grids for train/val."""
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    need_cols = [c for c in ["puzzle", "solution"] if c in df.columns]
    if not need_cols or "puzzle" not in need_cols:
        raise ValueError(f"Expected 'puzzle' (and optionally 'solution') columns; got: {list(df.columns)}")

    df_train = df.iloc[:n_train]
    df_val   = df.iloc[n_train:n_train+n_val]

    puzzles_train = [str_to_grid(p) for p in df_train["puzzle"].astype(str)]
    puzzles_val   = [str_to_grid(p) for p in df_val["puzzle"].astype(str)]

    solutions_train = [str_to_grid(s) for s in df_train["solution"].astype(str)] if "solution" in df_train else None
    solutions_val   = [str_to_grid(s) for s in df_val["solution"].astype(str)]   if "solution" in df_val   else None

    return (puzzles_train, solutions_train), (puzzles_val, solutions_val)


def prepare_data(
    parquet_path: str = "train_1.parquet",
    out_dir: str = "data",
    n_train: int = 1000,
    n_val: int = 100,
    img_h: int = 1200,
    img_w: int = 800,
    grid_margin: int = 60,
    cell_size: int = 28,
    font: int = cv.FONT_HERSHEY_SIMPLEX,
    seed: int = 0,
    use_global_aug: bool = False  # === NEW FLAG: True for OLD behavior, False for NEW local behavior ===
):
    # Set random seed for reproducibility
    random.seed(seed)
    out_dir_path = pathlib.Path(out_dir)
    
    # ---- Internal helper functions ----
    
    def draw_grid(img, size=9, margin=grid_margin):
        H, W = img.shape[:2]
        step = (min(H, W) - 2*margin) // size
        x0 = (W - size*step) // 2
        y0 = (H - size*step) // 2
        for i in range(size+1):
            thickness = 3 if i % 3 == 0 else 1
            cv.line(img, (x0 + i*step, y0), (x0 + i*step, y0 + size*step), (0, 0, 0), thickness)
            cv.line(img, (x0, y0 + i*step), (x0 + size*step, y0 + i*step), (0, 0, 0), thickness)
        return (x0, y0, step)

    def put_digit(img, r, c, d, x0, y0, step):
        if d == 0:
            return
        text = str(d)
        scale = step / 60.0
        thickness = 2
        (tw, th), base = cv.getTextSize(text, font, scale, thickness)
        cx = x0 + c*step + (step - tw)//2
        cy = y0 + r*step + (step + th)//2 - th//4
        cv.putText(img, text, (cx, cy), font, scale, (0, 0, 0), thickness, cv.LINE_AA)

    def render_page(puzzle9x9):
        page = np.full((img_h, img_w, 3), 255, np.uint8)
        x0, y0, step = draw_grid(page, 9, grid_margin)
        for r in range(9):
            for c in range(9):
                put_digit(page, r, c, puzzle9x9[r][c], x0, y0, step)
        return page, (x0, y0, step)

    def aug_camera_global(img):
        """OLD BEHAVIOR: Perspective jitter + Gaussian blur on the WHOLE page."""
        H, W = img.shape[:2]
        def jitter(pt, s=20):
            return (pt[0] + random.randint(-s, s), pt[1] + random.randint(-s, s))
        src = np.float32([(0, 0), (W, 0), (W, H), (0, H)])
        dst = np.float32([jitter((0,0)), jitter((W,0)), jitter((W,H)), jitter((0,H))])
        M = cv.getPerspectiveTransform(src, dst)
        warped = cv.warpPerspective(img, M, (W, H), flags=cv.INTER_LINEAR, borderValue=(220, 220, 220))
        if random.random() < 0.5:
            k = random.choice([1, 2])
            warped = cv.GaussianBlur(warped, (2*k+1, 2*k+1), 0)
        return warped

    def ensure_dirs(split):
        for cls in range(10):
            (out_dir_path / split / str(cls)).mkdir(parents=True, exist_ok=True)

    def save_crops(page, geom, puzzle9x9, split, base_id, is_local_cam=False):
        """
        is_local_cam: True if we apply local jitter/blur to the cropped ROI (NEW BEHAVIOR).
        """
        x0, y0, step = geom
        idx = 0
        for r in range(9):
            for c in range(9):
                x1, y1 = x0 + c*step, y0 + r*step
                
                # Crucial: .copy() to avoid modifying the original page
                roi = page[y1:y1+step, x1:x1+step].copy()
                
                if is_local_cam:
                    h_roi, w_roi = roi.shape[:2]
                    s = 20  # Local jitter offset (smaller to protect digit shape)
                    def jitter(pt):
                        return (pt[0] + random.randint(-s, s), pt[1] + random.randint(-s, s))
                    
                    src_pts = np.float32([(0, 0), (w_roi, 0), (w_roi, h_roi), (0, h_roi)])
                    dst_pts = np.float32([jitter((0,0)), jitter((w_roi,0)), 
                                          jitter((w_roi,h_roi)), jitter((0,h_roi))])
                    
                    M = cv.getPerspectiveTransform(src_pts, dst_pts)
                    # Fill edges with pure white
                    roi = cv.warpPerspective(roi, M, (w_roi, h_roi), flags=cv.INTER_LINEAR, borderValue=(255, 255, 255))
                    
                    if random.random() < 0.5:
                        k = random.choice([1, 2])
                        roi = cv.GaussianBlur(roi, (2*k+1, 2*k+1), 0)
            
                g = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                g = cv.resize(g, (cell_size, cell_size), interpolation=cv.INTER_AREA)
                label = puzzle9x9[r][c]
                out_path = out_dir_path / split / str(label) / f"{base_id}_{idx:02d}.png"
                cv.imwrite(str(out_path), g)
                idx += 1

    def process_split(puzzles, split_name, n_limit):
        ensure_dirs(split_name)
        grid_dir = out_dir_path / "grids" / split_name
        grid_dir.mkdir(parents=True, exist_ok=True)

        N = min(n_limit, len(puzzles))
        for i in tqdm(range(N), desc=f"render {split_name}"):
            puzzle = puzzles[i]

            # 1. Clean page
            page, geom = render_page(puzzle)
            save_crops(page, geom, puzzle, split_name, base_id=f"{i:06d}_clean", is_local_cam=False)
            cv.imwrite(str(grid_dir / f"{i:06d}_clean.png"), page)

            # 2. Camera-like dataset generation based on the flag
            if use_global_aug:
                # OLD BEHAVIOR: Warp the whole page first, then crop
                warped_page = aug_camera_global(page)
                save_crops(warped_page, geom, puzzle, split_name, base_id=f"{i:06d}_cam", is_local_cam=False)
                cv.imwrite(str(grid_dir / f"{i:06d}_cam.png"), warped_page)
            else:
                # NEW BEHAVIOR: Crop clean page first, then warp each individual cell inside save_crops
                save_crops(page, geom, puzzle, split_name, base_id=f"{i:06d}_cam", is_local_cam=True)
                warped_page = aug_camera_global(page)
                cv.imwrite(str(grid_dir / f"{i:06d}_cam.png"), warped_page)
                # Note: We do not save a warped page here because the whole page is not warped.

    # ---- Execute generation logic ----
    (p_train, _s_train), (p_val, _s_val) = load_puzzles(parquet_path, n_train, n_val)
    process_split(p_train, "train", n_train)
    process_split(p_val, "val", n_val)
    print("Done. Output under:", out_dir_path.resolve())


if __name__ == "__main__":
    # === EXPERIMENT 1: Generate Data with OLD behavior (Global Augmentation) ===
    # This produces bad data where digit 6 might become 9 due to grid lines.
    prepare_data(
        parquet_path="train_1.parquet",
        out_dir="data",
        n_train=2000,
        n_val=200,
        use_global_aug=True  
    )

    # === EXPERIMENT 2: Generate Data with NEW behavior (Local Augmentation) ===
    # This produces good data where digits are warped inside their cells.
    prepare_data(
        parquet_path="train_1.parquet",
        out_dir="data_new",
        n_train=2000,
        n_val=200,
        use_global_aug=False 
    )
