import os
import cv2 as cv

from sudoku_processor import SudokuProcessor

def print_board(board, title="Board"):
    print("\n" + title)
    for r in range(9):
        row = ""
        for c in range(9):
            v = board[r][c]
            row += (". " if v == 0 else f"{v} ")
            if c % 3 == 2 and c != 8:
                row += "| "
        print(row.strip())
        if r % 3 == 2 and r != 8:
            print("-" * 21)


def SolveSudoku(IMAGE_PATH: str, ARTIFACT_DIR: str, save_debug_images: bool = False) -> bool:
    """
    Processes an image of a Sudoku board, recognizes the digits, and solves the puzzle.
    
    Args:
        IMAGE_PATH (str): Full or relative path to the Sudoku image.
        ARTIFACT_DIR (str): Directory where the ONNX model is located and output images are saved.
        save_debug_images (bool): Whether to save intermediate processing images.
        
    Returns:
        bool: True if solved successfully, False otherwise.
    """
    # 1. Verify and load image
    bgr = cv.imread(IMAGE_PATH)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    # 2. Load ONNX model path from the artifact directory
    onnx_path = os.path.join(ARTIFACT_DIR, "sudoku_digitnet.onnx")

    # 3. Initialize processor and process the image
    proc = SudokuProcessor(
        onnx_path=onnx_path, 
        warp_size=450, 
        blank_conf_threshold=0.65, 
        save_debug_images=save_debug_images
    )

    board, solved, dbg, overlay = proc.process_image(bgr, overlay=True)

    print_board(board, "Recognized board")
    
    # 4. Save debug outputs if requested
    if save_debug_images:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        cv.imwrite(os.path.join(ARTIFACT_DIR, "warped.png"), dbg["warped"])
        if overlay is not None:
            cv.imwrite(os.path.join(ARTIFACT_DIR, "overlay_solution.png"), overlay)
            print(f"\nSaved: {os.path.join(ARTIFACT_DIR, 'overlay_solution.png')}")
        print(f"Saved: {os.path.join(ARTIFACT_DIR, 'warped.png')}")

    # 5. Return solving status
    if solved is None:
        print("\nSolver failed (board might contain recognition errors).")
        return False
    else:
        print_board(solved, "Solved board")
        return True


if __name__ == "__main__":
    # You can now pass any absolute or relative path directly
    IMAGE_PATH = os.path.join("data_v1", "grids", "val", "000002_cam.png")
    ARTIFACT_DIR = "artifacts_v4"
    
    SolveSudoku(
        IMAGE_PATH=IMAGE_PATH,
        ARTIFACT_DIR=ARTIFACT_DIR,
        save_debug_images=False
    )