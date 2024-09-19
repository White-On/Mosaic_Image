from rich.console import Console
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a mosaic image from a target image and a set of tiles."
    )
    parser.add_argument(
        "--tiles-path", type=str, default="tiles", help="Path to the tiles directory."
    )
    parser.add_argument(
        "--target-image",
        type=str,
        default="target_image.jfif",
        help="Path to the target image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mosaic_image.png",
        help="Path to save the mosaic image.",
    )
    parser.add_argument(
        "--partition-size",
        type=int,
        nargs=2,
        default=[50, 50],
        help="The target image is partitioned into tiles of this size.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        nargs=2,
        default=[50, 50],
        help="Size of the new tiles.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="intersection",
        choices=["correlation", "chi-square", "intersection", "bhattacharyya"],
        help="Method used to compare histogram. Use the OpenCV method name. more info: https://docs.opencv.org/4.x/d8/dc8/tutorial_histogram_comparison.html",
    )
    return parser.parse_args()


def partion_target_image(image, tiles_size):
    width, height = image.shape[:2]
    num_x_tiles = width // tiles_size[0]
    num_y_tiles = height // tiles_size[1]

    image = image[: num_x_tiles * tiles_size[0], : num_y_tiles * tiles_size[1]]

    tiles = np.array_split(image, num_x_tiles, axis=0)
    tiles = [np.array_split(tile, num_y_tiles, axis=1) for tile in tiles]
    tiles = [item for sublist in tiles for item in sublist]  # flatten the list

    return tiles, num_x_tiles, num_y_tiles


def similarityMatrix(listImage1, listImage2, compare_method=cv2.HISTCMP_INTERSECT):
    S = np.zeros((len(listImage1), len(listImage2)))
    progress_bar = tqdm(
        total=len(listImage1), ncols=100, desc="Creating similarity matrix"
    )

    list_histo_1 = [
        cv2.calcHist(img1, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        for img1 in listImage1
    ]
    list_histo_2 = [
        cv2.calcHist(img2, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        for img2 in listImage2
    ]

    for i, hist1 in enumerate(list_histo_1):
        progress_bar.update(1)
        for j, hist2 in enumerate(list_histo_2):
            S[i, j] = cv2.compareHist(hist1, hist2, compare_method)

    progress_bar.close()

    return S

def colorSimilarityMatrix(listImage1, listImage2):
    S = np.zeros((len(listImage1), len(listImage2)))
    progress_bar = tqdm(
        total=len(listImage1), ncols=100, desc="Creating similarity matrix"
    )

    for i, img1 in enumerate(listImage1):
        progress_bar.update(1)
        for j, img2 in enumerate(listImage2):
            S[i, j] = np.linalg.norm(img1 - img2)

    progress_bar.close()

    return S


args = parse_args()
tiles_path = Path(args.tiles_path)
target_image_path = Path(args.target_image)
mosaic_image_path = Path(args.output)
partition_size = tuple(args.partition_size)
new_tiles_size = tuple(args.tile_size)
possible_methods = {
    "correlation": cv2.HISTCMP_CORREL,
    "chi-square": cv2.HISTCMP_CHISQR,
    "intersection": cv2.HISTCMP_INTERSECT,
    "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
}
compare_method = possible_methods[args.method]

console = Console()

tiles_files = list(tiles_path.glob("*"))

console.print(f"Number of tiles: {len(tiles_files)}")
console.print(f"Resizing tiles to new size: {new_tiles_size}")
tiles_images = np.zeros(
    (len(tiles_files), new_tiles_size[0], new_tiles_size[1], 3), dtype=np.uint8
)
tiles_size_partition = np.zeros(
    (len(tiles_files), partition_size[0], partition_size[1], 3), dtype=np.uint8
)

progress_bar = tqdm(total=len(tiles_files), ncols=100, desc="Processing tiles")

for i, tile_file in enumerate(tiles_files):

    progress_bar.update(1)
    progress_bar.set_description(f"Processing {tile_file.name}")
    try:
        img = Image.open(tile_file).convert("RGB")
        preprocess_tile = np.asarray(
            img.resize(new_tiles_size)
        )
        img_same_size_partition = np.asarray(
            img.resize(partition_size)
        )
    except Exception as e:
        console.print(f"\n[red bold]Error[/] : Issue with: {tile_file.name} - {e}")
        continue
    tiles_images[i] = preprocess_tile
    tiles_size_partition[i] = img_same_size_partition

progress_bar.close()

console.print(f"Number of preprocess tiles: {len(tiles_images)}")

# Load target image
target_image = np.asarray(Image.open(target_image_path))

# Partition target image
console.print(f"Partitioning target image with size: {partition_size}")
partitioned_img, new_width, new_height = partion_target_image(
    target_image, partition_size
)
console.print(
    f"The target image has been partitioned into {(new_width, new_height)} tiles."
)

preprocess_tiles_img = tiles_images

console.print("Creating similarity matrix")
# S = similarityMatrix(partitioned_img, preprocess_tiles_img, compare_method)
S = colorSimilarityMatrix(partitioned_img, tiles_size_partition)
console.print(f"Similarity matrix created. Shape: {S.shape}")

# NOW WE CREATE THE MOSAIC
console.print("Creating mosaic image ...")
mosaic = np.zeros(
    (len(partitioned_img), new_tiles_size[0], new_tiles_size[1], 3), dtype=np.uint8
)

for i in range(len(partitioned_img)):
    # we take the 5 most similar tiles
    # top5_idx = np.argsort(S[i])[-5:]
    # # we take the most similar tile
    # idx = np.random.choice(top5_idx)
    idx = np.argmax(S[i])
    mosaic[i] = np.array(preprocess_tiles_img[idx])

mosaic_image = np.zeros(
    (new_width * new_tiles_size[0], new_height * new_tiles_size[1], 3), dtype=np.uint8
)
for i in range(new_width):
    for j in range(new_height):
        mosaic_image[
            i * new_tiles_size[0] : (i + 1) * new_tiles_size[0],
            j * new_tiles_size[1] : (j + 1) * new_tiles_size[1],
        ] = mosaic[i * new_height + j]

Image.fromarray(mosaic_image).save(mosaic_image_path)
console.print(
    f"Mosaic image created [green bold]succefully[/] and saved to {mosaic_image_path}"
)
