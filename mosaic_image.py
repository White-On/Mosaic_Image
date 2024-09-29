from rich import print
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import argparse
import chime

from similarity_methods import (
    histogramSimilarityMatrix,
    colorSimilarityMatrix,
    averageColorMatrix,
)

SIMILARITY_MATRIX_METHODS = {
    "histogram": histogramSimilarityMatrix,
    "color": colorSimilarityMatrix,
    "average": averageColorMatrix,
}

HISTOGRAM_COMPARE_METHODS = {
    "correlation": cv2.HISTCMP_CORREL,
    "chi-square": cv2.HISTCMP_CHISQR,
    "intersection": cv2.HISTCMP_INTERSECT,
    "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
}

FREQUENCY_METHODS = {
    "best": lambda x: [1] + [0] * (x - 1),
    "uniform": lambda x: [1 / x] * x,
    "normal": lambda x: generate_normal_distribution(x, 0, 1),
}


def parse_args() -> argparse.Namespace:
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
        "--histo-method",
        type=str,
        default="intersection",
        choices=HISTOGRAM_COMPARE_METHODS.keys(),
        help="Method used to compare histogram. Use the OpenCV method name. more info: https://docs.opencv.org/4.x/d8/dc8/tutorial_histogram_comparison.html",
    )
    parser.add_argument(
        "--similarity-method",
        type=str,
        default="color",
        choices=SIMILARITY_MATRIX_METHODS.keys(),
        help="Method used to compare tiles with target image.",
    )
    parser.add_argument(
        "--max-random-range",
        type=int,
        default=5,
        help="Number of most similar tiles to choose from.",
    )
    parser.add_argument(
        "--select-frequency",
        type=str,
        default="best",
        choices=["best", "uniform", "normal"],
        help="Frequency of selecting the most similar tiles.",
    )

    return parser.parse_args()


def partion_target_image(image: np.ndarray, tiles_size: tuple) -> tuple:
    """
    Partition the target image into tiles of a given size.
    """
    width, height = image.shape[:2]
    num_x_tiles = width // tiles_size[0]
    num_y_tiles = height // tiles_size[1]

    image = image[: num_x_tiles * tiles_size[0], : num_y_tiles * tiles_size[1]]

    tiles = np.array_split(image, num_x_tiles, axis=0)
    tiles = [np.array_split(tile, num_y_tiles, axis=1) for tile in tiles]
    tiles = [item for sublist in tiles for item in sublist]  # flatten the list

    color = "red" if width % tiles_size[0] or height % tiles_size[1] else "green"
    print(
        f"While partitioning the target image, [{color} bold]we lost {width - num_x_tiles * tiles_size[0]} pixels in width and {height - num_y_tiles * tiles_size[1]} pixels in height[/]."
    )

    return tiles, num_x_tiles, num_y_tiles


def generate_normal_distribution(size: int, center_index: int, sigma: float) -> list:
    """
    Generates a list of floats representing a normal distribution centered on a given index.
    """
    x = np.linspace(0, size - 1, size)
    mu = center_index
    distribution = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )
    normalized_distribution = distribution / np.sum(distribution)
    return normalized_distribution.tolist()


def prepare_tiles(tiles_path: Path, tile_size: tuple, partition_size: tuple) -> tuple:
    """
    Prepare the tiles by resizing them to the new size.
    """
    tiles_files = list(tiles_path.glob("*"))

    print(f"Number of tiles in the given file: {len(tiles_files)}")
    print(f"Resizing tiles to new size: {tile_size}")
    tiles_images = np.zeros(
        (len(tiles_files), tile_size[0], tile_size[1], 3), dtype=np.uint8
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
            preprocess_tile = np.asarray(img.resize(tile_size))
            img_same_size_partition = np.asarray(img.resize(partition_size))
        except Exception as e:
            print(f"\n[red bold]Error[/] : Issue with: {tile_file.name} - {e}")
            print(f"Be careful that all the files in the directory are images")
            print(f"Skipping {tile_file.name}")
            chime.error(sync=True)
            continue
        tiles_images[i] = preprocess_tile
        tiles_size_partition[i] = img_same_size_partition

    progress_bar.close()

    print(
        f"We have resized {len(tiles_images)}/{len(tiles_files)} ({len(tiles_images)/len(tiles_files):.0%}) tiles to the new size."
    )

    return tiles_images, tiles_size_partition


def create_mosaic(
    tiles_images: np.ndarray,
    similarity_matrix: np.ndarray,
    max_random_range: int,
    frequency: list,
    new_width: int,
    new_height: int,
    new_tiles_size: tuple,
) -> np.ndarray:
    """
    Create the mosaic image from the tiles images and the similarity matrix.
    """
    tile_height, tile_width = new_tiles_size
    mosaic_image = np.zeros(
        (new_width * tile_height, new_height * tile_width, 3), dtype=np.uint8
    )

    for i in range(new_width):
        for j in range(new_height):
            # Logique pour sélectionner l'image de tuile appropriée
            partitioned_idx = i * new_height + j
            selected_tile = select_tile(
                partitioned_idx,
                tiles_images,
                similarity_matrix,
                max_random_range,
                frequency,
            )
            start_row, end_row = i * tile_height, (i + 1) * tile_height
            start_col, end_col = j * tile_width, (j + 1) * tile_width
            mosaic_image[start_row:end_row, start_col:end_col] = selected_tile

    return mosaic_image


def select_tile(
    partitioned_idx: int,
    tiles_images: np.ndarray,
    similarity_matrix: np.ndarray,
    max_random_range: int,
    frequency: list,
) -> np.ndarray:
    """
    Select the tile to use in the mosaic image.
    """
    top_idx = np.argsort(similarity_matrix[partitioned_idx])[:max_random_range]
    idx = np.random.choice(top_idx, p=frequency)
    return np.array(tiles_images[idx])


def main():

    chime.theme("pokemon")
    # Parse arguments and set variables
    args = parse_args()
    tiles_path = Path(args.tiles_path)
    target_image_path = Path(args.target_image)
    mosaic_image_path = Path(args.output)
    partition_size = tuple(args.partition_size)
    new_tiles_size = tuple(args.tile_size)
    max_random_range = args.max_random_range
    frequency = FREQUENCY_METHODS.get(args.select_frequency)(max_random_range)
    chosen_similarity_method = SIMILARITY_MATRIX_METHODS.get(args.similarity_method)
    chosen_histo_method = HISTOGRAM_COMPARE_METHODS.get(args.histo_method)

    # TODO :
    # - add more error handling
    # - adaptative partition size to the picture size
    # - memory seems to be a problem with big images -> safety net to avoid memory error

    # We resize the tiles to the new size, this way we ensure that the tiles are all the same size
    # and that the mosaic image will have a uniform look.
    # We keep also resize the tiles to the partition size of the target image to compare them later.
    tiles_images, tiles_size_partition = prepare_tiles(
        tiles_path, new_tiles_size, partition_size
    )

    # Load target image
    target_image = np.asarray(Image.open(target_image_path))

    # Partition target image
    # This means that we divide the target image into smaller images.
    # We can lose some pixels if the target image size is not a multiple of the partition size.
    print(f"Partitioning target image with size: {partition_size}")
    partitioned_img, new_width, new_height = partion_target_image(
        target_image, partition_size
    )

    print(
        f"The target image has been partitioned into {(new_width, new_height)} smaller images."
    )

    # Create similarity matrix
    # This matrix will contain a value for each tile of the target image and each tile of the tiles images.
    # The value represents a score of similarity between the two tiles that depend on the chosen method.
    similarity_matrix = chosen_similarity_method(
        partitioned_img, tiles_size_partition, compare_method=chosen_histo_method
    )

    # Create mosaic image
    mosaic_image = create_mosaic(
        tiles_images,
        similarity_matrix,
        max_random_range,
        frequency,
        new_width,
        new_height,
        new_tiles_size,
    )

    Image.fromarray(mosaic_image).save(mosaic_image_path)
    print(
        f"Mosaic image created [green bold]succefully[/] and saved as [blue]{mosaic_image_path}[/]"
    )
    chime.success(sync=True)


if __name__ == "__main__":
    main()
