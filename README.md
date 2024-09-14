
# Mosaic Image Project 🎨🖼️🧩

## Project Overview
This project generates a mosaic image by recreating a target image using smaller images from a database. The smaller images (tiles) are selected based on their color similarity to the corresponding regions in the target image, creating a visually mosaic effect.

## Features
- Create mosaic images from any target image.
- Use a custom image database to generate the mosaic.
- Adjust tile size and other parameters to fine-tune the mosaic creation process.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mosaic-image-project.git
   cd mosaic-image-project
   ```

2. **Install the required dependencies:**
   - Make sure you have Python 3.x installed.
   - Install necessary Python libraries by running:
     ```bash
     pip install -r requirements.txt
     ```

## Usage

1. **Prepare your image database:**
   - Collect the small images (tiles) that you want to use in your mosaic.
   - Store them in a folder (e.g., `image_database/`).
   - The images will be resized to the tile size specified in the mosaic generator. For best results, use images of similar aspect ratios or crop them to the desired size before creating the mosaic.

2. **Run the mosaic generator:**
   - Use the following command to create the mosaic:
     ```bash
     python mosaic_generator.py --target-image path/to/target_image.jpg --tiles-path path/to/image_database/ --output path/to/output_mosaic.jpg --tile-size 50 --partition-size 50
     ```
   - Replace the paths and `tile_size` with your desired values.

3. **Parameters:**
   - `--target-image`: Path to the image you want to recreate as a mosaic.
   - `--tiles-path`: Path to the folder containing your small images.
   - `--output`: Path where the generated mosaic image will be saved.
   - `--tile-size`: Size of the tiles in the mosaic (default is 50 pixels).
   - `--partition-size`: The target image is partitioned into tiles of this size (default is 50 pixels).
   - `--method`: Method used to compare histogram, you can choose between `correlation`, `chi-square`, `intersection`, and `bhattacharyya` (default is `intersection`).
See the OpenCV documentation for more information on histogram comparison methods: [OpenCV Doc](https://docs.opencv.org/4.x/d8/dc8/tutorial_histogram_comparison.html)

## Example

```bash
python mosaic_generator.py --target-image images/sunset.jpg --tiles-path images/tiles/ --output output/mosaic_sunset.jpg --tile-size 50 --partition-size 50
```

This will create a mosaic of the `sunset.jpg` image using tiles from the `images/tiles/` directory and save the mosaic to `output/mosaic_sunset.jpg`.

## Contributing
Feel free to contribute to this project! You can fork the repository and create a pull request for any improvements, new features, or bug fixes.

