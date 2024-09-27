import cv2
import numpy as np
from tqdm import tqdm


def histogramSimilarityMatrix(
    listImage1: list, listImage2: list, compare_method=cv2.HISTCMP_INTERSECT, **kwargs
) -> np.ndarray:
    """
    Calculate the histogram similarity between each image
    """
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

    return S * -1


def colorSimilarityMatrix(listImage1: list, listImage2: list, **kwargs) -> np.ndarray:
    """
    Calculate the euclidean distance between each image
    """
    S = np.zeros((len(listImage1), len(listImage2)))
    progress_bar = tqdm(
        total=len(listImage1), ncols=100, desc="Creating similarity matrix"
    )

    for i, img1 in enumerate(listImage1):
        progress_bar.update(1)
        for j, img2 in enumerate(listImage2):
            # change dtype to int16 to avoid overflow
            img1 = img1.astype(np.int16)
            img2 = img2.astype(np.int16)
            S[i, j] = np.linalg.norm(img1 - img2)

    progress_bar.close()

    return S


def averageColorMatrix(listImage1: list, listImage2: list, **kwargs) -> np.ndarray:
    """
    Calculate the average color of each image and then calculate the euclidean distance between them
    """
    S = np.zeros((len(listImage1), len(listImage2)))
    progress_bar = tqdm(
        total=len(listImage1), ncols=100, desc="Creating similarity matrix"
    )

    for i, img1 in enumerate(listImage1):
        progress_bar.update(1)
        for j, img2 in enumerate(listImage2):
            # change dtype to int16 to avoid overflow
            img1 = img1.astype(np.int16)
            img2 = img2.astype(np.int16)
            S[i, j] = np.linalg.norm(np.mean(img1, axis=2) - np.mean(img2, axis=2))

    progress_bar.close()

    return S
