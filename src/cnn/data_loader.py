import pickle
import numpy as np
from utils import corr2d, extract_enhanced_features


def build_dataset(residuals_dict, scanner_fps, fp_keys):
    """
    Returns:
        X_img : (N, 256, 256, 1)
        X_feat: (N, F)
        y     : (N,)
    """

    X_img, X_feat, y = [], [], []

    for dataset_name in residuals_dict:
        for scanner, dpi_dict in residuals_dict[dataset_name].items():
            for dpi, residual_list in dpi_dict.items():
                for res in residual_list:

                    # Image
                    X_img.append(res[..., None])

                    # Correlation features (PRNU)
                    corr_feats = [
                        corr2d(res, scanner_fps[k]) for k in fp_keys
                    ]

                    # Enhanced features
                    enh_feats = extract_enhanced_features(res)

                    X_feat.append(corr_feats + enh_feats)
                    y.append(scanner)

    return (
        np.array(X_img, dtype=np.float32),
        np.array(X_feat, dtype=np.float32),
        np.array(y),
    )
