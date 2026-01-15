import os, pickle

RES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "results", "hybrid_cnn", "official_wiki_residuals.pkl"
)
RES_PATH = os.path.abspath(RES_PATH)

with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)

print("Top-level keys:", list(residuals_dict.keys()))
for ds_name, ds_dict in residuals_dict.items():
    print(f"{ds_name}: {len(ds_dict)} scanners")

    # print one scanner example
    if ds_dict:
        scanner, dpi_dict = next(iter(ds_dict.items()))
        print("  example scanner:", scanner)
        if isinstance(dpi_dict, dict) and dpi_dict:
            dpi, res_list = next(iter(dpi_dict.items()))
            print("  dpi:", dpi, "num residuals:", len(res_list))
