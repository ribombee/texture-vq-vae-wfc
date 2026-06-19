from image_diversity import ClipMetrics, InceptionMetrics
from pathlib import Path
import pandas as pd

def __parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the directory containing the image resultz.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = __parse_args()
    data_path = Path(args.data_path)

    reg_path = data_path / "our_images"
    pert_path = data_path / "perturbed_images"
    nca_path = data_path / "nca_images"
    ab_noes_path = data_path / "no_es_images"
    ab_nogate_path = data_path / "no_gating_images"

    clip_metrics = ClipMetrics(n_eigs=20)
    inception_metrics = InceptionMetrics(n_eigs=20)

    our_tce = clip_metrics.tce(str(reg_path))
    nca_tce = clip_metrics.tce(str(nca_path))
    perturbed_tce = clip_metrics.tce(str(pert_path))
    no_es_tce = clip_metrics.tce(str(ab_noes_path))
    no_gated_tce = clip_metrics.tce(str(ab_nogate_path))

    our_tie= inception_metrics.tie(str(reg_path))
    nca_tie = inception_metrics.tie(str(nca_path))
    perturbed_tie = inception_metrics.tie(str(pert_path))
    no_es_tie = inception_metrics.tie(str(ab_noes_path))
    no_gated_tie = inception_metrics.tie(str(ab_nogate_path))


    print(f"Our TCE: {our_tce}")
    print(f"NCA TCE: {nca_tce}")
    print(f"Perturbed TCE: {perturbed_tce}")
    print(f"No ES TCE: {no_es_tce}")
    print(f"No Gated TCE: {no_gated_tce}")


    print(f"Our TIE : {our_tie}")
    print(f"NCA TIE : {nca_tie}")
    print(f"Perturbed TIE : {perturbed_tie}")
    print(f"No ES TIE : {no_es_tie}")
    print(f"No Gated TIE : {no_gated_tie}")

    # real quick just read and parse results.csv
    results_path = data_path / "results.csv"
    results_df = pd.read_csv(results_path).drop(columns=["filename"])

    col_means = results_df.mean()
    col_stds = results_df.std()
    print(f"Results CSV means:\n{col_means}")
    print(f"Results CSV stds:\n{col_stds}")