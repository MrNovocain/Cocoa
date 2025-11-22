"""Baseline experiment for Cocoa data."""

import logging

from ..logging_config import setup_logging
from ..data import load_cocoa_raw, preprocess_cocoa, build_features
from ..models import naive_forecast, evaluate_forecast
from ..config import settings


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Loading raw Cocoa data...")
    df_raw = load_cocoa_raw()

    logger.info("Preprocessing...")
    df_clean = preprocess_cocoa(df_raw)

    logger.info("Building features...")
    df_feat = build_features(df_clean)

    y = df_feat[settings.target_column]
    y_hat = naive_forecast(y, horizon=1)

    metrics = evaluate_forecast(y, y_hat)
    logger.info("Baseline metrics: %s", metrics)


if __name__ == "__main__":
    main()
