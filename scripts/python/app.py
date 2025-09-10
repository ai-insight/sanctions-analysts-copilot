"""
Main application orchestrator for Sanctions False Positive Detection.

This module coordinates the model and UI components to create a complete
Streamlit application for sanctions screening analysis.
"""

import streamlit as st

# Import our modular components
from model import (
    load_data, train_and_evaluate, get_coeff_table, 
    build_scored_test_df, predict_single_alert
)
from ui_components import (
    setup_page_config, render_header, render_sidebar_controls,
    render_dataset_overview, render_model_results, render_model_interpretability,
    render_single_alert_controls, render_prediction_results, render_test_results,
    render_footer, show_error
)


def main() -> None:
    """
    Main Streamlit application for sanctions false positive detection.
    
    This application orchestrates the model and UI components to provide:
    - Dataset loading and preview
    - Model training and evaluation
    - Interactive single alert testing
    - Model interpretability through coefficients
    - Downloadable results
    """
    # Setup page configuration
    setup_page_config()
    
    # Render main header
    render_header()
    
    # Render sidebar controls and get user inputs
    use_sample, uploaded_file, test_size, threshold = render_sidebar_controls()

    try:
        # Load and display data
        dataset = load_data(use_sample, uploaded_file)
        render_dataset_overview(dataset)

        # Train and evaluate model
        with st.spinner("Training model..."):
            pipeline, features_test, labels_test, probabilities, predictions, confusion_matrix_result, metrics_df = train_and_evaluate(
                dataset=dataset, test_size=test_size, threshold=threshold
            )

        # Display model results
        render_model_results(metrics_df, confusion_matrix_result)

        # Model interpretability section
        coefficients_table = get_coeff_table(pipeline)
        render_model_interpretability(coefficients_table)

        # Interactive alert testing section
        single_alert = render_single_alert_controls()

        if st.button("🔍 Analyze Alert", type="primary"):
            true_match_probability, prediction, explanation = predict_single_alert(
                pipeline, single_alert, threshold
            )
            render_prediction_results(
                true_match_probability, prediction, threshold, explanation
            )

        # Test set results section
        scored_dataset = build_scored_test_df(features_test, labels_test, probabilities, predictions)
        render_test_results(scored_dataset)

    except Exception as e:
        show_error(str(e))
    
    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
