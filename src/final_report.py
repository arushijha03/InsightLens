import json
from pipeline import full_pipeline  # your main pipeline function

def generate_final_report(reviews_csv_path, output_json_path="final_report.json"):
    """
    Run full pipeline and save final insights as JSON
    """
    # Run pipeline to get structured insights
    report = full_pipeline(reviews_csv_path)

    # Save to JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)

    print(f"Final report saved at: {output_json_path}")
    return report

if __name__ == "__main__":
    reviews_csv_path = "data/cleaned_amazon_reviews.csv"
    generate_final_report(reviews_csv_path)