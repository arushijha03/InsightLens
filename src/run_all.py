# run_all.py
import os
from final_report import generate_final_report
from visualization import visualize_final_report

def main():
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    reviews_csv_path = os.path.join(BASE_DIR, "src", "cleaned_amazon_reviews.csv")
    output_json_path = os.path.join(BASE_DIR, "final_report.json")

    # 1. Run pipeline and generate JSON report
    report = generate_final_report(reviews_csv_path, output_json_path)

    # 2. Visualize insights
    visualize_final_report(output_json_path)
    print("✅ Pipeline executed successfully. Report and visualizations generated.")

if __name__ == "__main__":
    main()