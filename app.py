from flask import Flask, request, jsonify, render_template
from src.mlproject.pipelines.prediction_pipeline import PredictPipeline, CustomData
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import os, sys

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()

        if not body:
            return jsonify({"status": "error", "message": "No input data provided"}), 400

        # ── Validate required fields ──────────────────
        required = [
            "gender", "nationality", "age",
            "english_grade", "math_grade", "sciences_grade",
            "language_grade", "portfolio_rating",
            "coverletter_rating", "refletter_rating"
        ]
        missing = [f for f in required if body.get(f) is None]
        if missing:
            return jsonify({
                "status" : "error",
                "message": f"Missing fields: {missing}"
            }), 400

        
        data = CustomData(
            gender             = str(body["gender"]),
            nationality        = str(body["nationality"]),
            age                = int(body["age"]),
            english_grade      = float(body["english_grade"]),
            math_grade         = float(body["math_grade"]),
            sciences_grade     = float(body["sciences_grade"]),
            language_grade     = float(body["language_grade"]),
            portfolio_rating   = float(body["portfolio_rating"]),
            coverletter_rating = float(body["coverletter_rating"]),
            refletter_rating   = float(body["refletter_rating"]),
        )

        df                = data.get_data_as_dataframe()
        pipeline          = PredictPipeline()
        prediction, proba = pipeline.predict(df)

        result = "Eligible" if int(prediction[0]) == 1 else "Not Eligible"

        logging.info(f"Prediction: {result} | Confidence: {proba[0]:.4f}")

        return jsonify({
            "status"    : "success",
            "prediction": result,
            "confidence": f"{round(float(proba[0]) * 100, 2)}%"
        }), 200

    except CustomException as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)