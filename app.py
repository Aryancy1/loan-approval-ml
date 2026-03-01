import pandas as pd
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model
model = joblib.load("loan_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        form_data = request.form.to_dict()

        # Convert to DataFrame
        input_df = pd.DataFrame([form_data])

        # Convert numeric fields
        numeric_fields = [
            'ApplicantIncome',
            'CoapplicantIncome',
            'LoanAmount',
            'Loan_Amount_Term',
            'Credit_History'
        ]

        for field in numeric_fields:
            input_df[field] = pd.to_numeric(input_df[field])

        # One-hot encoding (same as training)
        input_df = pd.get_dummies(input_df)

        # Align columns with training model
        model_columns = model.feature_names_in_
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Prediction
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            result = "Loan Approved ✅"
        else:
            result = "Loan Not Approved ❌"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error occurred: {e}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)