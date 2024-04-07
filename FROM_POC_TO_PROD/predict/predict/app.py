from flask import Flask, request, render_template_string
from predict.predict.run import TextPredictionModel

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>stackoverflow tags Prediction Service</title>
    <!-- Bootstrap CSS -->
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f7f7f7;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .container {
            margin-top: 50px;
        }
        .card {
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s;
            border-radius: 5px; /* 5px rounded corners */
        }
        .card:hover {
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }
        .card-body {
            padding: 20px;
        }
        .card-title {
            margin-bottom: 20px;
            text-align: center;
            color: #007bff;
        }
        #predictions {
            margin-top: 20px;
            color: #333;
        }
        .footer {
            margin-top: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="card-body">
                <h3 class="card-title">stackoverflow tag Prediction</h3>
                <form action="/" method="post">
                    <div class="form-group">
                        <textarea class="form-control" name="text" rows="4" placeholder="Type your stackoverflow title here..."></textarea>
                    </div>
                    <button type="submit" class="btn btn-primary btn-block">Predict</button>
                </form>
                {% if predictions %}
                <div id="predictions" class="alert alert-success" role="alert">
                    <h4 class="alert-heading">Predictions</h4>
                    <p>{{ predictions }}</p>
                </div>
                {% endif %}
            </div>
        </div>
        <div class="footer">
            <p>stackoverflow tags Prediction Service</p>
        </div>
    </div>
    <!-- Bootstrap JS, Popper.js, and jQuery -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.3/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
'''


@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    if request.method == 'POST':
        text_list = [request.form['text']]
        model = TextPredictionModel.from_artefacts('train/data/artefacts/2024-01-06-12-46-21')
        predictions = model.predict(text_list)
    return render_template_string(HTML_TEMPLATE, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)

