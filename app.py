from flask import Flask, request, render_template, flash
import pickle
import numpy as np

app = Flask(__name__, static_url_path='/static', template_folder='templates')
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open('model.pkl', 'rb'))
    try:
        int_features = [int(x) for x in request.form.values()]
        final_features = [int_features[0]]
        final_features.extend(int_features[2:4])
        final_features.append(0)
        final_features.append(int_features[1])
        final_features.append(int_features[4])
        final_features.extend([0]*11)
        features = [np.array(final_features)]
        print(features)
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)
        output = prediction[0]
        probability_d = f'{prediction_proba[0, 1]*100:.1f}%'
        probability_r = f'{prediction_proba[0, 0]*100:.1f}%'

        prediction_text = f'Will {"deceased" if output == 1 else "recover"}'
        probability_text = f'Probability is {probability_d if output == 1 else probability_r}'
        return render_template('index.html', prediction_text=prediction_text, probability_text=probability_text)

    except Exception as e:
        flash(e)
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
