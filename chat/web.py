from flask import Flask, request, jsonify
import your_script  # Assuming your_script contains the defined code

app = Flask(__name__)

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text', '')
    
    # Here you should call the function where you process the text
    # assuming process_text is that function and it returns the processed text
    result = your_script.process_text(text)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
