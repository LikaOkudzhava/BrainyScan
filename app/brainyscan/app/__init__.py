import os
from flask import Flask, request, render_template, url_for, jsonify, abort

from .controller import BrainyController

def create_app(test_config = None):
    app = Flask(__name__, instance_relative_config = True)

    print(app.instance_path)

    if test_config is None:
        app.config.from_pyfile('config.py', silent = True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    controller = BrainyController(
        os.path.join(app.instance_path, 'model.keras')
    )

    @app.route('/')
    @app.route('/api')
    @app.route('/api/v1')
    def default():
        return render_template(
            'index.html',
            favicon = url_for('static', filename='favicon.png')
        )

    @app.get('/api/v1/scan')
    def get_scan_stats():
        return jsonify(
            controller.get_stats()
        ), 200

    @app.post('/api/v1/scan')
    def start_scan():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 422
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 422

        if not file.content_type.startswith('image/jpeg'):
            return jsonify({'error': 'Invalid file type'}), 422

        return jsonify(
            controller.start_predict(file)
        ), 200

    @app.get('/api/v1/scan/<string:id>')
    def get_scan_classification(id: str):
        if not id.isalnum():
            abort(400, description="Invalid ID supplied")

        result = controller.get_predict(id)
        if result is None:
            abort(404, description="MRI scan not found")

        print(result)
        return jsonify(result), 200

    return app

