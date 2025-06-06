import os
from flask import Flask, Response, request, render_template, url_for, jsonify, abort

from .controller import BrainyController

def create_app(test_config = None):
    app = Flask(__name__, instance_relative_config = True)

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
    def default() -> str:
        return render_template(
            'index.html'
        )

    @app.route('/api')
    @app.route('/api/v1')
    def reoute_to_default() -> Response:
        return redirect(url_for('/'))

    @app.get('/api/v1/scan')
    def get_scan_stats() -> tuple[Response, int]:
        return jsonify(
            controller.get_stats()
        ), 200

    @app.post('/api/v1/scan')
    def start_scan() -> tuple[Response, int]:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 422
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 422

        if not file.content_type.startswith('image/jpeg'):
            return jsonify({'error': 'Invalid file type'}), 422

        return jsonify(controller.start_predict(file)), 200

    @app.get('/api/v1/scan/<string:id>')
    def get_scan_classification(id: str) -> tuple[Response, int]:
        if not id.isalnum():
            abort(400, description="Invalid ID supplied")

        result = controller.get_predict(id)
        if result is None:
            abort(404, description="MRI scan not found")

        return jsonify(result), 200

    return app

