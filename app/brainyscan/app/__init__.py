import os
import PIL
import io
import flask

from .controller import BrainyController

def create_app(test_config = None):
    app = flask.Flask(__name__, instance_relative_config = True)

    if test_config is None:
        app.config.from_pyfile('config.py', silent = True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    controller = BrainyController(
        os.path.join(app.instance_path, 'model.keras'),
        (299, 299)
    )

    @app.route('/<string:fname>.js')
    def service_worker(fname):
        _, fname = os.path.split(fname)
        fname = f'{fname}.js'
        if not os.path.exists(os.path.join(app.static_folder, fname)):
            flask.abort(404)
        return flask.send_from_directory('static', fname, mimetype='application/javascript')

    @app.route('/')
    @app.route('/index.html')
    def default() -> str:
        return flask.render_template('index.html')
    
    @app.route('/api')
    @app.route('/api/v1')
    def reoute_to_default() -> flask.Response:
        return flask.redirect(flask.url_for('default'))

    @app.get('/api/v1/scan')
    def get_scan_stats() -> tuple[flask.Response, int]:
        return flask.jsonify(controller.get_stats()), 200

    @app.post('/api/v1/scan')
    def start_scan() -> tuple[flask.Response, int]:
        if 'file' not in flask.request.files:
            return flask.jsonify({'error': 'No file part'}), 422
        
        file = flask.request.files['file']

        if file.filename == '':
            return flask.jsonify({'error': 'No selected file'}), 422

        fileext = os.path.splitext(file.filename)[1]
        if not fileext:
            return flask.jsonify({'error': 'Invalid file type'}), 422
        
        fileext = fileext.lower()
        if fileext not in ('.dcm') and not file.content_type.startswith('image/jpeg'):
            print('here')
            return flask.jsonify({'error': 'Invalid file type'}), 422

        result = controller.start_predict(file)
        if not result:
            return flask.jsonify({'error': 'Invalid file type'}), 422

        return flask.jsonify(result), 200

    @app.get('/api/v1/scan/<string:id>')
    def get_scan_classification(id: str) -> tuple[flask.Response, int]:
        if not id.isalnum():
            flask.abort(400, description="Invalid ID supplied")

        result = controller.get_predict(id)
        if result is None:
            flask.abort(404, description="MRI scan not found")

        return flask.jsonify(result), 200

    @app.get('/api/v1/scan/<string:id>/image')
    def get_scan_image(id: str) -> flask.Response:
        img = controller.get_image(id)
        if img is None:
            return flask.send_from_directory(app.static_folder, 'brain_placement.jpg')
        else:
            buf = io.BytesIO()
            img['image'].seek(0)
            img['image'].save(buf, format='PNG')
            buf.seek(0)
            return flask.send_file(buf, mimetype='image/png')

    return app

