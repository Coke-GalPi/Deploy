from flask import render_template, jsonify
from app import app

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.errorhandler(404)
def not_found_error(error):
    code_error = {
        'number': 404,
        'message': 'Not found',
        'description': 'La página que buscas no pudo ser encontrada.'
    }
    return render_template('error.html', error=code_error)

@app.errorhandler(500)
def internal_error(error):
    code_error = {
        'number': 500,
        'message': 'Internal Server Error',
        'description': 'Ocurrió un error interno en el servidor.'
    }
    return render_template('error.html', error=code_error)