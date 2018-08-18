from flask import Flask

from web.neural import neural

app = Flask(__name__)
app.register_blueprint(neural, url_prefix='/neural')
app.run(
    host=app.config.get('HOST', '0.0.0.0'),
    port=app.config.get('PORT', 1234),
    debug=True
)
