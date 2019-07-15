from flask import Blueprint

bp = Blueprint('dashboards',
               __name__,
               url_prefix='/dashboards',
               )

from app.dashboards import routes
