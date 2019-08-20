from flask import jsonify, request, url_for, g, abort
from app.models import User
from app.api import bp
from app.api.auth import token_auth
from app.locallib.db_create.Physio import create_physio_from_polygons
import pandas as pd


@bp.route('/physio', methods=['GET'])
# @token_auth.login_required
def get_physio():
    polygons_path = request.args.get('polygons_path')
    unique_name = request.args.get('unique_name')
    try:
        df = create_physio_from_polygons(polygons_path, unique_name)
        table = pd.pivot_table(df, values='VALUE', index='NUMERO_STATION', columns=['VARIABLE'])
        output = table.reset_index().to_json(orient='records')
    except:
        output = 'Some or all of your polygons don"t have a column named : ' + unique_name
    return output
