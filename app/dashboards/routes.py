from flask import render_template
from flask_login import login_required
from . import bp
from . import Dash_App1, Dash_App2

@bp.route('/app1')
@login_required
def app1_template():
    return render_template('dashboards/app1.html', dash_url=Dash_App1.url_base)

@bp.route('/app2')
@login_required
def app2_template():
    return render_template('dashboards/app2.html', dash_url=Dash_App2.url_base)