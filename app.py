from flask import Flask, render_template, url_for, request, Blueprint
from flask import redirect
import os
import datetime
import time
import work as wk



app = Flask(__name__)
mod = Blueprint('app_main', __name__, template_folder='templates', static_folder='static')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print(APP_ROOT)

indx = wk.index()
anls = wk.analysis()
database = ['', '']
tddfdd = 'TDD'
filename = ""
mp_filename = ""


@mod.route("/")
def index():
    return render_template('index.html', title='Home')

@mod.route("/analysis")
def analysis():
    return render_template('analysis.html', title="Analysis")

@mod.route("/upload", methods=['POST'])
def upload():
    """
    Uploads files and call the perform function
    :param:
    :return: renders the analysis webpage
    """
    if request.method == 'POST':
        pcm = request.files['pcm']
        coord = request.files['coord']
        lpf = request.files['lpf']
        kpi = request.files.getlist("kpi")

        hc = False
        ull = False

        try:
            hc = request.form['hc']
            ull = request.form['ull']
        except:
            pass

        global indx
        indx.inp = [pcm, coord, lpf]
        indx.KPI_Files = kpi

        indx.st = datetime.datetime.strptime(request.form['start_dt'], "%Y-%m-%d").strftime("%d/%m/%Y")
        indx.en = datetime.datetime.strptime(request.form['end_dt'], "%Y-%m-%d").strftime("%d/%m/%Y")

        if ull == 'True':
            indx.ULL = True
        if hc == 'True':
            indx.HC = True

        if request.form['demo'] == "Load Data":
            indx.perform()
            print("perform complete")

        if request.form['demo'] == "Load Data (new format)":
            indx.perform(True)
            print("perform complete")

    return redirect(url_for('app_main.analysis'))


@mod.route("/display_table", methods=['GET', 'POST'])
def display_table():
    """
    :return: csv content as string
    """
    print(request.args)
    global anls
    mode = request.args['mode']
    if request.method == 'GET':

        anls.mode = mode
        anls.varEntry = float(request.args['thold'])

        if mode == 'IH':
            anls.setMode(0)
        elif mode == 'AH':
            anls.setMode(1)
        elif mode == 'PBI':
            anls.setMode(2)
        elif mode == 'PBA':
            anls.setMode(3)
        elif mode == 'LBA':
            anls.setMode(4)
        else:
            pass

        return anls.getResults()
    else:
        return


@mod.route("/site_data", methods=['GET'])
def site_data():
    global database
    global anls
    database = request.args.getlist('sitepop[]')
    print(database)
    anls.database = database
    return "success"


@mod.route("/site_pop_data", methods=['GET'])
def site_pop_data():
    global database
    database = request.args.getlist('sitepop[]')
    global tddfdd
    tddfdd = request.args['tddfdd']
    return "something"


@mod.route("/getGraph", methods=['GET', 'POST'])
def getGraph():
    """
    :param: site_id passed from the API from jquery
    :return:
    """
    global filename
    global anls
    anls.gtp = int(request.form['opt3'])
    anls.database = database
    anls.tddfdd = tddfdd
    fig = anls.get_graph()
    filepath = os.path.dirname(os.path.abspath(__file__))+"/static/images/"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = 'img'+timestr+'.png'
    filepath_filename = filepath + filename
    fig.savefig(filepath_filename)
    return redirect(url_for('app_main.Graph'))


    #return redirect(url_for(graph, file = filename))

    #img = StringIO.StringIO()
    #fig.savefig(img)
    #img.seek(0)
    #return send_file(img, mimetype='image/png')


@mod.route("/Graph", methods=['GET', 'POST'])
def Graph():
    return render_template('graph.html', title="Graph", file=filename)


@mod.route("/getMap", methods=['GET', 'POST'])
def getMap():

    global mp_filename
    mp_filename = anls.plot_folium()
    return redirect(url_for('app_main.Map'))


@mod.route("/Map", methods=['GET', 'POST'])
def Map():
    global mp_filename
    return render_template(mp_filename, title="Map")


app = Flask(__name__)
# url_prefix='/trans'
app.register_blueprint(mod)

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    #port = int(os.environ.get('PORT', 5000))

    app.run(host="0.0.0.0", port=80)
