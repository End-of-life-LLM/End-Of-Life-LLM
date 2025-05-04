from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/fine-tuning")
def finetuning():
    return render_template("fine-tuning.html")