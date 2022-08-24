# from crypt import methods
from flask import Flask
from flask import render_template, request
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def user_login():
    return render_template('login.html')

@app.route('/register')
def user_register():
    return render_template('register.html')

@app.route('/dashboard')
def user_dashboard():
    return render_template('Dashboard-index.html')

@app.route('/test', methods=['POST'])
def test():
    uname=request.form['uname']  
    passwrd=request.form['pass']  
    if uname=="harsh" and passwrd=="123":  
        return "Welcome %s" %uname  
    # return 'Hello'

if __name__ == "__main__":
    app.run(debug=True)