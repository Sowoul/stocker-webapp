from flask import Flask,render_template,request
from pred import predict
app=Flask(__name__)

@app.route('/')
def serach():
    company=request.args.get('search')
    return  predict(company)


app.run(port=8080)