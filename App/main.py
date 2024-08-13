from flask import Flask, request, send_file, make_response, url_for, render_template
from main_ds_test import detect

app = Flask(__name__)
app.debug=True
@app.route("/")
def login():
    return render_template("Login/index.html")

@app.route("/login")
def Login():
    return render_template("Login/index.html")

@app.route('/home_page', methods=["GET", "POST"])
def Home():
    return render_template("Home/home.html")

@app.route('/about')
def about():
    return render_template("About/about.html")

@app.route('/upload_image')
def upload_image():
    return render_template("Upload_Image/upload_image.html")

@app.route('/result', methods=['GET', 'POST'])
def result():
    img = request.files['in_img']
    i_path = "static\\uploaded_files\\image.jpg"
    img.save(i_path)
    msg =detect.detect()
    return render_template("Result/result.html", content=msg)#, img_pth=image_path)

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5001)
   