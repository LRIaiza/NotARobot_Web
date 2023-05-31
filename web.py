import cv2
from tensorflow import keras
import numpy as np
import os
from skimage import transform
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp', 'webp'])
# app.config['UPLOAD_FOLDER'] = "C:/Users/lilyr/OneDrive/Desktop/NotARobot_WebApp/static/Images"
model = keras.models.load_model('my_model')

def img_pred(img):
    np_image = np.array(img).astype('float32') / 255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)
    array = model.predict(np_image)
    if array[0] > 0.5:
        return "Real"
    else:
        return "Fake"

def test_image(img_path):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print(faces)
    for (x, y, w, h) in faces:
        cropped_img = img[y:y+h, x:x+w]
        if img_pred(img) == "Real":
            cv2.putText(img, img_pred(img), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        else:
            cv2.putText(img, img_pred(img), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 12, 255), 2)
        
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/send-message',methods=['GET', 'POST'])
def message():
    return render_template('index.html')


@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/upload.html')
def upload_html():
    if (os.path.isfile((os.path.join(app.config['UPLOAD_FOLDER'], "processed_image.png")))):
        os.remove((os.path.join(app.config['UPLOAD_FOLDER'], "processed_image.png")))
    return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']        
        if f.filename == '':
            return redirect(url_for('upload_html'))
        if f and allowed_file(f.filename):
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
            f1 = test_image(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
            processed_filename = secure_filename("processed_image.png")
            cv2.imwrite((os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)), f1)
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
            # processed_image = url_for('static', filename='Images/' + processed_filename)
            return render_template('upload.html')
    return redirect(url_for('upload_html'))

if __name__ == '__main__':
    app.run(debug=True)


# import cv2
# from tensorflow import keras
# import numpy as np
# import os
# from skimage import transform
# from flask import Flask, render_template, request, redirect, url_for
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# ## Where unput images are saved
# # app.config['UPLOAD_FOLDER'] = 'C:/Users/Thomas/Desktop/WEbsite Demo/static/Images'
# app.config['UPLOAD_FOLDER'] = 'C:/Users/lilyr/OneDrive/Desktop/Web Demo/static/Images'

# model = keras.models.load_model('my_model')

# #Runs a image through the model to determine its label
# def img_pred(img):
#     np_image = np.array(img).astype('float32')/255
#     np_image = transform.resize(np_image, (256, 256, 3))
#     np_image = np.expand_dims(np_image, axis=0)
#     array = model.predict(np_image)
#     if(array[0] > 0.5):
#         return "Real"
#     else:
#         return "Fake"

# #Processes an input image, crops it and returns it with bounding boxes
# def test_image(img_path):
#     # Load the cascade
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     # Read the input image
#     img = cv2.imread(img_path)    
#     # Convert into grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#     print(faces)
#     # Draw rectangle around the faces
#     for (x, y, w, h) in faces:
#         cropped_img = img[y:y+h,x:x+w]
#         if(img_pred(img) == "Real"):
#             cv2.putText(img, img_pred(img), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)    
#         else:
#             cv2.putText(img, img_pred(img), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,12,255), 2)    
        
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     return img

# #Loads the upload page of the website
# @app.route('/upload')
# def upload_html():
#    return render_template('upload.html')

# #Loads the image when the POST method is used	
# @app.route('/uploader', methods = ['GET', 'POST'])
# def upload_file():
#    if request.method == 'POST':
#       f = request.files['file']
#       f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
#       f1 = test_image(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
#       cv2.imwrite((os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))),f1)
#    return redirect(url_for('static', filename='Images/' + f.filename), code=301)
		
# if __name__ == '__main__':
#    app.run(debug = True)