import os
import cv2
import numpy as np
import mysql.connector
import tensorflow as tf

from flask import Flask, render_template, request, redirect, session
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

app = Flask(__name__)
app.secret_key = "secret123"

# =============================
# FOLDERS
# =============================

UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# =============================
# MODEL
# =============================

MODEL_PATH = "model/breast_cancer_densenet_model.keras"

model = load_model(MODEL_PATH)

IMG_SIZE = 224
LAST_CONV_LAYER = "conv5_block16_concat"

# =============================
# DATABASE
# =============================

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Yash@8765",
    database="breast_cancer_db"
)

cursor = db.cursor()

# =============================
# HOME
# =============================

@app.route("/")
def index():
    return render_template("index.html")


# =============================
# USER REGISTRATION
# =============================

@app.route("/register", methods=["GET","POST"])
def register():

    if request.method=="POST":

        name=request.form["name"]
        email=request.form["email"]
        password=request.form["password"]
        age=request.form["age"]
        gender=request.form["gender"]
        phone=request.form["phone"]

        sql="INSERT INTO users(name,email,password,age,gender,phone) VALUES(%s,%s,%s,%s,%s,%s)"

        cursor.execute(sql,(name,email,password,age,gender,phone))
        db.commit()

        return redirect("/login")

    return render_template("register.html")


# =============================
# USER LOGIN
# =============================

@app.route("/login", methods=["GET","POST"])
def login():

    if request.method=="POST":

        email=request.form["email"]
        password=request.form["password"]

        cursor.execute(
            "SELECT * FROM users WHERE email=%s AND password=%s",
            (email,password)
        )

        user=cursor.fetchone()

        if user:
            session["user_id"]=user[0]
            return redirect("/dashboard")

    return render_template("login.html")


# =============================
# ADMIN LOGIN
# =============================

@app.route("/admin", methods=["GET","POST"])
def admin():

    if request.method=="POST":

        username=request.form["username"]
        password=request.form["password"]

        if username=="admin" and password=="admin":
            session["admin"]=True
            return redirect("/admin_dashboard")

    return render_template("admin_login.html")


# =============================
# USER DASHBOARD
# =============================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect("/login")

    return render_template("dashboard.html")


# =============================
# IMAGE UPLOAD + PREDICTION
# =============================

@app.route("/upload", methods=["GET","POST"])
def upload():

    if "user_id" not in session:
        return redirect("/login")

    if request.method=="POST":

        file=request.files["image"]

        filename=file.filename

        filepath=os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        img=cv2.imread(filepath)
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        img_array=np.expand_dims(img, axis=0)
        img_array=preprocess_input(img_array)

        prediction=model.predict(img_array)[0][0]

        if prediction>0.5:
            result="Malignant"
            confidence=float(prediction)
        else:
            result="Benign"
            confidence=float(1-prediction)

        cursor.execute(
            "INSERT INTO predictions(user_id,image_path,result,confidence) VALUES(%s,%s,%s,%s)",
            (session["user_id"], filepath, result, confidence)
        )

        db.commit()

        return render_template(
            "result.html",
            image=filepath,
            result=result,
            confidence=round(confidence*100,2)
        )

    return render_template("upload.html")


# =============================
# GRAD-CAM VISUALIZATION
# =============================

@app.route("/gradcam")
def gradcam():

    img_path = request.args.get("img")

    img = cv2.imread(img_path)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        label = "Malignant"
        confidence = float(pred)

        risk_score = confidence * 100

        # Risk levels only for malignant cases
        if risk_score < 60:
            risk_level = "Moderate Risk"
            risk_color = "orange"

        elif risk_score < 80:
            risk_level = "High Risk"
            risk_color = "red"

        else:
            risk_level = "Critical Risk"
            risk_color = "darkred"

    else:
        label = "Benign"
        confidence = float(1 - pred)

        risk_score = None
        risk_level = None
        risk_color = None

    # GradCAM model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:,0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:,:,i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap,0)
    heatmap = heatmap/(np.max(heatmap)+1e-8)

    heatmap = cv2.resize(heatmap,(IMG_SIZE,IMG_SIZE))

    heatmap_uint8 = np.uint8(255*heatmap)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img,0.6,heatmap_color,0.4,0)

    # Detect suspicious regions
    thresh = cv2.threshold(heatmap_uint8,180,255,cv2.THRESH_BINARY)[1]

    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    region_count = 0

    for c in contours:

        x,y,w,h = cv2.boundingRect(c)

        if w*h > 500:
            cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,0),2)
            region_count += 1

    gradcam_path = os.path.join(GRADCAM_FOLDER,"gradcam.jpg")

    cv2.imwrite(gradcam_path,overlay)

    # =========================
    # XAI + MEDICAL EXPLANATION
    # =========================

    if label == "Malignant":

        treatment = """
        The model detected patterns associated with malignant breast tissue.
        This may indicate the presence of cancerous cells.
        Treatment options typically include surgery, chemotherapy,
        radiation therapy, targeted therapy, or hormone therapy.
        Early consultation with an oncologist is strongly recommended.
        """

        suggestions = """
        • Consult a certified oncologist immediately.
        • Perform additional diagnostic tests such as biopsy or MRI.
        • Follow medical advice regarding treatment plans.
        • Maintain a healthy diet and avoid smoking or alcohol.
        """

    else:

        treatment = """
        The model prediction indicates benign breast tissue.
        Benign conditions are non-cancerous and usually not life-threatening.
        However, regular screening and medical check-ups are recommended.
        """

        suggestions = """
        • Continue regular breast health screening.
        • Maintain a healthy lifestyle.
        • Consult a doctor if unusual symptoms appear.
        """

    xai_explanation = f"""
    The AI model analyzed the uploaded breast image using a Convolutional Neural Network.
    Grad-CAM visualization highlights the regions that influenced the prediction.
    Red regions indicate strong activation where the model detected important
    tissue patterns related to breast abnormalities.

    The system detected {region_count} suspicious region(s) contributing to the prediction.
    """

    return render_template(
        "gradcam.html",
        original=img_path,
        heatmap=gradcam_path,
        result=label,
        confidence=round(confidence*100,2),
        regions=region_count,
        explanation=xai_explanation,
        treatment=treatment,
        suggestions=suggestions,
        risk_level=risk_level,
        risk_color=risk_color,
        risk_score=risk_score
    )


from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

@app.route("/download_report")
def download_report():

    original = request.args.get("original")
    heatmap = request.args.get("heatmap")
    result = request.args.get("result")
    confidence = request.args.get("confidence")
    explanation = request.args.get("explanation")
    treatment = request.args.get("treatment")
    suggestions = request.args.get("suggestions")
    risk_level = request.args.get("risk_level")

    pdf_path = "static/report/breast_cancer_report.pdf"

    os.makedirs("static/report", exist_ok=True)

    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("AI Breast Cancer Diagnostic Report", styles['Title']))
    story.append(Spacer(1,20))

    story.append(Paragraph(f"<b>Prediction Result:</b> {result}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence Level:</b> {confidence}%", styles['Normal']))

    if risk_level:
        story.append(Paragraph(f"<b>Risk Level:</b> {risk_level}", styles['Normal']))

    story.append(Spacer(1,20))

    story.append(Paragraph("<b>Original Image</b>", styles['Heading3']))
    story.append(Image(original, width=300, height=300))

    story.append(Spacer(1,20))

    story.append(Paragraph("<b>Grad-CAM Visualization</b>", styles['Heading3']))
    story.append(Image(heatmap, width=300, height=300))

    story.append(Spacer(1,20))

    story.append(Paragraph("<b>Explainable AI Interpretation</b>", styles['Heading3']))
    story.append(Paragraph(explanation, styles['Normal']))

    story.append(Spacer(1,15))

    story.append(Paragraph("<b>Medical Interpretation</b>", styles['Heading3']))
    story.append(Paragraph(treatment, styles['Normal']))

    story.append(Spacer(1,15))

    story.append(Paragraph("<b>Suggestions & Recommendations</b>", styles['Heading3']))
    story.append(Paragraph(suggestions, styles['Normal']))

    story.append(Spacer(1,20))

    story.append(Paragraph(
        "Disclaimer: This AI-generated report is for educational purposes only and should not replace professional medical diagnosis.",
        styles['Italic']
    ))

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)

    doc.build(story)

    return redirect("/static/report/breast_cancer_report.pdf")
# =============================
# USER HISTORY
# =============================

@app.route("/history")
def history():

    cursor.execute(
        "SELECT * FROM predictions WHERE user_id=%s",
        (session["user_id"],)
    )

    data=cursor.fetchall()

    return render_template("history.html", data=data)


# =============================
# ADMIN DASHBOARD
# =============================

@app.route("/admin_dashboard")
def admin_dashboard():

    if "admin" not in session:
        return redirect("/admin")

    # total users
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    # total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    # benign count
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result='Benign'")
    benign_count = cursor.fetchone()[0]

    # malignant count
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result='Malignant'")
    malignant_count = cursor.fetchone()[0]

    # latest predictions
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 10")
    predictions = cursor.fetchall()

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_predictions=total_predictions,
        benign_count=benign_count,
        malignant_count=malignant_count,
        predictions=predictions
    )

# =============================
# LOGOUT
# =============================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# =============================
# RUN APP
# =============================

if __name__=="__main__":
    app.run(debug=True)