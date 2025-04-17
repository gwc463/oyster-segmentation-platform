from flask import Flask, request, jsonify, send_file
from sqlalchemy.dialects import mysql
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime, timezone
import os
import io
import base64
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import uuid
import numpy as np
import logging

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/avatars/*": {"origins": "*"}, r"/masks/*": {"origins": "*"}})

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MySQL 配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:032466@localhost/oyster_segmentation_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 确保目录存在
for directory in ['uploads', 'avatars', 'models', 'masks']:
    os.makedirs(directory, exist_ok=True)

# 存储验证码
captcha_store = {}

# 模型定义
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    avatar_url = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    def to_dict(self):
        avatar_url = f"http://localhost:8081{self.avatar_url}" if self.avatar_url else None
        return {
            "id": self.id,
            "username": self.username,
            "password": self.password,
            "email": self.email,
            "avatar_url": avatar_url,
            "createdAt": self.created_at.isoformat()
        }

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    creator = db.Column(db.String(80), nullable=False)
    description = db.Column(db.Text)
    def to_dict(self):
        return { "id": self.id, "name": self.name, "creator": self.creator, "description": self.description }

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    size = db.Column(db.Integer, nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=True)
    project = db.relationship('Project', backref=db.backref('datasets', lazy=True))
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "size": self.size,
            "project_id": self.project_id,
            "project_name": self.project.name if self.project else None
        }

class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(50), nullable=False, default="未训练")
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=True)
    learning_rate = db.Column(db.Float, nullable=True)
    epochs = db.Column(db.Integer, nullable=True)
    accuracy = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    version = db.Column(db.Integer, nullable=True, default=1)
    file_path = db.Column(db.String(255), nullable=True)
    progress = db.Column(db.Integer, nullable=True, default=0)
    project = db.relationship('Project', backref=db.backref('models', lazy=True))
    dataset = db.relationship('Dataset', backref=db.backref('models', lazy=True))
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "status": self.status,
            "dataset_id": self.dataset_id,
            "project_id": self.project_id,
            "project_name": self.project.name if self.project else None,
            "dataset_name": self.dataset.name if self.dataset else None,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "version": self.version,
            "file_path": self.file_path,
            "progress": self.progress
        }

class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('model.id', ondelete='CASCADE'), nullable=False)
    gender = db.Column(db.String(50), nullable=False)
    gray_value = db.Column(db.Integer, nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    mask_path = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, server_default=db.func.now())
    __table_args__ = (
        db.Index('idx_user_created', 'user_id', 'created_at'),
        db.Index('idx_confidence', 'confidence'),
        db.Index('idx_gender', 'gender'),
    )

    def to_dict(self):
        mask_url = f"http://localhost:8081/masks/{os.path.basename(self.mask_path)}" if self.mask_path else None
        created_at_str = self.created_at.isoformat() if self.created_at else None
        return {
            "id": self.id,
            "filename": self.filename,
            "model_id": self.model_id,
            "gender": self.gender,
            "gray_value": self.gray_value,
            "confidence": self.confidence,
            "mask_url": mask_url,
            "created_at": created_at_str
        }

# 生成验证码图片
def generate_captcha_image(code):
    width, height = 100, 40
    image = Image.new('RGB', (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    for _ in range(5):
        draw.line(
            [(random.randint(0, width), random.randint(0, height)),
             (random.randint(0, width), random.randint(0, height))],
            fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            width=1
        )
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    draw.text((10, 5), code, font=font, fill=(50, 50, 50))
    for _ in range(50):
        draw.point(
            (random.randint(0, width), random.randint(0, height)),
            fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        )
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# 生成模拟分割掩码（优化版）
def generate_dummy_mask(width=200, height=200):
    mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    points = [
        (random.randint(50, 150), random.randint(50, 150)),
        (random.randint(50, 150), random.randint(50, 150)),
        (random.randint(50, 150), random.randint(50, 150)),
        (random.randint(50, 150), random.randint(50, 150))
    ]
    draw.polygon(points, fill=(255, 0, 0, 128))
    mask = mask.filter(ImageFilter.SMOOTH)
    mask_filename = f"{uuid.uuid4()}.png"
    mask_path = os.path.join('masks', mask_filename)
    try:
        mask.save(mask_path)
        logger.info(f"Generated mask: {mask_path}")
    except Exception as e:
        logger.error(f"Failed to save mask: {e}")
        raise
    return mask_path

# 验证码接口
@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    chars = string.ascii_letters + string.digits
    code = ''.join(random.choice(chars) for _ in range(4))
    img_base64 = generate_captcha_image(code)
    captcha_id = str(uuid.uuid4())
    captcha_store[captcha_id] = code.lower()
    return jsonify({
        "captchaId": captcha_id,
        "captchaImage": f"data:image/png;base64,{img_base64}"
    })

@app.route('/api/verify-captcha', methods=['POST'])
def verify_captcha():
    data = request.get_json()
    captcha_id = data.get('captchaId')
    user_input = data.get('captchaInput')
    if not captcha_id or not user_input:
        return jsonify({"error": "Missing captchaId or captchaInput"}), 400
    if captcha_id not in captcha_store:
        return jsonify({"error": "Invalid captchaId"}), 400
    if captcha_store[captcha_id] != user_input.lower():
        return jsonify({"error": "Invalid captcha"}), 400
    del captcha_store[captcha_id]
    return jsonify({"message": "Captcha verified"}), 200

# 用户相关接口
@app.route('/api/users/upload-avatar', methods=['POST'])
def upload_avatar():
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID required"}), 401
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    if 'avatar' not in request.files:
        return jsonify({"error": "No avatar file uploaded"}), 400
    file = request.files['avatar']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    filename = f"{user_id}_{uuid.uuid4()}.png"
    file_path = os.path.join('avatars', filename)
    file.save(file_path)
    user.avatar_url = f"/avatars/{filename}"
    db.session.commit()
    logger.info(f"Uploaded avatar for user {user_id}: {file_path}")
    return jsonify({"avatar_url": f"http://localhost:8081/avatars/{filename}"}), 200

@app.route('/api/users/me', methods=['GET'])
def get_user_info():
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID required"}), 401
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.to_dict()), 200

@app.route('/api/users/update', methods=['PUT'])
def update_user_info():
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID required"}), 401
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    data = request.get_json()
    new_username = data.get('username', user.username)
    new_email = data.get('email', user.email)
    new_password = data.get('password', user.password)
    if new_username != user.username and User.query.filter_by(username=new_username).first():
        return jsonify({"error": "Username already exists"}), 400
    if new_email != user.email and User.query.filter_by(email=new_email).first():
        return jsonify({"error": "Email already exists"}), 400
    user.username = new_username
    user.email = new_email
    user.password = new_password
    db.session.commit()
    logger.info(f"Updated user info for user {user_id}")
    return jsonify(user.to_dict()), 200

@app.route('/avatars/<filename>')
def serve_avatar(filename):
    file_path = os.path.join('avatars', filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({"error": "Avatar not found"}), 404

@app.route('/masks/<filename>')
def serve_mask(filename):
    file_path = os.path.join('masks', filename)
    if os.path.exists(file_path):
        response = send_file(file_path, mimetype='image/png')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response
    logger.error(f"Mask file not found: {file_path}")
    return jsonify({"error": "Mask not found"}), 404

# 登录注册
@app.route('/api/users/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    if user and user.password == data['password']:
        logger.info(f"User {user.username} logged in")
        return jsonify(user.to_dict()), 200
    return jsonify({"error": "Invalid username or password"}), 401

@app.route('/api/users/register', methods=['POST'])
def register():
    data = request.get_json()
    if not all(key in data for key in ['username', 'password', 'email']):
        return jsonify({"error": "Missing required fields"}), 400
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"error": "Username already exists"}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"error": "Email already exists"}), 400
    new_user = User(username=data['username'], password=data['password'], email=data['email'])
    db.session.add(new_user)
    db.session.commit()
    logger.info(f"Registered new user: {new_user.username}")
    return jsonify(new_user.to_dict()), 201

# 项目管理
@app.route('/api/projects', methods=['GET'])
def get_projects():
    projects = Project.query.all()
    return jsonify([p.to_dict() for p in projects])

@app.route('/api/projects', methods=['POST'])
def add_project():
    data = request.get_json()
    project = Project(name=data['name'], creator=data['creator'], description=data['description'])
    db.session.add(project)
    db.session.commit()
    logger.info(f"Added project: {project.name}")
    return jsonify(project.to_dict()), 201

@app.route('/api/projects/<int:id>', methods=['GET'])
def get_project(id):
    project = Project.query.get_or_404(id)
    datasets = Dataset.query.filter_by(project_id=id).all()
    models = Model.query.filter_by(project_id=id).all()
    return jsonify({
        "project": project.to_dict(),
        "datasets": [d.to_dict() for d in datasets],
        "models": [m.to_dict() for m in models]
    })

@app.route('/api/projects/<int:id>', methods=['PUT'])
def update_project(id):
    project = Project.query.get_or_404(id)
    data = request.get_json()
    project.name = data.get('name', project.name)
    project.creator = data.get('creator', project.creator)
    project.description = data.get('description', project.description)
    db.session.commit()
    logger.info(f"Updated project {id}")
    return jsonify(project.to_dict()), 200

@app.route('/api/projects/<int:id>', methods=['DELETE'])
def delete_project(id):
    project = Project.query.get_or_404(id)
    db.session.delete(project)
    db.session.commit()
    logger.info(f"Deleted project {id}")
    return '', 204

# 数据集管理
@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    datasets = Dataset.query.all()
    return jsonify([d.to_dict() for d in datasets])

@app.route('/api/datasets', methods=['POST'])
def add_dataset():
    data = request.get_json()
    dataset = Dataset(
        name=data['name'],
        type=data['type'],
        size=data['size'],
        project_id=data.get('project_id')
    )
    db.session.add(dataset)
    db.session.commit()
    logger.info(f"Added dataset: {dataset.name}")
    return jsonify(dataset.to_dict()), 201

@app.route('/api/datasets/<int:id>', methods=['GET'])
def get_dataset(id):
    dataset = Dataset.query.get_or_404(id)
    return jsonify(dataset.to_dict())

@app.route('/api/datasets/<int:id>', methods=['PUT'])
def update_dataset(id):
    dataset = Dataset.query.get_or_404(id)
    data = request.get_json()
    dataset.name = data.get('name', dataset.name)
    dataset.type = data.get('type', dataset.type)
    dataset.size = data.get('size', dataset.size)
    dataset.project_id = data.get('project_id', dataset.project_id)
    db.session.commit()
    logger.info(f"Updated dataset {id}")
    return jsonify(dataset.to_dict()), 200

@app.route('/api/datasets/<int:id>', methods=['DELETE'])
def delete_dataset(id):
    dataset = Dataset.query.get_or_404(id)
    db.session.delete(dataset)
    db.session.commit()
    logger.info(f"Deleted dataset {id}")
    return '', 204

# 模型管理
@app.route('/api/models', methods=['GET'])
def get_models():
    models = Model.query.all()
    return jsonify([m.to_dict() for m in models])

@app.route('/api/models/train', methods=['POST'])
def train_model():
    data = request.get_json()
    base_name = data['name']
    existing_models = Model.query.filter(Model.name.like(f"{base_name}%")).all()
    version = 1
    if existing_models:
        versions = [m.version for m in existing_models if m.version]
        version = max(versions) + 1 if versions else 1
    model_name = f"{base_name}-v{version}" if version > 1 else base_name

    file_path = f"models/{model_name}.pth"
    with open(file_path, 'w') as f:
        f.write("This is a dummy model file.")

    new_model = Model(
        name=model_name,
        type=data['type'],
        status="训练中",
        dataset_id=data['dataset_id'],
        project_id=data.get('project_id'),
        learning_rate=data['learning_rate'],
        epochs=data['epochs'],
        version=version,
        file_path=file_path,
        progress=0
    )
    db.session.add(new_model)
    db.session.commit()
    logger.info(f"Started training model: {model_name}")
    return jsonify(new_model.to_dict()), 201

@app.route('/api/models/<int:id>/progress', methods=['GET'])
def get_training_progress(id):
    model = Model.query.get_or_404(id)
    if model.status == "已训练":
        progress = 100
    else:
        progress = model.progress + 10
        if progress >= 100:
            model.status = "已训练"
            model.accuracy = random.uniform(0.8, 0.95)
            model.f1_score = random.uniform(0.75, 0.9)
            progress = 100
        model.progress = progress
        db.session.commit()
    logger.info(f"Model {id} progress: {progress}%")
    return jsonify({"progress": progress, "status": model.status})

@app.route('/api/models/evaluate/<int:id>', methods=['POST'])
def evaluate_model(id):
    model = Model.query.get_or_404(id)
    data = request.get_json()
    dataset_id = data.get('dataset_id')
    random.seed(dataset_id + id)
    tp = random.randint(40, 60)
    fp = random.randint(0, 10)
    fn = random.randint(0, 10)
    tn = random.randint(40, 60)
    confusion_matrix = [[tp, fp], [fn, tn]]
    accuracy = random.uniform(0.8, 0.95)
    f1_score = random.uniform(0.75, 0.9)
    model.accuracy = accuracy
    model.f1_score = f1_score
    db.session.commit()
    logger.info(f"Evaluated model {id} with dataset {dataset_id}, accuracy: {accuracy:.2f}")
    return jsonify({
        "model_id": model.id,
        "dataset_id": dataset_id,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "confusion_matrix": confusion_matrix
    })

@app.route('/api/models/<int:id>', methods=['DELETE'])
def delete_model(id):
    model = Model.query.get_or_404(id)
    if model.file_path and os.path.exists(model.file_path):
        os.remove(model.file_path)
    db.session.delete(model)
    db.session.commit()
    logger.info(f"Deleted model {id}")
    return '', 204

@app.route('/api/models/download/<int:id>', methods=['GET'])
def download_model(id):
    model = Model.query.get_or_404(id)
    if not model.file_path or not os.path.exists(model.file_path):
        return jsonify({"error": "Model file not found"}), 404
    logger.info(f"Downloaded model {id}")
    return send_file(model.file_path, as_attachment=True, download_name=f"{model.name}.pth")

# 预测相关接口
@app.route('/api/predict/upload', methods=['POST'])
def upload_image():
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID required"}), 401
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    filename = f"{user_id}_{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join('uploads', filename)
    file.save(file_path)
    logger.info(f"Uploaded image: {file_path}")
    return jsonify({"message": "Image uploaded", "filename": filename}), 200

@app.route('/api/predict/upload-batch', methods=['POST'])
def upload_batch_images():
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID required"}), 401
    if 'images' not in request.files:
        return jsonify({"error": "No images uploaded"}), 400
    files = request.files.getlist('images')
    filenames = []
    for file in files:
        if file.filename:
            filename = f"{user_id}_{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            filenames.append(filename)
            logger.info(f"Uploaded batch image: {file_path}")
    return jsonify({"filenames": filenames}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID required"}), 401
    filename = data.get('filename')
    model_id = data.get('model_id')
    if not filename or not model_id:
        return jsonify({"error": "Missing filename or model_id"}), 400
    file_path = os.path.join('uploads', filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Image file not found"}), 404
    model = db.session.get(Model, model_id)
    if not model:
        return jsonify({"error": "Model not found"}), 404
    random.seed(model_id)
    gender = random.choice(["雌性", "雄性"])
    gray_value = random.randint(100, 150)
    confidence = random.uniform(0.8, 0.99)
    mask_path = generate_dummy_mask()
    created_at = datetime.utcnow()
    prediction = PredictionResult(
        user_id=user_id,
        filename=filename,
        model_id=model_id,
        gender=gender,
        gray_value=gray_value,
        confidence=confidence,
        mask_path=mask_path,
        created_at=created_at
    )
    db.session.add(prediction)
    db.session.commit()
    result = prediction.to_dict()
    logger.info(f"Prediction result: {result}")
    return jsonify(result)

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID required"}), 401
    filenames = data.get('filenames', [])
    model_id = data.get('model_id')
    if not filenames or not model_id:
        return jsonify({"error": "Missing filenames or model_id"}), 400
    model = Model.query.get(model_id)
    if not model:
        return jsonify({"error": "Model not found"}), 404
    results = []
    random.seed(model_id)
    predictions = []
    for filename in filenames:
        file_path = os.path.join('uploads', filename)
        if not os.path.exists(file_path):
            logger.warning(f"Image file not found: {file_path}")
            continue
        gender = random.choice(["雌性", "雄性"])
        gray_value = random.randint(100, 150)
        confidence = random.uniform(0.8, 0.99)
        mask_path = generate_dummy_mask()
        created_at = datetime.utcnow()
        prediction = PredictionResult(
            user_id=user_id,
            filename=filename,
            model_id=model_id,
            gender=gender,
            gray_value=gray_value,
            confidence=confidence,
            mask_path=mask_path,
            created_at=created_at
        )
        predictions.append(prediction)
        results.append(prediction.to_dict())
    try:
        db.session.add_all(predictions)
        db.session.commit()
        logger.info(f"Batch prediction for user {user_id}, model {model_id}, files: {filenames}")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Batch prediction failed: {e}")
        return jsonify({"error": "Failed to save predictions"}), 500
    return jsonify({"results": results})

@app.route('/api/predict/compare', methods=['POST'])
def predict_compare():
    data = request.get_json()
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID required"}), 401
    filename = data.get('filename')
    model_ids = data.get('model_ids', [])
    if not filename or not model_ids:
        return jsonify({"error": "Missing filename or model_ids"}), 400
    file_path = os.path.join('uploads', filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Image file not found"}), 404
    results = []
    predictions = []
    for model_id in model_ids:
        model = db.session.get(Model, model_id)
        if model:
            random.seed(model_id)
            gender = random.choice(["雌性", "雄性"])
            gray_value = random.randint(100, 150)
            confidence = random.uniform(0.8, 0.99)
            mask_path = generate_dummy_mask()
            created_at = datetime.now(timezone.utc)
            prediction = PredictionResult(
                user_id=user_id,
                filename=filename,
                model_id=model_id,
                gender=gender,
                gray_value=gray_value,
                confidence=confidence,
                mask_path=mask_path,
                created_at=created_at
            )
            predictions.append(prediction)
            db.session.add(prediction)
    try:
        db.session.commit()
        for prediction in predictions:
            result = {
                **prediction.to_dict(),
                "model_name": db.session.get(Model, prediction.model_id).name
            }
            results.append(result)
            logger.info(f"Prediction for model {prediction.model_id}, file {filename}: {result}")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Compare prediction failed: {e}")
        return jsonify({"error": "Failed to save predictions"}), 500
    logger.info(f"Compare prediction for user {user_id}, file {filename}, models: {model_ids}")
    return jsonify({"results": results})

@app.route('/api/predict/history', methods=['GET'])
def prediction_history():
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID required"}), 401

    # 获取查询参数
    search_query = request.args.get('search_query')
    gender = request.args.get('gender')
    confidence_min = request.args.get('confidence_min', type=float)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    sort_by = request.args.get('sort_by', 'created_at-desc')

    # 构建查询
    query = PredictionResult.query.filter_by(user_id=user_id)

    # 应用过滤条件
    if search_query:
        query = query.filter(PredictionResult.filename.ilike(f'%{search_query}%'))
    if gender:
        query = query.filter(PredictionResult.gender == gender)
    if confidence_min is not None:
        query = query.filter(PredictionResult.confidence >= confidence_min)
    if start_date:
        try:
            query = query.filter(PredictionResult.created_at >= datetime.fromisoformat(start_date))
        except ValueError:
            return jsonify({"error": "Invalid start_date format"}), 400
    if end_date:
        try:
            query = query.filter(PredictionResult.created_at <= datetime.fromisoformat(end_date))
        except ValueError:
            return jsonify({"error": "Invalid end_date format"}), 400

    # 应用排序
    if sort_by == 'created_at-asc':
        query = query.order_by(PredictionResult.created_at.asc())
    elif sort_by == 'confidence-desc':
        query = query.order_by(PredictionResult.confidence.desc())
    elif sort_by == 'gray_value-asc':
        query = query.order_by(PredictionResult.gray_value.asc())
    else:
        query = query.order_by(PredictionResult.created_at.desc())

    # 执行查询
    try:
        predictions = query.all()
        results = [{
            **p.to_dict(),
            'model_name': db.session.get(Model, p.model_id).name if db.session.get(Model, p.model_id) else '未知'
        } for p in predictions]
        logger.info(f"Fetched {len(results)} history records for user {user_id}")
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"History query failed: {e}")
        return jsonify({"error": "Failed to fetch history"}), 500

@app.route('/api/query', methods=['GET'])
def query():
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID required"}), 401

    project_id = request.args.get('project_id')
    project_name = request.args.get('project_name')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # 构建基础查询，关联 Project、Model 和 PredictionResult
    query = db.session.query(
        Project.id.label('project_id'),
        Project.name.label('project_name'),
        Project.creator,
        PredictionResult.gender,
        PredictionResult.confidence,
        PredictionResult.created_at
    ).outerjoin(
        Model,
        Project.id == Model.project_id
    ).outerjoin(
        PredictionResult,
        Model.id == PredictionResult.model_id
    ).filter(PredictionResult.user_id == user_id)

    # 应用过滤条件
    if project_id:
        query = query.filter(Project.id == project_id)
    if project_name:
        query = query.filter(Project.name.ilike(f'%{project_name}%'))
    if start_date:
        try:
            query = query.filter(PredictionResult.created_at >= datetime.fromisoformat(start_date))
        except ValueError:
            return jsonify({"error": "Invalid start_date format"}), 400
    if end_date:
        try:
            query = query.filter(PredictionResult.created_at <= datetime.fromisoformat(end_date))
        except ValueError:
            return jsonify({"error": "Invalid end_date format"}), 400

    # 打印 SQL 查询以便调试
    compiled_query = query.statement.compile(dialect=mysql.dialect(), compile_kwargs={"literal_binds": True})
    logger.info(f"Executing SQL: {compiled_query}")

    try:
        results = query.all()
        formatted_results = [{
            'project_id': r.project_id,
            'project_name': r.project_name,
            'creator': r.creator,
            'gender': r.gender,
            'confidence': r.confidence,
            'created_at': r.created_at.isoformat() if r.created_at else None
        } for r in results]
        logger.info(f"Fetched {len(formatted_results)} query results for user {user_id}")
        return jsonify({"results": formatted_results})
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return jsonify({"error": "Failed to fetch query results"}), 500

@app.route('/api/models/<int:id>', methods=['PUT'])
def update_model(id):
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID required"}), 401
    
    model = Model.query.get_or_404(id)
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # 验证必填字段
    name = data.get('name', model.name)
    if not name:
        return jsonify({"error": "Model name is required"}), 400
    
    # 验证 project_id
    project_id = data.get('project_id', model.project_id)
    if project_id is not None:
        project = Project.query.get(project_id)
        if not project:
            return jsonify({"error": "Invalid project_id"}), 400
    
    # 验证 dataset_id
    dataset_id = data.get('dataset_id', model.dataset_id)
    if dataset_id is not None:
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            jsonify({"error": "Invalid dataset_id"}), 400
    
    # 更新字段
    model.name = name
    model.type = data.get('type', model.type)
    model.project_id = project_id
    model.dataset_id = dataset_id
    model.learning_rate = data.get('learning_rate', model.learning_rate)
    model.epochs = data.get('epochs', model.epochs)
    model.version = data.get('version', model.version)
    
    try:
        db.session.commit()
        logger.info(f"Updated model {id} with data: {data}")
        return jsonify(model.to_dict()), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to update model {id}: {e}")
        return jsonify({"error": "Failed to update model"}), 500
# 初始化数据库
def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username="admin").first():
            db.session.add(User(username="admin", password="123456", email="admin@example.com"))
            db.session.commit()
        logger.info("Database initialized")

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8081, debug=True)