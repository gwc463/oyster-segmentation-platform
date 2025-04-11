from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import os
import io
import base64
import random
import string
from PIL import Image, ImageDraw, ImageFont
import uuid

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/avatars/*": {"origins": "*"}})

# MySQL 配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:032466@localhost/oyster_segmentation_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 确保 uploads 和 avatars 目录存在
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists('avatars'):
    os.makedirs('avatars')

# 存储验证码的字典（模拟会话存储）
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

# 验证码生成接口
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

# 验证码验证接口
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

# 上传头像接口
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

    return jsonify({"avatar_url": f"http://localhost:8081/avatars/{filename}"}), 200

# 获取当前用户信息接口
@app.route('/api/users/me', methods=['GET'])
def get_user_info():
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID required"}), 401
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.to_dict()), 200

# 更新用户信息接口
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
    return jsonify(user.to_dict()), 200

# 提供头像文件访问
@app.route('/avatars/<filename>')
def serve_avatar(filename):
    return send_file(os.path.join('avatars', filename))

# 其他接口
@app.route('/test', methods=['GET'])
def test():
    return "Hello, Oyster!"

@app.route('/api/users/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    if user and user.password == data['password']:
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
    return jsonify(new_user.to_dict()), 201

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
    return jsonify(project.to_dict()), 201

@app.route('/api/projects/<int:id>', methods=['GET'])
def get_project(id):
    project = Project.query.get_or_404(id)
    datasets = Dataset.query.filter_by(project_id=id).all()
    return jsonify({
        "project": project.to_dict(),
        "datasets": [d.to_dict() for d in datasets]
    })

@app.route('/api/projects/<int:id>', methods=['PUT'])
def update_project(id):
    project = Project.query.get_or_404(id)
    data = request.get_json()
    project.name = data.get('name', project.name)
    project.creator = data.get('creator', project.creator)
    project.description = data.get('description', project.description)
    db.session.commit()
    return jsonify(project.to_dict()), 200

@app.route('/api/projects/<int:id>', methods=['DELETE'])
def delete_project(id):
    project = Project.query.get_or_404(id)
    db.session.delete(project)
    db.session.commit()
    return '', 204

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
    return jsonify(dataset.to_dict()), 200

@app.route('/api/datasets/<int:id>', methods=['DELETE'])
def delete_dataset(id):
    dataset = Dataset.query.get_or_404(id)
    db.session.delete(dataset)
    db.session.commit()
    return '', 204

@app.route('/api/predict/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    filename = file.filename
    file.save(f"uploads/{filename}")
    return jsonify({"message": "Image uploaded", "filename": filename}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    return jsonify({"gender": "雌性", "grayValue": 120})

@app.route('/api/query/<project_id>', methods=['GET'])
def query(project_id):
    project = Project.query.get_or_404(project_id)
    return jsonify({"result": f"项目ID: {project_id} 的分割结果: {project.name} 已分割，性别: 雌性"})

# 初始化数据库
def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username="admin").first():
            db.session.add(User(username="admin", password="123456", email="admin@example.com"))
            db.session.commit()

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8081, debug=True)