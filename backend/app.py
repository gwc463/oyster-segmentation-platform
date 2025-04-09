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
CORS(app)  # 支持跨域请求

# MySQL 配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:032466@localhost/oyster_segmentation_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 确保 uploads 目录存在
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# 存储验证码的字典（模拟会话存储）
captcha_store = {}

# 模型定义
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    def to_dict(self):
        return { "id": self.id, "username": self.username, "password": self.password, "email": self.email, "createdAt": self.created_at.isoformat() }

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
    def to_dict(self):
        return { "id": self.id, "name": self.name, "type": self.type, "size": self.size }

# 生成验证码图片
def generate_captcha_image(code):
    width, height = 100, 40
    image = Image.new('RGB', (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(image)

    # 绘制干扰线
    for _ in range(5):
        draw.line(
            [(random.randint(0, width), random.randint(0, height)),
             (random.randint(0, width), random.randint(0, height))],
            fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            width=1
        )

    # 绘制验证码文字
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    draw.text((10, 5), code, font=font, fill=(50, 50, 50))

    # 绘制干扰点
    for _ in range(50):
        draw.point(
            (random.randint(0, width), random.randint(0, height)),
            fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        )

    # 将图片转为 base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# 验证码生成接口
@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    # 生成随机验证码（4位字母和数字）
    chars = string.ascii_letters + string.digits
    code = ''.join(random.choice(chars) for _ in range(4))
    
    # 生成验证码图片
    img_base64 = generate_captcha_image(code)
    
    # 生成验证码 ID 并存储
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

    # 验证通过后删除验证码
    del captcha_store[captcha_id]
    return jsonify({"message": "Captcha verified"}), 200

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
    dataset = Dataset(name=data['name'], type=data['type'], size=data['size'])
    db.session.add(dataset)
    db.session.commit()
    return jsonify(dataset.to_dict()), 201

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