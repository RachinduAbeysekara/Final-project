from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pickle
import pandas as pd

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/lung_cancer'
db = SQLAlchemy(app)

class User(db.Model):
    U_ID = db.Column(db.String(255), primary_key=True)
    U_Name = db.Column(db.String(255), nullable=False)
    U_Address = db.Column(db.String(255))
    U_ContactNumber = db.Column(db.Integer)
    U_Email = db.Column(db.String(255), unique=True, nullable=False)
    U_Username = db.Column(db.String(255), unique=True, nullable=False)
    U_Password = db.Column(db.String(255), nullable=False)
    U_Dob = db.Column(db.Date)

    def serialize(self):
        return {
            'U_ID': self.U_ID,
            'U_Name': self.U_Name,
            'U_Address': self.U_Address,
            'U_ContactNumber': self.U_ContactNumber,
            'U_Email': self.U_Email,
            'U_Username': self.U_Username,
            'U_Password': self.U_Password,
            'U_Dob': self.U_Dob.strftime('%Y-%m-%d') if self.U_Dob else None
        }
class Task(db.Model):
    T_ID = db.Column(db.String(255), primary_key=True)
    T_Description = db.Column(db.String(255))
    T_Date = db.Column(db.Date)
    T_Time = db.Column(db.Time)
    T_Userinput = db.Column(db.String(255))
    T_Result = db.Column(db.String(255))
    U_ID = db.Column(db.String(255), db.ForeignKey('user.U_ID'))

    def serialize(self):
        return {
            'T_ID': self.T_ID,
            'T_Description': self.T_Description,
            'T_Date': self.T_Date.isoformat() if self.T_Date else None,
            'T_Time': self.T_Time.isoformat() if self.T_Time else None,
            'T_Userinput': self.T_Userinput,
            'T_Result': self.T_Result,
            'U_ID': self.U_ID
        }

@app.route('/', methods=['GET'])
def rootRoutes():
    return jsonify({
        "message": "Lung Cancer Server On Going!"
    })


# Load the trained model
with open('./model/lung_cancer_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# ================================================================================================================= USER
@app.route('/users', methods=['POST'])
def create_user():
    data = request.json
    username = data.get('U_Username')
    email = data.get('U_Email')
    user_id = data.get('U_ID')

    existing_user = User.query.filter_by(U_ID=user_id).first()
    existing_user_username = User.query.filter_by(U_Username=username).first()
    existing_user_email = User.query.filter_by(U_Email=email).first()

    if existing_user:
        return jsonify({'message': 'User with the same ID already exists'}), 400
    if existing_user_username:
        return jsonify({'message': 'Username already exists'}), 400
    if existing_user_email:
        return jsonify({'message': 'Email already exists'}), 400

    try:
        new_user = User(**data)
        db.session.add(new_user)
        db.session.commit()
        return jsonify(new_user.serialize()), 201
    except IntegrityError:
        db.session.rollback()
        return jsonify({'message': 'IntegrityError: Unable to create user due to duplicate values'}), 400

@app.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([user.serialize() for user in users])

@app.route('/users/<id>', methods=['GET'])
def get_user(id):
    user = User.query.get(id)
    if user:
        return jsonify(user.serialize())
    return jsonify({'message': 'User not found'}), 404

@app.route('/users/<id>', methods=['PUT'])
def update_user(id):
    user = User.query.get(id)
    if user:
        data = request.json
        for key, value in data.items():
            setattr(user, key, value)
        db.session.commit()
        return jsonify(user.serialize())
    return jsonify({'message': 'User not found'}), 404

@app.route('/users/<id>', methods=['DELETE'])
def delete_user(id):
    user = User.query.get(id)
    if user:
        db.session.delete(user)
        db.session.commit()
        return jsonify({'message': 'User deleted'})
    return jsonify({'message': 'User not found'}), 404
# ================================================================================================================= USER


# ================================================================================================================= TASK
@app.route('/tasks', methods=['POST'])
def create_task():
    data = request.json
    task_id = data.get('T_ID')

    existing_task = Task.query.filter_by(T_ID=task_id).first()
    if existing_task:
        return jsonify({'message': 'Task with the same ID already exists'}), 400

    new_task = Task(**data)
    db.session.add(new_task)
    db.session.commit()
    return jsonify(new_task.serialize()), 201

@app.route('/tasks', methods=['GET'])
def get_tasks():
    tasks = Task.query.all()
    return jsonify([task.serialize() for task in tasks])

@app.route('/tasks/<id>', methods=['GET'])
def get_task(id):
    task = Task.query.get(id)
    if task:
        return jsonify(task.serialize())
    return jsonify({'message': 'Task not found'}), 404

@app.route('/tasks/<id>', methods=['PUT'])
def update_task(id):
    task = Task.query.get(id)
    if task:
        data = request.json
        for key, value in data.items():
            setattr(task, key, value)
        db.session.commit()
        return jsonify(task.serialize())
    return jsonify({'message': 'Task not found'}), 404

@app.route('/tasks/<id>', methods=['DELETE'])
def delete_task(id):
    task = Task.query.get(id)
    if task:
        db.session.delete(task)
        db.session.commit()
        return jsonify({'message': 'Task deleted successfully'})
    return jsonify({'message': 'Task not found'}), 404
# ================================================================================================================= TASK

# ============================================================================================================= ML MODEL

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Convert JSON to DataFrame
        user_input = pd.DataFrame(data)

        # Reorder columns to match the order during training
        user_input = user_input[['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                                 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING',
                                 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN',
                                 'GENDER_FEMALE', 'GENDER_MALE']]

        # Make predictions
        prediction = model.predict(user_input)

        # Return prediction as JSON
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        # Handle any errors and return appropriate response
        error_message = str(e)
        return jsonify({'error': error_message}), 500
# ============================================================================================================= ML MODEL

def create_tables():
    with app.app_context():
        db.create_all()


if __name__ == '__main__':
    create_tables()
    app.run(debug=True)
