from flask import Flask, request, jsonify, render_template, redirect, url_for,render_template_string
from flask_sqlalchemy import SQLAlchemy
import pickle
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

app = Flask(__name__, template_folder='templates\src')
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/lung_cancer'
db = SQLAlchemy(app)

#================================================================================================================== DB Connection Check


@app.route('/check_db_connection')
def check_db_connection():
    try:
        db.session.execute(text('SELECT 1'))
        return jsonify({'message': 'Database connection successful'})
    except Exception as e:
        return jsonify({'error': 'Failed to connect to the database', 'details': str(e)}), 500


#==================================================================================================================

# Load the trained model
with open('./model/lung_cancer_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

class User(db.Model):
    U_ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    U_Name = db.Column(db.String(255), nullable=False)
    #U_Address = db.Column(db.String(255))
    #U_ContactNumber = db.Column(db.Integer)
    U_Email = db.Column(db.String(255), unique=True, nullable=False)
    U_Username = db.Column(db.String(255), unique=True, nullable=False)
    U_Password = db.Column(db.String(255), nullable=False)
    U_Dob = db.Column(db.Date)

    def serialize(self):
        return {
            'U_ID': self.U_ID,
            'U_Name': self.U_Name,
            #'U_Address': self.U_Address,
            #'U_ContactNumber': self.U_ContactNumber,
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
    U_ID = db.Column(db.Integer, db.ForeignKey('user.U_ID'))

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
    

#Direct to the Home Page
@app.route('/', methods=['GET'])
def home():
    print('Lung Cancer Server On Going!')
    return render_template(r'index.html')



# ================================================================================================================= USER
# Direct to the SignIn Page
@app.route('/signIn', methods=['GET']) 
def signin():
        return render_template(r'signIn.html')

@app.route('/signIn/Login', methods=['POST'])
def login_user():
    data = request.form 
    username = data.get('userName')
    password = data.get('password')

    # Query the database to check if the user exists and the password is correct
    user = User.query.filter_by(U_Username=username).first()
    if user:
        if user.U_Password == password:
            # Authentication successful, redirect or perform further actions
            return render_template(r'userPage.html')
        else:
            return jsonify({'message': 'Incorrect password'}), 401
    else:
        return jsonify({'message': 'User not found'}), 404

#Direct to the SignUp Page
@app.route('/signUp', methods=['GET'])
def signup():
    return render_template(r'signUp.html')

#Creation of a new user
@app.route('/signUp/newuser', methods=['POST'])
def create_user():
    data = request.form  # Retrieve form data
    first_name = data.get('firstName')
    last_name = data.get('lastName')
    dob = data.get('dob')
    email = data.get('email')
    username = data.get('userName')
    password = data.get('password')

    # Check if the email or username already exists
    existing_user_email = User.query.filter_by(U_Email=email).first()
    existing_user_username = User.query.filter_by(U_Username=username).first()

    if existing_user_email:
        return jsonify({'message': 'Email already exists'}), 400
    if existing_user_username:
        return jsonify({'message': 'Username already exists'}), 400

    full_name = (first_name or '') + ' ' + (last_name or '')  # Ensure first_name and last_name are not None

    try:
        new_user = User(
            U_Name=full_name.strip(),
            U_Email=email,
            U_Username=username,
            U_Password=password,
            U_Dob =dob,  # Make sure this matches the field name in the model (U_Dob)
            
        )
        db.session.add(new_user)
        db.session.commit()
        #return jsonify(new_user.serialize()), 201
        return render_template(r'userPage.html')
    except IntegrityError:
        db.session.rollback()
        return jsonify({'message': 'IntegrityError: Unable to create user due to duplicate values'}), 400

# ================================================================================================================= USER

# ================================================================================================================= Test and History Page
@app.route('/test')
def test_page():
    return render_template('test.html')

@app.route('/history')
def history_page():
    return render_template('history.html')
# =================================================================================================================

# ============================================================================================================= ML MODEL

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = request.form['age']
        gender = request.form['gender']
        alcoholUse = request.form['alcoholUse'] 
        dustAllergy = request.form['dustAllergy']
        occuPationalHezards = request.form['occuPationalHezards']
        geneticRisk = request.form['geneticRisk']
        chronicLungDisease = request.form['chronicLungDisease']
        BalancedDiet = request.form['BalancedDiet']
        obesity = request.form['obesity']
        possiveSmoker = request.form['possiveSmoker']
        chestPain = request.form['chestPain']
        CoughingOfBlood = request.form['CoughingOfBlood']
        weightLoos = request.form['weightLoos']
        wheezing = request.form['wheezing']
        swallowingDifficulty = request.form['swallowingDifficulty']
        clubbingOfFingerNails = request.form['clubbingOfFingerNails']
        frequentCold = request.form['frequentCold']
        airPollution = request.form['airPollution']
        dryCough = request.form['dryCough']
        snoring = request.form['snoring']
        
        data = {
            'AGE': [age],
            'SMOKING': [possiveSmoker],
            'YELLOW_FINGERS': [clubbingOfFingerNails],
            'ANXIETY': [weightLoos],
            'PEER_PRESSURE': [occuPationalHezards],
            'CHRONIC DISEASE': [chronicLungDisease],
            'FATIGUE ': [CoughingOfBlood], 
            'ALLERGY ': [dustAllergy], 
            'WHEEZING': [wheezing], 
            'ALCOHOL CONSUMING': [alcoholUse],
            'COUGHING': [dryCough], 
            'SHORTNESS OF BREATH': [snoring], 
            'SWALLOWING DIFFICULTY': [swallowingDifficulty], 
            'CHEST PAIN': [chestPain],
            'GENDER_FEMALE': [1 if gender == 'female' else 0],  
            'GENDER_MALE': [1 if gender == 'male' else 0],
                        
            # Add more columns and corresponding form fields as needed
        }
        
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
        return render_template_string('<script>alert("Prediction: {{ prediction }}"); window.history.back();</script>', prediction=prediction[0])

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
