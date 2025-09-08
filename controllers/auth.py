from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from models.user import User, db
from functools import wraps

auth = Blueprint('auth', __name__)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin():
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

@auth.route('/')
def home():
    return render_template('home.html', year=2025)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please fill in all fields.', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            
            if user.is_admin():
                return redirect(url_for('admin.dashboard'))
            else:
                return redirect(url_for('user.dashboard'))
        else:
            flash('Invalid credentials.', 'error')
    
    return render_template('login.html')

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not password or not confirm_password:
            flash('Please fill in all fields.', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return render_template('register.html')
        
        user = User(username=username, role='user')
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('register.html')

@auth.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    if request.method == 'POST':
        session.clear()
        flash('You have been logged out.', 'success')
        return redirect(url_for('auth.home'))
    
    return render_template('logout.html')

@auth.route('/profile')
@login_required
def profile():
    user = User.query.get(session['user_id'])
    bookings = []
    for reservation in user.reservations:
        spot = reservation.parking_spot
        lot = spot.parking_lot if spot else None
        bookings.append({
            'spot_id': spot.id if spot else '-',
            'lot_name': lot.name if lot else '-',
            'status': 'Occupied' if reservation.leaving_timestamp is None else 'Completed',
            'parking_timestamp': reservation.parking_timestamp,
            'leaving_timestamp': reservation.leaving_timestamp,
            'cost': reservation.cost
        })
    
    # Calculate total spent
    total_spent = sum(b['cost'] for b in bookings if b['cost'] is not None)

    return render_template('profile.html', user=user, bookings=bookings, total_spent=total_spent) 