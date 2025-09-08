from flask import Flask, render_template, flash
from models.user import db, User
from models.parking_lot import ParkingLot
from models.parking_spot import ParkingSpot
from models.reservation import Reservation
from controllers.auth import auth
from controllers.admin import admin
from controllers.user import user
import os

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change in production
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///parking_app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    
    # Register blueprints
    app.register_blueprint(auth)
    app.register_blueprint(admin)
    app.register_blueprint(user)
    
    # Create database tables
    with app.app_context():
        db.create_all()
        
        # Create admin user if it doesn't exist
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            admin_user = User(username='admin', role='admin')
            admin_user.set_password('admin123')  # Change in production
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user created: username='admin', password='admin123'")

        # Seed sample parking lot and spots if none exist
        if ParkingLot.query.count() == 0:
            lot = ParkingLot(name='Sample Lot', price=20.0, address='123 Main St', pincode='123456', max_spots=5)
            db.session.add(lot)
            db.session.commit()
            for _ in range(lot.max_spots):
                spot = ParkingSpot(lot_id=lot.id)
                db.session.add(spot)
            db.session.commit()
            print("Sample parking lot and spots created.")
    
    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('error.html', 
                             error_code='404 - Not Found',
                             error_message='The page you are looking for does not exist.'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template('error.html', 
                             error_code='500 - Internal Server Error',
                             error_message='Something went wrong on our end.'), 500
    
    @app.errorhandler(403)
    def forbidden_error(error):
        return render_template('error.html', 
                             error_code='403 - Forbidden',
                             error_message='You do not have permission to access this resource.'), 403
    
    # API endpoints for lots, spots, users
    @app.route('/api/lots')
    def api_lots():
        lots = ParkingLot.query.all()
        return {
            'lots': [
                {
                    'id': lot.id,
                    'name': lot.name,
                    'price': lot.price,
                    'address': lot.address,
                    'pincode': lot.pincode,
                    'max_spots': lot.max_spots
                } for lot in lots
            ]
        }

    @app.route('/api/spots')
    def api_spots():
        spots = ParkingSpot.query.all()
        return {
            'spots': [
                {
                    'id': spot.id,
                    'lot_id': spot.lot_id,
                    'status': spot.status
                } for spot in spots
            ]
        }

    @app.route('/api/users')
    def api_users():
        users = User.query.all()
        return {
            'users': [
                {
                    'id': user.id,
                    'username': user.username,
                    'role': user.role
                } for user in users
            ]
        }
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000) 