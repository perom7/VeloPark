from models.user import db
from datetime import datetime

class Reservation(db.Model):
    __tablename__ = 'reservations'
    
    id = db.Column(db.Integer, primary_key=True)
    spot_id = db.Column(db.Integer, db.ForeignKey('parking_spots.id', ondelete='CASCADE'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    parking_timestamp = db.Column(db.DateTime, nullable=False)
    leaving_timestamp = db.Column(db.DateTime, nullable=True)
    cost = db.Column(db.Float, nullable=True)  # Calculated when leaving
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def is_active(self):
        """Check if reservation is currently active (not left yet)"""
        return self.leaving_timestamp is None
    
    def get_duration_hours(self):
        """Get duration in hours"""
        if not self.leaving_timestamp:
            return None
        
        duration = self.leaving_timestamp - self.parking_timestamp
        return duration.total_seconds() / 3600
    
    def calculate_cost(self):
        """Calculate cost based on duration and lot price"""
        if not self.leaving_timestamp:
            return None
        
        duration_hours = self.get_duration_hours()
        lot_price = self.parking_spot.parking_lot.price
        return duration_hours * lot_price
    
    def complete_reservation(self):
        """Mark reservation as complete and calculate cost"""
        self.leaving_timestamp = datetime.utcnow()
        self.cost = self.calculate_cost()
    
    def __repr__(self):
        return f'<Reservation {self.id} - Spot {self.spot_id}>' 