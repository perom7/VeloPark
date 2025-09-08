from models.user import db
from datetime import datetime

class ParkingLot(db.Model):
    __tablename__ = 'parking_lots'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)  # Price per unit time
    address = db.Column(db.String(200), nullable=False)
    pincode = db.Column(db.String(10), nullable=False)
    max_spots = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    parking_spots = db.relationship('ParkingSpot', backref='parking_lot', lazy=True, cascade='all, delete-orphan')
    
    def get_available_spots_count(self):
        return len([spot for spot in self.parking_spots if spot.status == 'A'])
    
    def get_occupied_spots_count(self):
        return len([spot for spot in self.parking_spots if spot.status == 'O'])
    
    def can_delete(self):
        """Check if all spots in the lot are empty"""
        return self.get_occupied_spots_count() == 0
    
    def __repr__(self):
        return f'<ParkingLot {self.name}>' 