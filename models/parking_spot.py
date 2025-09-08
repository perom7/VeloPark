from models.user import db
from datetime import datetime

class ParkingSpot(db.Model):
    __tablename__ = 'parking_spots'
    
    id = db.Column(db.Integer, primary_key=True)
    lot_id = db.Column(db.Integer, db.ForeignKey('parking_lots.id'), nullable=False)
    status = db.Column(db.String(1), default='A')  # 'A' for Available, 'O' for Occupied
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    reservations = db.relationship(
        'Reservation',
        backref='parking_spot',
        lazy=True,
        cascade='all, delete-orphan',
        passive_deletes=True
    )
    
    def is_available(self):
        return self.status == 'A'
    
    def is_occupied(self):
        return self.status == 'O'
    
    def occupy(self):
        self.status = 'O'
    
    def release(self):
        self.status = 'A'
    
    def get_current_reservation(self):
        """Get the current active reservation for this spot"""
        from models.reservation import Reservation
        return Reservation.query.filter_by(
            spot_id=self.id,
            leaving_timestamp=None
        ).first()
    
    def __repr__(self):
        return f'<ParkingSpot {self.id} - {self.status}>' 