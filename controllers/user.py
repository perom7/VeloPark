from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from models.user import User, db
from models.parking_lot import ParkingLot
from models.parking_spot import ParkingSpot
from models.reservation import Reservation
from controllers.auth import login_required
from datetime import datetime

user = Blueprint('user', __name__, url_prefix='/user')

@user.route('/dashboard')
@login_required
def dashboard():
    # Get available parking lots
    parking_lots = ParkingLot.query.all()
    
    # Get user's bookings
    user_id = session['user_id']
    bookings = db.session.query(
        Reservation.id,
        ParkingSpot.id.label('spot_id'),
        ParkingLot.name.label('lot_name'),
        Reservation.parking_timestamp,
        Reservation.leaving_timestamp,
        Reservation.cost
    ).join(ParkingSpot, Reservation.spot_id == ParkingSpot.id)\
     .join(ParkingLot, ParkingSpot.lot_id == ParkingLot.id)\
     .filter(Reservation.user_id == user_id)\
     .order_by(Reservation.created_at.desc()).all()
    
    # Convert to list of dicts for template
    booking_list = []
    for booking in bookings:
        status = 'Occupied' if booking.leaving_timestamp is None else 'Completed'
        booking_list.append({
            'id': booking.id,
            'spot_id': booking.spot_id,
            'lot_name': booking.lot_name,
            'status': status,
            'parking_timestamp': booking.parking_timestamp,
            'leaving_timestamp': booking.leaving_timestamp,
            'cost': booking.cost
        })
    
    # Prepare data for user chart: bookings per lot
    lot_booking_counts = {}
    for booking in booking_list:
        lot_booking_counts[booking['lot_name']] = lot_booking_counts.get(booking['lot_name'], 0) + 1
    chart_lot_names = list(lot_booking_counts.keys())
    chart_booking_counts = list(lot_booking_counts.values())

    # Calculate total spent
    total_spent = sum(b['cost'] for b in booking_list if b['cost'] is not None)

    return render_template('user_dashboard.html', 
                         parking_lots=parking_lots,
                         bookings=booking_list,
                         chart_lot_names=chart_lot_names,
                         chart_booking_counts=chart_booking_counts,
                         total_spent=total_spent)

@user.route('/book', methods=['POST'])
@login_required
def book_spot():
    lot_id = request.form.get('lot_id')
    
    if not lot_id:
        flash('Please select a parking lot.', 'error')
        return redirect(url_for('user.dashboard'))
    
    try:
        lot_id = int(lot_id)
    except ValueError:
        flash('Invalid parking lot selection.', 'error')
        return redirect(url_for('user.dashboard'))
    
    parking_lot = ParkingLot.query.get_or_404(lot_id)
    
    # Find first available spot
    available_spot = ParkingSpot.query.filter_by(
        lot_id=lot_id,
        status='A'
    ).first()
    
    if not available_spot:
        flash('No available spots in this parking lot.', 'error')
        return redirect(url_for('user.dashboard'))
    
    # Check if user already has an active booking
    user_id = session['user_id']
    active_booking = Reservation.query.filter_by(
        user_id=user_id,
        leaving_timestamp=None
    ).first()
    
    if active_booking:
        flash('You already have an active booking. Please release it first.', 'error')
        return redirect(url_for('user.dashboard'))
    
    # Create reservation
    reservation = Reservation(
        spot_id=available_spot.id,
        user_id=user_id,
        parking_timestamp=datetime.utcnow()
    )
    
    # Occupy the spot
    available_spot.occupy()
    
    db.session.add(reservation)
    db.session.commit()
    
    flash(f'Spot {available_spot.id} booked successfully in {parking_lot.name}!', 'success')
    return redirect(url_for('user.dashboard'))

@user.route('/release/<int:reservation_id>', methods=['POST'])
@login_required
def release_spot(reservation_id):
    user_id = session['user_id']
    reservation = Reservation.query.filter_by(
        id=reservation_id,
        user_id=user_id
    ).first_or_404()
    
    if reservation.leaving_timestamp is not None:
        flash('This booking is already completed.', 'error')
        return redirect(url_for('user.dashboard'))
    
    # Complete the reservation
    reservation.complete_reservation()
    
    # Release the spot
    spot = reservation.parking_spot
    spot.release()
    
    db.session.commit()
    
    flash(f'Spot {spot.id} released successfully! Cost: ${reservation.cost:.2f}', 'success')
    return redirect(url_for('user.dashboard')) 