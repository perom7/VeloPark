from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from models.user import User, db
from models.parking_lot import ParkingLot
from models.parking_spot import ParkingSpot
from models.reservation import Reservation
from controllers.auth import admin_required
from datetime import datetime
from sqlalchemy import func

admin = Blueprint('admin', __name__, url_prefix='/admin')

@admin.route('/dashboard')
@admin_required
def dashboard():
    parking_lots = ParkingLot.query.all()
    users = User.query.filter(User.role != 'admin').all()
    
    # Get all parking spots with lot names and user info (show all slots)
    parking_spots = []
    for spot in ParkingSpot.query.join(ParkingLot).all():
        lot_name = spot.parking_lot.name
        current_res = spot.get_current_reservation() if spot.is_occupied() else None
        parking_spots.append({
            'id': spot.id,
            'lot_name': lot_name,
            'status': 'Occupied' if spot.is_occupied() else 'Available',
            'user': current_res.user.username if current_res else None
        })
    
    # Prepare data for charts
    lot_names = [lot.name for lot in parking_lots]
    available_spots = [lot.get_available_spots_count() for lot in parking_lots]
    occupied_spots = [lot.get_occupied_spots_count() for lot in parking_lots]

    # Calculate total earnings and earnings per user
    total_earnings = db.session.query(func.sum(Reservation.cost)).scalar() or 0
    earnings_by_user = db.session.query(
        User.username,
        func.sum(Reservation.cost).label('total_paid')
    ).join(Reservation, Reservation.user_id == User.id).filter(
        Reservation.cost.isnot(None)
    ).group_by(User.username).order_by(func.sum(Reservation.cost).desc()).all()

    return render_template('admin_dashboard.html', 
                         parking_lots=parking_lots,
                         users=users,
                         parking_spots=parking_spots,
                         lot_names=lot_names,
                         available_spots=available_spots,
                         occupied_spots=occupied_spots,
                         total_earnings=total_earnings,
                         earnings_by_user=earnings_by_user)

@admin.route('/search_spots', methods=['GET', 'POST'])
@admin_required
def search_spots():
    query = request.args.get('query', '').strip()
    status = request.args.get('status', '')
    spots_query = ParkingSpot.query
    if query:
        if query.isdigit():
            spots_query = spots_query.filter(ParkingSpot.id == int(query))
    if status in ['A', 'O']:
        spots_query = spots_query.filter(ParkingSpot.status == status)
    spots = spots_query.all()
    # Attach lot name and user info
    spot_data = []
    for spot in spots:
        current_res = spot.get_current_reservation() if spot.is_occupied() else None
        spot_data.append({
            'id': spot.id,
            'lot_name': spot.parking_lot.name,
            'status': 'Occupied' if spot.is_occupied() else 'Available',
            'user': current_res.user.username if current_res else None,
            'parking_timestamp': current_res.parking_timestamp if current_res else None,
            'leaving_timestamp': current_res.leaving_timestamp if current_res else None
        })
    return render_template('admin_dashboard.html',
        parking_lots=ParkingLot.query.all(),
        users=User.query.filter(User.role != 'admin').all(),
        parking_spots=spot_data,
        lot_names=[], available_spots=[], occupied_spots=[],
        search_mode=True, search_query=query, search_status=status)

@admin.route('/parking_lot/create', methods=['GET', 'POST'])
@admin_required
def create_parking_lot():
    if request.method == 'POST':
        name = request.form.get('name')
        price = request.form.get('price')
        address = request.form.get('address')
        pincode = request.form.get('pincode')
        max_spots = request.form.get('max_spots')
        
        if not all([name, price, address, pincode, max_spots]):
            flash('Please fill in all fields.', 'error')
            return render_template('parking_lot_form.html', form_title='Create Parking Lot')
        
        try:
            price = float(price)
            max_spots = int(max_spots)
        except ValueError:
            flash('Invalid price or max spots value.', 'error')
            return render_template('parking_lot_form.html', form_title='Create Parking Lot')
        
        parking_lot = ParkingLot(
            name=name,
            price=price,
            address=address,
            pincode=pincode,
            max_spots=max_spots
        )
        
        db.session.add(parking_lot)
        db.session.commit()
        
        # Create parking spots
        for i in range(max_spots):
            spot = ParkingSpot(lot_id=parking_lot.id)
            db.session.add(spot)
        
        db.session.commit()
        flash('Parking lot created successfully!', 'success')
        return redirect(url_for('admin.dashboard'))
    
    return render_template('parking_lot_form.html', 
                         form_title='Create Parking Lot',
                         form_action=url_for('admin.create_parking_lot'))

@admin.route('/parking_lot/edit/<int:lot_id>', methods=['GET', 'POST'])
@admin_required
def edit_parking_lot(lot_id):
    parking_lot = ParkingLot.query.get_or_404(lot_id)
    
    if request.method == 'POST':
        name = request.form.get('name')
        price = request.form.get('price')
        address = request.form.get('address')
        pincode = request.form.get('pincode')
        max_spots = request.form.get('max_spots')
        
        if not all([name, price, address, pincode, max_spots]):
            flash('Please fill in all fields.', 'error')
            return render_template('parking_lot_form.html', 
                                 form_title='Edit Parking Lot',
                                 lot=parking_lot)
        
        try:
            price = float(price)
            new_max_spots = int(max_spots)
        except ValueError:
            flash('Invalid price or max spots value.', 'error')
            return render_template('parking_lot_form.html', 
                                 form_title='Edit Parking Lot',
                                 lot=parking_lot)
        
        # Update lot details
        parking_lot.name = name
        parking_lot.price = price
        parking_lot.address = address
        parking_lot.pincode = pincode
        
        # Handle spot count changes
        current_spots = len(parking_lot.parking_spots)
        if new_max_spots > current_spots:
            # Add more spots
            for i in range(new_max_spots - current_spots):
                spot = ParkingSpot(lot_id=parking_lot.id)
                db.session.add(spot)
        elif new_max_spots < current_spots:
            # Remove spots (only if they're available)
            available_spots = [spot for spot in parking_lot.parking_spots if spot.is_available()]
            if len(available_spots) >= (current_spots - new_max_spots):
                spots_to_remove = available_spots[:current_spots - new_max_spots]
                for spot in spots_to_remove:
                    db.session.delete(spot)
            else:
                flash('Cannot reduce spots. Some spots are occupied.', 'error')
                return render_template('parking_lot_form.html', 
                                     form_title='Edit Parking Lot',
                                     lot=parking_lot)
        
        parking_lot.max_spots = new_max_spots
        db.session.commit()
        flash('Parking lot updated successfully!', 'success')
        return redirect(url_for('admin.dashboard'))
    
    return render_template('parking_lot_form.html', 
                         form_title='Edit Parking Lot',
                         form_action=url_for('admin.edit_parking_lot', lot_id=lot_id),
                         lot=parking_lot,
                         submit_label='Update')

@admin.route('/parking_lot/delete/<int:lot_id>')
@admin_required
def delete_parking_lot(lot_id):
    parking_lot = ParkingLot.query.get_or_404(lot_id)
    
    if not parking_lot.can_delete():
        flash('Cannot delete parking lot. Some spots are occupied.', 'error')
        return redirect(url_for('admin.dashboard'))
    
    db.session.delete(parking_lot)
    db.session.commit()
    flash('Parking lot deleted successfully!', 'success')
    return redirect(url_for('admin.dashboard'))

@admin.route('/parking_spot/view/<int:spot_id>')
@admin_required
def view_parking_spot(spot_id):
    spot = ParkingSpot.query.get_or_404(spot_id)
    
    # Get current reservation if occupied
    current_reservation = None
    if spot.is_occupied():
        current_reservation = spot.get_current_reservation()
    
    spot_data = {
        'id': spot.id,
        'lot_name': spot.parking_lot.name,
        'status': 'Occupied' if spot.is_occupied() else 'Available',
        'user': current_reservation.user.username if current_reservation else None,
        'parking_timestamp': current_reservation.parking_timestamp if current_reservation else None,
        'leaving_timestamp': current_reservation.leaving_timestamp if current_reservation else None
    }
    
    return render_template('parking_spot_detail.html', spot=spot_data) 