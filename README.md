# Vehicle Parking App - V1

A multi-user Flask application for managing parking lots, parking spots, and vehicle reservations.

## Project Overview

This is a Vehicle Parking Management System built with Flask that allows:
- **Admin users** to manage parking lots and view system statistics
- **Regular users** to register, login, and reserve parking spots
- **Automatic spot allocation** based on availability
- **Real-time tracking** of parking status and costs

## Technologies Used

- **Backend:** Flask 2.3.3
- **Database:** SQLite (programmatically created)
- **Frontend:** HTML, CSS, Bootstrap, Jinja2 templating
- **ORM:** SQLAlchemy 2.0.21
- **Authentication:** Custom session-based authentication

## Features

### Admin Features
-  Create, edit, and delete parking lots
-  View all parking spots and their status
-  View registered users
-  Dashboard with charts and statistics
-  Search parking spots by ID and status
-  API endpoints for data access

### User Features
-  User registration and login
-  Browse available parking lots
-  Automatic spot booking
-  Release parking spots
-  View booking history and costs
-  Personal dashboard with statistics

## Database Schema

### Tables
1. **users** - User accounts and authentication
2. **parking_lots** - Parking lot information
3. **parking_spots** - Individual parking spots
4. **reservations** - Booking records

### Key Relationships
- Parking spots belong to parking lots
- Reservations link users to parking spots
- Cascade delete ensures data integrity

## Installation & Setup

1. **Clone/Download** the project files
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python app.py
   ```
4. **Access the application** at `http://localhost:5000`

## Default Credentials

- **Admin:** username=`admin`, password=`admin123`
- **Users:** Register new accounts through the web interface

## API Endpoints

- `GET /api/lots` - List all parking lots
- `GET /api/spots` - List all parking spots
- `GET /api/users` - List all users

## Project Structure

```
MAD Project/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── controllers/          # Route handlers
│   ├── auth.py          # Authentication routes
│   ├── admin.py         # Admin functionality
│   └── user.py          # User functionality
├── models/              # Database models
│   ├── user.py          # User model
│   ├── parking_lot.py   # Parking lot model
│   ├── parking_spot.py  # Parking spot model
│   └── reservation.py   # Reservation model
├── templates/           # HTML templates
├── static/css/         # CSS styling
└── instance/           # Database files
```

## Core Functionalities Implemented

 **Admin and User Login System**
 **Admin Dashboard** with parking lot management
 **User Dashboard** with booking functionality
 **Automatic Spot Allocation**
 **Parking Spot Status Management**
 **Cost Calculation** based on duration
 **Search Functionality** for parking spots
 **API Resources** for data access
 **Responsive Design** with Bootstrap
 **Form Validation** (frontend and backend)
 **Error Handling** with custom error pages

## Additional Features

- **Real-time Charts** using Chart.js
- **Responsive Bootstrap UI**
- **Session-based Authentication**
- **Cascade Delete** for data integrity
- **Search and Filter** functionality
- **Cost Calculation** with timestamps

## Database Creation

The database is created programmatically when the application starts:
- Tables are created using SQLAlchemy models
- Admin user is automatically created
- Sample parking lot is seeded for testing

## Security Features

- Password hashing using Werkzeug
- Session-based authentication
- Role-based access control
- Input validation and sanitization

---

