VEHICLE PARKING APP - V1
PROJECT REPORT
Modern Application Development I - IIT Madras

================================================================================

STUDENT DETAILS
================================================================================

Name: [Your Name]
Roll Number: [Your Roll Number]
Course: Modern Application Development I
Term: May 2025
Project: Vehicle Parking App - V1

================================================================================

PROJECT DETAILS
================================================================================

PROBLEM STATEMENT
The project required developing a multi-user Vehicle Parking Management System with two distinct roles:
- Admin (superuser) with full control over parking lots and system management
- Regular users who can register, login, and reserve parking spots

The application needed to manage different parking lots, parking spots, and vehicle reservations for 4-wheeler parking with automatic spot allocation and real-time status tracking.

APPROACH TO PROBLEM STATEMENT
1. Database Design: Started with designing the database schema with four main entities - Users, Parking Lots, Parking Spots, and Reservations
2. Authentication System: Implemented role-based authentication with session management
3. Admin Functionality: Created comprehensive admin dashboard for parking lot management
4. User Functionality: Developed user registration, booking, and release system
5. API Development: Added RESTful API endpoints for data access
6. Frontend Design: Built responsive UI using Bootstrap and Chart.js
7. Testing and Validation: Implemented form validation and error handling

CORE FEATURES IMPLEMENTED
- Admin and User Login System
- Admin Dashboard with parking lot CRUD operations
- User Registration and Booking System
- Automatic Spot Allocation Algorithm
- Real-time Parking Status Tracking
- Cost Calculation based on Duration
- Search and Filter Functionality
- API Resources for Data Access
- Responsive Design with Bootstrap
- Form Validation (Frontend and Backend)
- Error Handling with Custom Pages

================================================================================

FRAMEWORKS AND LIBRARIES USED
================================================================================

BACKEND FRAMEWORKS
- Flask 2.3.3: Main web framework for application development
- SQLAlchemy 2.0.21: Object-Relational Mapping for database operations
- Flask-SQLAlchemy 3.0.5: Flask extension for SQLAlchemy integration
- Werkzeug 2.3.7: WSGI utility library for password hashing and security

FRONTEND TECHNOLOGIES
- HTML5: Markup language for web pages
- CSS3: Styling and responsive design
- Bootstrap 5: CSS framework for responsive UI components
- Jinja2: Template engine for dynamic HTML generation
- Chart.js: JavaScript library for data visualization

DATABASE
- SQLite: Lightweight, serverless database system
- Programmatic Database Creation: Tables created via SQLAlchemy models

DEVELOPMENT TOOLS
- Python 3.12: Programming language
- pip: Package manager for Python dependencies
- Git: Version control system (for development tracking)

================================================================================

ER DIAGRAM AND DATABASE RELATIONS
================================================================================

DATABASE SCHEMA

USERS TABLE
- id (Primary Key, Integer, Auto-increment)
- username (String, Unique, Not Null)
- password_hash (String, Not Null)
- role (String, Default: 'user')
- created_at (DateTime, Default: UTC now)

PARKING_LOTS TABLE
- id (Primary Key, Integer, Auto-increment)
- name (String, Not Null)
- price (Float, Not Null)
- address (String, Not Null)
- pincode (String, Not Null)
- max_spots (Integer, Not Null)
- created_at (DateTime, Default: UTC now)

PARKING_SPOTS TABLE
- id (Primary Key, Integer, Auto-increment)
- lot_id (Foreign Key -> parking_lots.id, Not Null)
- status (String, Default: 'A' for Available, 'O' for Occupied)
- created_at (DateTime, Default: UTC now)

RESERVATIONS TABLE
- id (Primary Key, Integer, Auto-increment)
- spot_id (Foreign Key -> parking_spots.id, Not Null)
- user_id (Foreign Key -> users.id, Not Null)
- parking_timestamp (DateTime, Not Null)
- leaving_timestamp (DateTime, Nullable)
- cost (Float, Nullable)
- created_at (DateTime, Default: UTC now)

RELATIONSHIPS
1. Parking Lots to Parking Spots: One-to-Many
   - One parking lot can have multiple parking spots
   - Each parking spot belongs to exactly one parking lot

2. Users to Reservations: One-to-Many
   - One user can have multiple reservations
   - Each reservation belongs to exactly one user

3. Parking Spots to Reservations: One-to-Many
   - One parking spot can have multiple reservations (over time)
   - Each reservation is for exactly one parking spot

4. Cascade Delete Relationships
   - When a parking lot is deleted, all associated parking spots are deleted
   - When a parking spot is deleted, all associated reservations are deleted
   - When a user is deleted, all associated reservations are deleted

CONSTRAINTS
- Foreign key constraints ensure referential integrity
- Unique constraints on username and parking lot names
- Not null constraints on required fields
- Default values for status and timestamps

================================================================================

API RESOURCE ENDPOINTS
================================================================================

BASE URL: http://localhost:5000

AUTHENTICATION ENDPOINTS
- GET /: Home page
- GET /login: Login page
- POST /login: User authentication
- GET /register: Registration page
- POST /register: User registration
- GET /logout: Logout functionality
- POST /logout: Session termination

ADMIN ENDPOINTS
- GET /admin/dashboard: Admin dashboard
- GET /admin/search_spots: Search parking spots
- GET /admin/parking_lot/create: Create parking lot form
- POST /admin/parking_lot/create: Create new parking lot
- GET /admin/parking_lot/edit/<id>: Edit parking lot form
- POST /admin/parking_lot/edit/<id>: Update parking lot
- GET /admin/parking_lot/delete/<id>: Delete parking lot
- GET /admin/parking_spot/view/<id>: View spot details

USER ENDPOINTS
- GET /user/dashboard: User dashboard
- POST /user/book: Book parking spot
- POST /user/release/<id>: Release parking spot

PROFILE ENDPOINTS
- GET /profile: User profile page

API DATA ENDPOINTS
- GET /api/lots: Returns JSON of all parking lots
- GET /api/spots: Returns JSON of all parking spots
- GET /api/users: Returns JSON of all users

RESPONSE FORMATS
API endpoints return JSON data in the following format:

Parking Lots API:
{
  "lots": [
    {
      "id": 1,
      "name": "Sample Lot",
      "price": 20.0,
      "address": "123 Main St",
      "pincode": "123456",
      "max_spots": 5
    }
  ]
}

Parking Spots API:
{
  "spots": [
    {
      "id": 1,
      "lot_id": 1,
      "status": "A"
    }
  ]
}

Users API:
{
  "users": [
    {
      "id": 1,
      "username": "admin",
      "role": "admin"
    }
  ]
}

================================================================================

ADDITIONAL FEATURES IMPLEMENTED
================================================================================

SEARCH AND FILTER FUNCTIONALITY
- Admin can search parking spots by ID
- Filter spots by status (Available/Occupied)
- Real-time search results display

DATA VISUALIZATION
- Chart.js integration for parking lot statistics
- Bar charts showing available vs occupied spots
- Revenue tracking and user spending analytics

SECURITY FEATURES
- Password hashing using Werkzeug
- Session-based authentication
- Role-based access control
- Input validation and sanitization

RESPONSIVE DESIGN
- Bootstrap 5 for mobile-friendly interface
- Custom CSS for enhanced styling
- Cross-browser compatibility

ERROR HANDLING
- Custom 404, 403, and 500 error pages
- Form validation with user feedback
- Database constraint handling

================================================================================

TECHNICAL IMPLEMENTATION DETAILS
================================================================================

DATABASE CREATION
The database is created programmatically when the application starts:
- Tables are created using SQLAlchemy models
- Admin user is automatically created (username: admin, password: admin123)
- Sample parking lot is seeded for testing

AUTHENTICATION SYSTEM
- Custom session-based authentication
- Role-based access control (admin/user)
- Secure password hashing
- Login/logout functionality

SPOT ALLOCATION ALGORITHM
- First-come-first-served basis
- Automatic selection of first available spot
- Status tracking (Available/Occupied)
- Cost calculation based on duration

COST CALCULATION
- Hourly rate based on parking lot price
- Duration calculation from timestamps
- Automatic cost computation on spot release

================================================================================

CONCLUSION
================================================================================

The Vehicle Parking App - V1 successfully implements all core requirements specified in the problem statement. The application provides a complete parking management system with separate interfaces for administrators and users, automatic spot allocation, real-time status tracking, and comprehensive data management capabilities.

The project demonstrates proficiency in Flask web development, SQLAlchemy database operations, frontend development with Bootstrap, and API design. The modular code structure ensures maintainability and scalability for future enhancements.

All mandatory frameworks (Flask, Jinja2, HTML, CSS, Bootstrap, SQLite) have been properly implemented, and the database is created programmatically as required. The application is ready for deployment and evaluation.

================================================================================

VIDEO PRESENTATION LINK
================================================================================

[Insert your Google Drive video link here]

================================================================================

AI/LLM USAGE DECLARATION
================================================================================

[If applicable, describe the extent of AI/LLM usage in your project development]

================================================================================ 