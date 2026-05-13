"""
AI Network Threat Detector API - JWT Authentication

FEATURES:
- JWT token-based authentication (faster than API keys)
- User registration with username/password
- Password hashing with bcrypt
- Token-based requests (no database lookup on every request)
- Admin and regular user roles
"""

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import ollama
import uvicorn
import json
import io
import os
from datetime import datetime, timedelta
from pathlib import Path
import jwt
from passlib.context import CryptContext

# JWT Configuration
SECRET_KEY = "nirvana5849"  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

app = FastAPI(title="AI Network Threat Detector API - JWT Auth")
security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Storage paths
USERS_FILE = "data/users.json"
MODELS_DIR = "models"
DATA_DIR = "data"

# Create directories
Path(DATA_DIR).mkdir(exist_ok=True)
Path(MODELS_DIR).mkdir(exist_ok=True)

# User database functions
def load_users():
    try:
        with open(USERS_FILE, 'r') as f:
            content = f.read().strip()
            if not content:  # Empty file
                raise ValueError("Empty file")
            return json.loads(content)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        # Create default admin user with short password (bcrypt max 72 bytes)
        default_users = {
            "admin": {
                "username": "admin",
                "hashed_password": pwd_context.hash("admin123"),  # Simple default password
                "email": "admin@example.com",
                "role": "admin",
                "created_at": datetime.now().isoformat(),
                "is_active": True
            }
        }
        save_users(default_users)
        return default_users

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

users_db = load_users()

# Pydantic Models
class UserRegister(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    
    @field_validator('username')
    def username_validator(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v
    
    @field_validator('password')
    def password_validator(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        return v

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    role: str

class UserResponse(BaseModel):
    username: str
    email: Optional[str]
    role: str
    created_at: str
    is_active: bool

class ThreatAnalysisRequest(BaseModel):
    network_data: List[Dict[str, Any]]
    use_llm: bool = True
    llm_model: str = "Gemma3:1b"
    confidence_threshold: float = 0.7

class ThreatAnalysisResponse(BaseModel):
    total_records: int
    threats_detected: int
    average_confidence: float
    predictions: List[Dict[str, Any]]
    llm_analysis: Optional[List[Dict[str, Any]]] = None

# Password utilities
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# JWT utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# Authentication dependencies
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return current user"""
    token = credentials.credentials
    payload = decode_token(token)
    
    username = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    if username not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    user = users_db[username]
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is inactive"
        )
    
    return user

def get_current_admin(current_user: dict = Depends(get_current_user)):
    """Verify user is admin"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# Load ML model and preprocessor
def load_ml_model():
    """Load ML model with proper feature selection support"""
    try:
        model = joblib.load(f"{MODELS_DIR}/threat_classifier.pkl")
        preprocessor = joblib.load(f"{MODELS_DIR}/preprocessor.pkl")
        
        # Load label mapping
        try:
            label_mapping = joblib.load(f"{MODELS_DIR}/label_mapping.pkl")
        except:
            label_mapping = {}
        
        # Load feature engineer (for feature selection)
        try:
            feature_engineer = joblib.load(f"{MODELS_DIR}/feature_engineer.pkl")
            print("✓ Feature engineer loaded")
        except:
            feature_engineer = None
            print("⚠ Feature engineer not found - using all features")
        
        # Load selected features
        try:
            selected_features = joblib.load(f"{MODELS_DIR}/selected_features.pkl")
            print(f"✓ Selected features loaded: {len(selected_features)} features")
            print(f"  First 3 features: {selected_features[:3]}")
        except:
            selected_features = None
            print("⚠ Selected features not found - using all features")
        
        # Get expected feature count from model
        if hasattr(model, 'n_features_in_'):
            print(f"✓ Model expects {model.n_features_in_} features")
        
        print("✓ ML model loaded successfully")
        return model, preprocessor, label_mapping, feature_engineer, selected_features
    except Exception as e:
        print(f"✗ ML model not loaded: {e}")
        import traceback
        traceback.print_exc()
        return None, None, {}, None, None

model, preprocessor, label_mapping, feature_engineer, selected_features = load_ml_model()

# LLM Analyzer
class LLMAnalyzer:
    def __init__(self, model_name='Gemma3:1b'):
        self.model_name = model_name
    
    def format_network_data(self, row_data, feature_names):
        formatted = "Network Traffic Data:\n"
        for feature, value in zip(feature_names, row_data):
            formatted += f"  - {feature}: {value}\n"
        return formatted
    
    def analyze_threat(self, network_data, ml_prediction, confidence, label_mapping=None):
        threat_label = ml_prediction
        if label_mapping and ml_prediction in label_mapping:
            threat_label = label_mapping[ml_prediction]
        
        prompt = f"""You are a network security expert analyzing potential threats.

{network_data}

Machine Learning Model Results:
- Prediction: {threat_label}
- Confidence: {confidence:.2%}

Provide a concise analysis (max 200 words):
1. THREAT ASSESSMENT: Is this malicious? (Yes/No/Uncertain)
2. THREAT TYPE: What type of attack?
3. EXPLANATION: Why is this a threat?
4. RECOMMENDED ACTION: What should be done?
5. RISK LEVEL: (Low/Medium/High/Critical)"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': 'You are a cybersecurity analyst. Be concise.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"LLM analysis unavailable: {str(e)}"

# ============================================================================
# PUBLIC ROUTES (No authentication required)
# ============================================================================

@app.get("/")
def root():
    return {
        "service": "AI Network Threat Detector API - JWT Auth",
        "version": "2.0.0",
        "status": "online",
        "model_loaded": model is not None,
        "authentication": "JWT",
        "endpoints": {
            "register": "/register",
            "login": "/login",
            "analyze": "/analyze (requires token)",
            "admin": "/admin/* (requires admin token)"
        }
    }

@app.get("/health")
def health_check():
    model_info = {}
    if model is not None:
        model_info["model_type"] = type(model).__name__
        if hasattr(model, 'n_features_in_'):
            model_info["expected_features"] = model.n_features_in_
    
    return {
        "status": "healthy",
        "ml_model": model is not None,
        "preprocessor": preprocessor is not None,
        "selected_features": selected_features is not None,
        "feature_count": len(selected_features) if selected_features is not None else None,
        "model_info": model_info,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/register", response_model=UserResponse)
def register(user: UserRegister):
    """Register a new user"""
    if user.username in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Create new user
    users_db[user.username] = {
        "username": user.username,
        "hashed_password": get_password_hash(user.password),
        "email": user.email,
        "role": "user",  # Default role
        "created_at": datetime.now().isoformat(),
        "is_active": True
    }
    
    save_users(users_db)
    
    return UserResponse(
        username=user.username,
        email=user.email,
        role="user",
        created_at=users_db[user.username]["created_at"],
        is_active=True
    )

@app.post("/login", response_model=Token)
def login(user: UserLogin):
    """Login and get JWT token"""
    # Check if user exists
    if user.username not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    user_data = users_db[user.username]
    
    # Verify password
    if not verify_password(user.password, user_data["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Check if user is active
    if not user_data.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled"
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user.username, "role": user_data["role"]}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        username=user.username,
        role=user_data["role"]
    )

# ============================================================================
# USER ROUTES (Requires authentication)
# ============================================================================

@app.get("/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user info"""
    return UserResponse(
        username=current_user["username"],
        email=current_user.get("email"),
        role=current_user["role"],
        created_at=current_user["created_at"],
        is_active=current_user["is_active"]
    )

@app.post("/analyze", response_model=ThreatAnalysisResponse)
def analyze_threats(
    request: ThreatAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """Analyze network traffic for threats - FIXED VERSION"""
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="ML model not available. Admin must train model first.")
    
    try:
        print(f"\n{'='*60}")
        print(f"ANALYZING THREATS - User: {current_user['username']}")
        print(f"{'='*60}")
        
        # Convert to DataFrame
        df = pd.DataFrame(request.network_data)
        print(f"\n📊 Input data: {len(df)} records, {len(df.columns)} columns")
        
        # Preprocess - FIX: Use fit=False for prediction
        print("  ↳ Preprocessing...")
        X, _, _ = preprocessor.preprocess_data(df.copy(), fit=False)
        print(f"  ✓ After preprocessing: {X.shape[0]} rows, {X.shape[1]} features")
        
        # Apply feature selection if available
        if selected_features is not None:
            print(f"  ↳ Applying feature selection...")
            print(f"    Model expects: {len(selected_features)} features")
            print(f"    Available features: {X.shape[1]}")
            
            # Handle both list and array types
            if isinstance(selected_features, (list, tuple)):
                feature_names = selected_features
            elif isinstance(selected_features, np.ndarray):
                if selected_features.dtype == bool:
                    feature_names = X.columns[selected_features].tolist()
                else:
                    feature_names = selected_features.tolist()
            else:
                feature_names = list(selected_features)
            
            # Check which features are missing
            missing_features = set(feature_names) - set(X.columns)
            if missing_features:
                print(f"  ⚠ WARNING: Missing features: {missing_features}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Input data missing required features: {missing_features}"
                )
            
            # Select only the required features
            try:
                X = X[feature_names]
                print(f"  ✓ Selected {X.shape[1]} features")
            except KeyError as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Feature selection failed: {e}"
                )
        else:
            print(f"  ℹ No feature selection - using all {X.shape[1]} features")
        
        # Verify feature count matches model expectation
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            if X.shape[1] != expected_features:
                error_msg = (
                    f"Feature count mismatch! "
                    f"Model expects {expected_features} features, "
                    f"but got {X.shape[1]} features."
                )
                print(f"  ✗ {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
        
        # Predict
        print("  ↳ Making predictions...")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        confidences = np.max(probabilities, axis=1)
        print(f"  ✓ Predictions complete")
        
        # Format results
        results = []
        for i in range(len(predictions)):
            pred_raw = predictions[i]
            # Convert numpy int to Python int before dict lookup
            pred_key = int(pred_raw)
            pred_label = label_mapping.get(pred_key, f"Class {pred_key}")
            # Also ensure label is a plain string
            pred_label = str(pred_label)

            results.append({
                "record_id": i,
                "prediction": pred_label,
                "confidence": float(confidences[i]),
                "is_threat": pred_label.lower() != "normal"
                      })
        
        threats_count = sum(1 for r in results if r["is_threat"])
        print(f"  ✓ Detected {threats_count} threats out of {len(results)} records")
        
        # LLM Analysis
        llm_analysis = None
        if request.use_llm:
            print("  ↳ Running LLM analysis...")
            analyzer = LLMAnalyzer(model_name=request.llm_model)
            llm_analysis = []
            
            # Analyze top threats
            high_conf_indices = [
                i for i in range(len(confidences)) 
                if confidences[i] >= request.confidence_threshold and results[i]["is_threat"]
            ]
            
            print(f"    Analyzing {min(len(high_conf_indices), 5)} high-confidence threats...")
            
            for idx in high_conf_indices[:10]:  # Max 5 detailed analyses
                # Use the selected feature columns for LLM formatting
                feature_cols = feature_names if selected_features is not None else X.columns
                
                network_data = analyzer.format_network_data(
                    X.iloc[idx].values,
                    feature_cols
                )
                
                analysis = analyzer.analyze_threat(
                    network_data,
                    predictions[idx],
                    confidences[idx],
                    label_mapping
                )
                
                llm_analysis.append({
                    "record_id": int(idx),
                    "analysis": str(analysis)
                })
            
            print(f"  ✓ LLM analysis complete")
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}\n")
        
        return ThreatAnalysisResponse(
            total_records=len(results),
            threats_detected=sum(1 for r in results if r["is_threat"]),
            average_confidence=float(np.mean(confidences)),
            predictions=results,
            llm_analysis=llm_analysis
        )
    
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"\n✗ ANALYSIS FAILED:")
        print(error_detail)
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/models")
def list_models(current_user: dict = Depends(get_current_user)):
    """List available LLM models"""
    try:
        models_list = ollama.list()
        return {
            "models": [m['name'] for m in models_list.get('models', [])],
            "recommended": "llama3.2:1b"
        }
    except:
        return {
            "models": ["llama3.2:1b", "phi3:mini", "Gemma3:1b"],
            "recommended": "llama3.2:1b"
        }

# ============================================================================
# ADMIN ROUTES (Requires admin role)
# ============================================================================

@app.get("/admin/users")
def list_users(current_admin: dict = Depends(get_current_admin)):
    """Admin: List all users"""
    users_list = []
    for username, user_data in users_db.items():
        users_list.append({
            "username": username,
            "email": user_data.get("email"),
            "role": user_data["role"],
            "created_at": user_data["created_at"],
            "is_active": user_data["is_active"]
        })
    
    return {
        "total_users": len(users_list),
        "users": users_list
    }

@app.post("/admin/users/{username}/promote")
def promote_user(username: str, current_admin: dict = Depends(get_current_admin)):
    """Admin: Promote user to admin"""
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    users_db[username]["role"] = "admin"
    save_users(users_db)
    
    return {"message": f"User {username} promoted to admin"}

@app.post("/admin/users/{username}/deactivate")
def deactivate_user(username: str, current_admin: dict = Depends(get_current_admin)):
    """Admin: Deactivate user"""
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    if username == current_admin["username"]:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")
    
    users_db[username]["is_active"] = False
    save_users(users_db)
    
    return {"message": f"User {username} deactivated"}

@app.post("/admin/users/{username}/activate")
def activate_user(username: str, current_admin: dict = Depends(get_current_admin)):
    """Admin: Activate user"""
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    users_db[username]["is_active"] = True
    save_users(users_db)
    
    return {"message": f"User {username} activated"}

@app.delete("/admin/users/{username}")
def delete_user(username: str, current_admin: dict = Depends(get_current_admin)):
    """Admin: Delete user"""
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    if username == current_admin["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    del users_db[username]
    save_users(users_db)
    
    return {"message": f"User {username} deleted"}

@app.post("/admin/train")
async def train_model(
    files: List[UploadFile] = File(...),
    current_admin: dict = Depends(get_current_admin)
):
    """Admin: Train model on uploaded datasets"""
    
    try:
        # Load all uploaded files
        print(f"\n{'='*60}")
        print("LOADING DATASETS")
        print(f"{'='*60}\n")
        print(f"Received {len(files)} file(s)")
        
        all_data = []
        
        for file in files:
            print(f"\n📁 Processing: {file.filename}")
            print(f"  Content type: {file.content_type}")
            
            try:
                contents = await file.read()
                
                # Parse based on content type
                if file.filename.endswith('.csv') or 'csv' in file.content_type.lower():
                    df = pd.read_csv(io.BytesIO(contents))
                elif file.filename.endswith('.json') or 'json' in file.content_type.lower():
                    df = pd.read_json(io.BytesIO(contents))
                else:
                    print(f"    ⚠ Unknown file type: {file.filename}")
                    print(f"    ℹ Attempting to read as CSV...")
                    df = pd.read_csv(io.BytesIO(contents))
                
                print(f"    ✓ Loaded: {len(df)} rows, {len(df.columns)} columns")
                
                # Show label distribution if label column exists
                label_candidates = ['label', 'Label', 'class', 'attack_type', 'threat']
                label_col = None
                for candidate in label_candidates:
                    if candidate in df.columns:
                        label_col = candidate
                        break
                
                if label_col:
                    print(f"    Label distribution:")
                    for label, count in df[label_col].value_counts().items():
                        print(f"      - {label}: {count}")
                
                all_data.append(df)
                
            except Exception as e:
                print(f"    ✗ Error processing {file.filename}: {str(e)}")
                continue
        
        if not all_data:
            raise HTTPException(
                status_code=400, 
                detail="No valid datasets could be loaded. Ensure files are CSV or JSON format."
            )
        
        # Combine all datasets
        print(f"\n{'='*60}")
        print("COMBINING DATASETS")
        print(f"{'='*60}\n")
        
        training_df = pd.concat(all_data, ignore_index=True)
        
        print(f"✓ Combined dataset:")
        print(f"  Total rows: {len(training_df):,}")
        print(f"  Total columns: {len(training_df.columns)}")
        
        # Show overall label distribution
        label_col = None
        for candidate in ['label', 'Label', 'class', 'attack_type', 'threat']:
            if candidate in training_df.columns:
                label_col = candidate
                break
        
        if label_col:
            print(f"\n  Combined label distribution:")
            for label, count in training_df[label_col].value_counts().items():
                percentage = (count / len(training_df)) * 100
                print(f"    - {label}: {count:,} ({percentage:.1f}%)")
        
        # Save temporary combined file
        print(f"\n{'='*60}")
        print("TRAINING MODEL")
        print(f"{'='*60}\n")
        
        temp_path = f"{DATA_DIR}/temp_training_combined.csv"
        training_df.to_csv(temp_path, index=False)
        print(f"✓ Saved combined dataset to: {temp_path}")
        
        # Train model
        from train_model import ThreatModelTrainer
        
        trainer = ThreatModelTrainer()
        accuracy = trainer.train(temp_path)
        
        # Save model
        print("\nSaving trained model...")
        success = trainer.save_model()
        
        if not success:
            raise Exception("Model training succeeded but saving failed. Check permissions.")
        
        # Reload model in the API
        print("Reloading model in API...")
        global model, preprocessor, label_mapping, feature_engineer, selected_features
        model, preprocessor, label_mapping, feature_engineer, selected_features = load_ml_model()
        
        # Clean up temp file
        try:
            os.remove(temp_path)
            print(f"✓ Cleaned up temporary file")
        except:
            pass
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE!")
        print(f"{'='*60}\n")
        
        return {
            "message": "Model trained successfully with multiple datasets",
            "accuracy": float(accuracy),
            "datasets_processed": len(all_data),
            "total_records": len(training_df),
            "records_per_dataset": [len(df) for df in all_data],
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"\n✗ TRAINING FAILED:")
        print(error_detail)
        raise HTTPException(
            status_code=500, 
            detail=f"Training failed: {str(e)}\n\nFull error:\n{error_detail}"
        )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI Network Threat Detector - JWT Authentication")
    print("="*60)
    print(f"\nDefault Admin Credentials:")
    print(f"  Username: admin")
    print(f"  Password: admin123")
    print(f"\nAPI will be available at: http://localhost:8000")
    print(f"Interactive docs: http://localhost:8000/docs")
    print("\nStarting server...\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)