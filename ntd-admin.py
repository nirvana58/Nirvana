"""
Network Threat Detector - Admin Client (JWT Authentication)

Usage:
    python ntd-admin-jwt.py login
    python ntd-admin-jwt.py train dataset1.csv dataset2.csv
    python ntd-admin-jwt.py users
    python ntd-admin-jwt.py promote <username>
"""

import requests
import json
import sys
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
TOKEN_FILE = ".ntd_token"

class AdminClient:
    def __init__(self):
        self.base_url = API_BASE_URL
        self.token = self.load_token()
    
    def load_token(self):
        """Load saved token from file"""
        try:
            with open(TOKEN_FILE, 'r') as f:
                data = json.load(f)
                return data.get('access_token')
        except:
            return None
    
    def save_token(self, token_data):
        """Save token to file"""
        with open(TOKEN_FILE, 'w') as f:
            json.dump(token_data, f)
    
    def get_headers(self):
        """Get authorization headers"""
        if not self.token:
            print("❌ Not logged in. Please run: python ntd-admin-jwt.py login")
            sys.exit(1)
        
        return {"Authorization": f"Bearer {self.token}"}
    
    def login(self, username=None, password=None):
        """Login and get JWT token"""
        if not username:
            username = input("Username: ")
        if not password:
            import getpass
            password = getpass.getpass("Password: ")
        
        try:
            response = requests.post(
                f"{self.base_url}/login",
                json={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.save_token(data)
                self.token = data['access_token']
                
                print(f"\n✅ Login successful!")
                print(f"   Username: {data['username']}")
                print(f"   Role: {data['role']}")
                print(f"   Token saved to: {TOKEN_FILE}")
                
                if data['role'] != 'admin':
                    print(f"\n⚠️  Warning: Your role is '{data['role']}', not 'admin'")
                    print(f"   You may not have access to admin commands")
            else:
                print(f"❌ Login failed: {response.json()['detail']}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def logout(self):
        """Logout and remove token"""
        try:
            Path(TOKEN_FILE).unlink()
            print("✅ Logged out successfully")
        except:
            print("⚠️  No active session found")
    
    def get_me(self):
        """Get current user info"""
        try:
            response = requests.get(
                f"{self.base_url}/me",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                user = response.json()
                print(f"\n👤 Current User:")
                print(f"   Username: {user['username']}")
                print(f"   Email: {user.get('email', 'N/A')}")
                print(f"   Role: {user['role']}")
                print(f"   Created: {user['created_at']}")
                print(f"   Active: {user['is_active']}")
            else:
                print(f"❌ Error: {response.json()['detail']}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def list_users(self):
        """List all users (admin only)"""
        try:
            response = requests.get(
                f"{self.base_url}/admin/users",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n👥 Total Users: {data['total_users']}\n")
                print(f"{'Username':<20} {'Email':<30} {'Role':<10} {'Active':<10}")
                print("-" * 70)
                
                for user in data['users']:
                    username = user['username']
                    email = user.get('email', 'N/A')[:28]
                    role = user['role']
                    active = '✓' if user['is_active'] else '✗'
                    print(f"{username:<20} {email:<30} {role:<10} {active:<10}")
            else:
                print(f"❌ Error: {response.json()['detail']}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def promote_user(self, username):
        """Promote user to admin"""
        try:
            response = requests.post(
                f"{self.base_url}/admin/users/{username}/promote",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                print(f"✅ {response.json()['message']}")
            else:
                print(f"❌ Error: {response.json()['detail']}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def deactivate_user(self, username):
        """Deactivate user"""
        try:
            response = requests.post(
                f"{self.base_url}/admin/users/{username}/deactivate",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                print(f"✅ {response.json()['message']}")
            else:
                print(f"❌ Error: {response.json()['detail']}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def activate_user(self, username):
        """Activate user"""
        try:
            response = requests.post(
                f"{self.base_url}/admin/users/{username}/activate",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                print(f"✅ {response.json()['message']}")
            else:
                print(f"❌ Error: {response.json()['detail']}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def delete_user(self, username):
        """Delete user"""
        confirm = input(f"⚠️  Are you sure you want to delete user '{username}'? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Cancelled")
            return
        
        try:
            response = requests.delete(
                f"{self.base_url}/admin/users/{username}",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                print(f"✅ {response.json()['message']}")
            else:
                print(f"❌ Error: {response.json()['detail']}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def train_model(self, file_paths):
        """Train model with datasets"""
        print(f"\n🚀 Training model with {len(file_paths)} dataset(s)...\n")
        
        # Prepare files
        files = []
        for path in file_paths:
            if not Path(path).exists():
                print(f"❌ File not found: {path}")
                continue
            
            files.append(('files', (Path(path).name, open(path, 'rb'), 'text/csv')))
        
        if not files:
            print("❌ No valid files to upload")
            return
        
        try:
            response = requests.post(
                f"{self.base_url}/admin/train",
                headers=self.get_headers(),
                files=files
            )
            
            # Close files
            for _, (_, f, _) in files:
                f.close()
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ {result['message']}")
                print(f"\n📊 Training Results:")
                print(f"   Accuracy: {result['accuracy']:.4f}")
                print(f"   Datasets processed: {result['datasets_processed']}")
                print(f"   Total records: {result['total_records']}")
                print(f"   Timestamp: {result['timestamp']}")
            else:
                error = response.json()
                print(f"\n❌ Training failed:")
                print(f"   {error['detail']}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n🏥 API Health Check:")
                print(f"   Status: {data['status']}")
                print(f"   ML Model: {'✓' if data['ml_model'] else '✗'}")
                print(f"   Preprocessor: {'✓' if data['preprocessor'] else '✗'}")
                print(f"   Features: {data.get('feature_count', 'N/A')}")
                print(f"   Timestamp: {data['timestamp']}")
            else:
                print(f"❌ Health check failed")
        
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    client = AdminClient()
    
    if len(sys.argv) < 2:
        print("""
Network Threat Detector - Admin Client (JWT Auth)

Usage:
    python ntd-admin-jwt.py login [username] [password]
    python ntd-admin-jwt.py logout
    python ntd-admin-jwt.py me
    python ntd-admin-jwt.py users
    python ntd-admin-jwt.py promote <username>
    python ntd-admin-jwt.py deactivate <username>
    python ntd-admin-jwt.py activate <username>
    python ntd-admin-jwt.py delete <username>
    python ntd-admin-jwt.py train <file1.csv> [file2.csv ...]
    python ntd-admin-jwt.py health

Examples:
    python ntd-admin-jwt.py login admin admin123
    python ntd-admin-jwt.py train data/training_data.csv
    python ntd-admin-jwt.py users
    python ntd-admin-jwt.py promote john
        """)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "login":
        username = sys.argv[2] if len(sys.argv) > 2 else None
        password = sys.argv[3] if len(sys.argv) > 3 else None
        client.login(username, password)
    
    elif command == "logout":
        client.logout()
    
    elif command == "me":
        client.get_me()
    
    elif command == "users":
        client.list_users()
    
    elif command == "promote":
        if len(sys.argv) < 3:
            print("❌ Usage: python ntd-admin-jwt.py promote <username>")
            sys.exit(1)
        client.promote_user(sys.argv[2])
    
    elif command == "deactivate":
        if len(sys.argv) < 3:
            print("❌ Usage: python ntd-admin-jwt.py deactivate <username>")
            sys.exit(1)
        client.deactivate_user(sys.argv[2])
    
    elif command == "activate":
        if len(sys.argv) < 3:
            print("❌ Usage: python ntd-admin-jwt.py activate <username>")
            sys.exit(1)
        client.activate_user(sys.argv[2])
    
    elif command == "delete":
        if len(sys.argv) < 3:
            print("❌ Usage: python ntd-admin-jwt.py delete <username>")
            sys.exit(1)
        client.delete_user(sys.argv[2])
    
    elif command == "train":
        if len(sys.argv) < 3:
            print("❌ Usage: python ntd-admin-jwt.py train <file1.csv> [file2.csv ...]")
            sys.exit(1)
        client.train_model(sys.argv[2:])
    
    elif command == "health":
        client.health_check()
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()