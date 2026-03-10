# app_state.py — global state shared across all screens

class AppState:
    """
    Global state shared across all screens.
    Pass this object to every screen constructor.
    """
    def __init__(self):
        self.user_role     = None          # "signer" or "speaker"
        self.session_id    = None          # conversation session ID string
        self.current_user  = None          # active signed-in user record
        self.conv_history  = []            # list of {who, text, grammar_type}
        self.is_connected  = False         # WebSocket connection status
        self.current_translation = ""
        self.current_grammar     = ""
        self.users = {
            "demo@sense.ai": {
                "full_name": "Maya Johnson",
                "password": "Sense1234",
            }
        }
        self.tier_scores = {
            "physical": 0,
            "grammar":  0,
            "semantic": 0,
        }
    
    def add_message(self, who: str, text: str, grammar_type: str = ""):
        """Add a message to conversation history."""
        self.conv_history.append({
            "who": who,
            "text": text,
            "grammar_type": grammar_type
        })
    
    def clear_history(self):
        """Clear conversation history."""
        self.conv_history = []

    def register_user(self, full_name: str, email: str, password: str):
        """Register a new in-memory user. Returns (success, message)."""
        normalized_email = email.strip().lower()
        if normalized_email in self.users:
            return False, "An account with that email already exists."

        self.users[normalized_email] = {
            "full_name": full_name.strip(),
            "password": password,
        }
        self.current_user = {"email": normalized_email, **self.users[normalized_email]}
        return True, "Account created successfully."

    def authenticate_user(self, email: str, password: str):
        """Authenticate a user. Returns (success, message)."""
        normalized_email = email.strip().lower()
        user = self.users.get(normalized_email)
        if not user:
            return False, "No account found for that email. Create one first."
        if user["password"] != password:
            return False, "Incorrect password. Try again."

        self.current_user = {"email": normalized_email, **user}
        return True, f"Welcome back, {user['full_name'].split()[0]}."

    def logout(self):
        """Clear active user/session state."""
        self.current_user = None
        self.user_role = None
        self.session_id = None
